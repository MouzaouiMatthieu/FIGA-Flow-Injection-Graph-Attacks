import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import os
import copy
import dgl
from dgl import init as dgl_init
import matplotlib.pyplot as plt
import numpy as np
import json
import random

from src.utils.logger import setup_logging
from src.attacks.gradient_influence import GradientInfluenceCalculator
from src.attacks.graph_ops import ensure_node_capacity, inject_cover_flow
from src.attacks.feature_selectors import FeatureSelector

setup_logging()
logger = logging.getLogger(__name__)


class TargetedEvasionAttack:
    """
    Targeted evasion attack using Joint Selection (Virtual Edge Gradient Trick) + Surrogate Training.
    Supports checkpoint resume, lightweight mode, and optional side effects tracking.
    """

    def __init__(
        self,
        G,
        victim_model: torch.nn.Module,
        surrogate_models: Union[Dict[str, torch.nn.Module], List[torch.nn.Module], torch.nn.Module],
        feature_pool: Optional[torch.Tensor] = None,
        pool_indices: Optional[torch.Tensor] = None,
        pool_labels: Optional[List[int]] = None,
        device: Union[str, torch.device] = "cpu",
        label_benign: int = 0,
        label_malicious: int = 1,
        verbose: bool = True,
        pool_strategy: str = "auto",
        pool_mask: str = "test_mask",
        pool_k: int = 50,
        exclude_victim_as_dest: bool = False,
        separate_attackers: bool = False,
        exp_folder: Optional[str] = None,
        top_k: int = 10,
        pool_metadata: Optional[Dict] = None,
        test_mode: str = "test_only",
        split_config: str = "",
        feature_selection_mode: str = "best",
        lightweight: bool = False,
        track_side_effects: bool = False,
    ):
        self.victim_model = victim_model.to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.label_benign = label_benign
        self.label_malicious = label_malicious
        self.verbose = verbose
        self.exclude_victim_as_dest = exclude_victim_as_dest
        self.separate_attackers = separate_attackers
        self.exp_folder = exp_folder
        self.top_k = top_k
        self.history = []
        self.pool_metadata = pool_metadata or {}
        self.test_mode = test_mode
        self.split_config = split_config
        self.feature_selection_mode = feature_selection_mode
        self.lightweight = lightweight
        self.track_side_effects = track_side_effects

        # Normalise surrogates
        if isinstance(surrogate_models, dict):
            self.surrogate_models: List[torch.nn.Module] = list(surrogate_models.values())
        elif isinstance(surrogate_models, list):
            self.surrogate_models = surrogate_models
        elif surrogate_models is not None:
            self.surrogate_models = [surrogate_models]
        else:
            self.surrogate_models = []

        self.influence_calc = GradientInfluenceCalculator(self.surrogate_models, device=self.device)

        # Attack node IDs
        self.attacker_endpoint_id: int = -1
        self.malicious_flow_id: int = -1
        self.malicious_flow_destination: int = -1

        # Feature pool and selector
        self.feature_pool = feature_pool
        self.pool_indices = pool_indices
        self.pool_labels = pool_labels
        self.pool_strategy = pool_strategy
        self.pool_mask = pool_mask
        self.pool_k = pool_k

        self.feature_selector = None
        if self.feature_pool is not None:
            self.feature_selector = FeatureSelector(self.feature_pool, self.pool_labels)

        self.G = copy.deepcopy(G).to(self.device)

        # History for injected flows
        self.injected_flow_ids: List[int] = []

        # Side effects tracking (only if enabled)
        self.side_effects_history: List[dict] = []
        self.baseline_predictions: Optional[Dict[int, dict]] = None

        # Checkpoint file paths
        self.checkpoint_path = None
        self.metrics_path = None
        if self.exp_folder:
            self.checkpoint_path = os.path.join(self.exp_folder, "attack_checkpoint.json")
            self.metrics_path = os.path.join(self.exp_folder, "attack_metrics.json")

        if self.verbose and self.lightweight:
            logger.info("Lightweight mode enabled: injecting directly to victim endpoint (no influence calculation)")
        if self.verbose and self.track_side_effects:
            logger.info("Side effects tracking enabled (slower)")

    # ------------------------------------------------------------------
    # Prediction utilities for side effects (only used if tracking)
    # ------------------------------------------------------------------

    def _get_all_predictions(self, graph: dgl.DGLGraph) -> Dict[int, dict]:
        """Get predictions for ALL flow nodes in the graph."""
        self.victim_model.eval()
        feats = {
            'flow': graph.nodes['flow'].data['h'],
            'endpoint': graph.nodes['endpoint'].data['h'],
        }

        with torch.no_grad():
            logits = self.victim_model(graph, feats)
            if isinstance(logits, dict):
                logits = logits['flow']

            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        predictions = {}
        for flow_id in range(graph.num_nodes('flow')):
            predictions[int(flow_id)] = {
                'label': int(preds[flow_id].item()),
                'prob_malicious': float(probs[flow_id, self.label_malicious].item()),
                'prob_benign': float(probs[flow_id, self.label_benign].item()),
                'logits': logits[flow_id].cpu().tolist(),
            }
        return predictions

    def _compute_side_effects(self, step: int, before_preds: Dict[int, dict], after_preds: Dict[int, dict]) -> dict:
        """Compare predictions before and after injection to detect changes."""
        changed_flows = []
        label_flips = 0
        delta_probs = []

        injected_id = self.injected_flow_ids[-1] if self.injected_flow_ids else None
        neighbor_flow_ids = set()
        
        if injected_id is not None:
            endpoints = self.G.successors(injected_id, etype='depends_on')
            for ep in endpoints:
                connected_flows = self.G.in_edges(ep, etype='depends_on')[0]
                neighbor_flow_ids.update(connected_flows.cpu().tolist())
            neighbor_flow_ids.discard(injected_id)

        for flow_id in before_preds:
            before = before_preds[flow_id]
            after = after_preds[flow_id]

            if before['label'] != after['label']:
                label_flips += 1
                is_neighbor = flow_id in neighbor_flow_ids
                changed_flows.append({
                    'flow_id': flow_id,
                    'is_neighbor_of_injected': is_neighbor,
                    'old_label': before['label'],
                    'new_label': after['label'],
                    'old_prob_malicious': before['prob_malicious'],
                    'new_prob_malicious': after['prob_malicious'],
                    'delta_prob': after['prob_malicious'] - before['prob_malicious'],
                    'old_logits': before['logits'],
                    'new_logits': after['logits'],
                })
            elif abs(after['prob_malicious'] - before['prob_malicious']) > 1e-6:
                is_neighbor = flow_id in neighbor_flow_ids
                delta_probs.append(after['prob_malicious'] - before['prob_malicious'])
                if len(changed_flows) < 1000:
                    changed_flows.append({
                        'flow_id': flow_id,
                        'is_neighbor_of_injected': is_neighbor,
                        'old_label': before['label'],
                        'new_label': after['label'],
                        'old_prob_malicious': before['prob_malicious'],
                        'new_prob_malicious': after['prob_malicious'],
                        'delta_prob': after['prob_malicious'] - before['prob_malicious'],
                        'old_logits': before['logits'],
                        'new_logits': after['logits'],
                        'only_confidence_change': True,
                    })

        neighbor_changes = [f for f in changed_flows if f.get('is_neighbor_of_injected', False)]
        non_neighbor_changes = [f for f in changed_flows if not f.get('is_neighbor_of_injected', False)]

        result = {
            'step': step,
            'injected_flow_id': injected_id,
            'n_total_flows': len(before_preds),
            'n_label_flips': label_flips,
            'n_confidence_changes': len(changed_flows),
            'n_neighbor_changes': len(neighbor_changes),
            'n_non_neighbor_changes': len(non_neighbor_changes),
            'mean_delta_prob': float(np.mean(delta_probs)) if delta_probs else 0.0,
            'std_delta_prob': float(np.std(delta_probs)) if delta_probs else 0.0,
            'changed_flows': changed_flows,
            'neighbor_flow_ids': list(neighbor_flow_ids),
        }
        return result

    # ------------------------------------------------------------------
    # Checkpoint methods
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int) -> None:
        if not self.checkpoint_path:
            return

        checkpoint = {
            "step": step,
            "history": self.history,
            "side_effects_history": self.side_effects_history,
            "injected_flow_ids": self.injected_flow_ids,
            "attacker_endpoint_id": self.attacker_endpoint_id,
            "malicious_flow_id": self.malicious_flow_id,
            "malicious_flow_destination": self.malicious_flow_destination,
            "pool_strategy": self.pool_strategy,
            "test_mode": self.test_mode,
            "split_config": self.split_config,
            "feature_selection_mode": self.feature_selection_mode,
            "lightweight": self.lightweight,
            "track_side_effects": self.track_side_effects,
            "label_benign": self.label_benign,
            "label_malicious": self.label_malicious,
            "last_step_data": self.history[-1] if self.history else None,
        }

        if self.feature_pool is not None:
            checkpoint["feature_pool"] = self.feature_pool.cpu().tolist()
        if self.pool_indices is not None:
            checkpoint["pool_indices"] = self.pool_indices.cpu().tolist()

        try:
            with open(self.checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
            logger.info(f"Checkpoint saved at step {step} → {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> Tuple[bool, int]:
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return False, 0

        try:
            with open(self.checkpoint_path, "r") as f:
                checkpoint = json.load(f)

            self.history = checkpoint.get("history", [])
            self.side_effects_history = checkpoint.get("side_effects_history", [])
            self.injected_flow_ids = checkpoint.get("injected_flow_ids", [])
            self.attacker_endpoint_id = checkpoint.get("attacker_endpoint_id", -1)
            self.malicious_flow_id = checkpoint.get("malicious_flow_id", -1)
            self.malicious_flow_destination = checkpoint.get("malicious_flow_destination", -1)

            if "feature_pool" in checkpoint and checkpoint["feature_pool"] is not None:
                self.feature_pool = torch.tensor(checkpoint["feature_pool"]).to(self.device)
                self.feature_selector = FeatureSelector(self.feature_pool, self.pool_labels)
            if "pool_indices" in checkpoint and checkpoint["pool_indices"] is not None:
                self.pool_indices = torch.tensor(checkpoint["pool_indices"]).to(self.device)

            step = len(self.history) - 1
            logger.info(f"Loaded checkpoint: completed {step} steps, {len(self.injected_flow_ids)} flows injected")

            if step != len(self.injected_flow_ids):
                logger.warning(f"Checkpoint inconsistency: steps={step}, injected_flows={len(self.injected_flow_ids)}")
                step = min(step, len(self.injected_flow_ids))

            return True, step

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False, 0

    def reconstruct_graph_from_history(self, base_graph: dgl.DGLGraph) -> dgl.DGLGraph:
        G = copy.deepcopy(base_graph).to(self.device)

        for entry in self.history:
            if entry.get("step", 0) == 0:
                continue

            cover_flow_id = entry.get("cover_flow_id")
            if cover_flow_id is None:
                continue

            if cover_flow_id < G.num_nodes("flow"):
                continue

            target_endpoint = entry.get("target_endpoint")
            attacker_id = entry.get("attacker_endpoint_id")
            feature_idx = entry.get("feature_idx")

            if target_endpoint is None or attacker_id is None or feature_idx is None:
                continue

            if self.feature_pool is not None and feature_idx < len(self.feature_pool):
                features = self.feature_pool[feature_idx]
            else:
                continue

            inject_cover_flow(
                G,
                features,
                feature_idx,
                self.label_benign,
                self.label_malicious,
                attacker_id,
                target_endpoint,
                pool_indices=self.pool_indices,
                device=self.device,
                verbose=False,
            )

        logger.info(f"Reconstructed graph with {G.num_nodes('flow')} flows ({len(self.injected_flow_ids)} injected)")
        return G

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_attack(self, resume: bool = False) -> None:
        self.victim_model.eval()
        for model in self.surrogate_models:
            model.eval()

        self._configure_initializers(self.G)

        if self.feature_pool is None:
            raise RuntimeError("Feature pool not provided.")

        if isinstance(self.feature_pool, list):
            self.feature_pool = torch.stack(self.feature_pool)

        if isinstance(self.feature_pool, torch.Tensor):
            self.feature_pool = self.feature_pool.to(self.device)

        self.feature_selector = FeatureSelector(self.feature_pool, self.pool_labels)

        if self.verbose:
            logger.info(f"Attack initialized with feature pool size {len(self.feature_pool)}")
            if self.lightweight:
                logger.info("Lightweight mode: will inject directly to victim endpoint")
            if self.track_side_effects:
                logger.info("Side effects tracking enabled (will compute predictions for all flows)")

        # Store baseline predictions for side effects (only if tracking)
        if self.track_side_effects and self.baseline_predictions is None:
            self.baseline_predictions = self._get_all_predictions(self.G)

    def set_target(self, endpoint_id: int, flow_id: int, destination_id: int = None):
        self.attacker_endpoint_id = endpoint_id
        self.malicious_flow_id = flow_id
        if destination_id is not None:
            self.malicious_flow_destination = destination_id

        if self.G is None:
            self.setup_attack()

    # ------------------------------------------------------------------
    # Endpoint metrics
    # ------------------------------------------------------------------

    def _calculate_endpoint_metrics(self, endpoint_id: int) -> dict:
        if self.G is None or endpoint_id == -1:
            return {'degree': 0, 'purity': 0.0}

        try:
            connected_flows_in = self.G.in_edges(endpoint_id, etype='depends_on')
            if isinstance(connected_flows_in, tuple) or isinstance(connected_flows_in, list):
                try:
                    connected_flows_in = connected_flows_in[0]
                except Exception:
                    pass
            unique_flows = torch.unique(connected_flows_in)
            degree = len(unique_flows)

            if degree == 0:
                purity = 0.0
            else:
                flow_labels = self.G.nodes['flow'].data['label'][unique_flows]
                malicious_count = (flow_labels == self.label_malicious).sum().item()
                purity = malicious_count / degree

            return {'degree': degree, 'purity': purity}

        except Exception as e:
            logger.warning(f"Failed to calc metrics for ep {endpoint_id}: {e}")
            return {'degree': 0, 'purity': 0.0}

    # ------------------------------------------------------------------
    # Candidate set
    # ------------------------------------------------------------------

    def _build_candidate_set(self) -> List[int]:
        if self.G is None:
            raise RuntimeError("Graph not initialized.")

        ep_data = self.G.nodes['endpoint'].data
        num_eps = self.G.num_nodes('endpoint')

        mask = None
        if self.pool_mask and self.pool_mask in ep_data:
            mask = ep_data[self.pool_mask]
        elif 'test_mask' in ep_data:
            mask = ep_data['test_mask']
        elif 'val_mask' in ep_data:
            mask = ep_data['val_mask']

        if mask is None:
            valid_idx = torch.arange(num_eps, device=self.device)
        else:
            try:
                if not isinstance(mask, torch.Tensor):
                    mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)
                else:
                    mask_t = mask.bool().to(self.device)
                valid_idx = torch.nonzero(mask_t, as_tuple=False).squeeze(1)
            except Exception:
                valid_idx = torch.arange(num_eps, device=self.device)

        def apply_attr_filter(attr_name, keep_true=True):
            nonlocal valid_idx
            if attr_name in ep_data:
                try:
                    attr = ep_data[attr_name]
                    if not isinstance(attr, torch.Tensor):
                        attr = torch.tensor(attr, dtype=torch.bool, device=self.device)
                    else:
                        attr = attr.bool().to(self.device)
                    if keep_true:
                        mask_local = attr[valid_idx]
                    else:
                        mask_local = ~attr[valid_idx]
                    valid_idx = valid_idx[mask_local]
                except Exception:
                    pass

        apply_attr_filter('is_internal', keep_true=True)
        apply_attr_filter('is_destination', keep_true=True)
        if 'is_source' in ep_data:
            apply_attr_filter('is_source', keep_true=False)

        try:
            if self.attacker_endpoint_id != -1:
                valid_idx = valid_idx[valid_idx != int(self.attacker_endpoint_id)]
        except Exception:
            pass

        if self.exclude_victim_as_dest and self.malicious_flow_id != -1 and self.G is not None:
            try:
                victim_endpoints = self.G.successors(self.malicious_flow_id, etype='depends_on')
                for vep in victim_endpoints.tolist():
                    valid_idx = valid_idx[valid_idx != int(vep)]
            except Exception as exc:
                logger.debug(f"exclude_victim_as_dest: could not determine victim endpoint: {exc}")

        try:
            in_degs = self.G.in_degrees(etype='depends_on') if hasattr(self.G, 'in_degrees') else self.G.in_degrees()
            if isinstance(in_degs, torch.Tensor):
                deg_mask = in_degs.to(valid_idx.device)[valid_idx] > 0
                valid_idx = valid_idx[deg_mask]
        except Exception:
            pass

        try:
            return [int(x) for x in valid_idx.tolist()]
        except Exception:
            return [int(x) for x in valid_idx.cpu().numpy().tolist()]

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def _select_feature(self, gradient: torch.Tensor = None) -> Tuple[int, torch.Tensor]:
        if self.feature_selector is None:
            raise RuntimeError("Feature selector not initialized")

        if self.feature_selection_mode in ["best", "random_same_label"]:
            if gradient is None:
                logger.warning(f"Gradient needed for {self.feature_selection_mode}, falling back to random")
                return self.feature_selector.select(None, self.label_benign, "random")
            return self.feature_selector.select(gradient, self.label_benign, self.feature_selection_mode)
        else:
            return self.feature_selector.select(None, self.label_benign, self.feature_selection_mode)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def _compute_gradient(self) -> torch.Tensor:
        self.G.nodes['flow'].data['h'].requires_grad_(True)

        grads = []
        for model in self.surrogate_models:
            model.eval()
            logits = model(self.G)
            if isinstance(logits, dict):
                logits = logits['flow']

            log_probs = F.log_softmax(logits, dim=-1)
            loss = log_probs[self.malicious_flow_id, self.label_benign]

            if self.G.nodes['flow'].data['h'].grad is not None:
                self.G.nodes['flow'].data['h'].grad.zero_()

            loss.backward(retain_graph=True)
            grad = self.G.nodes['flow'].data['h'].grad[self.malicious_flow_id].detach().clone()
            grads.append(grad)

        self.G.nodes['flow'].data['h'].requires_grad_(False)

        if not grads:
            raise RuntimeError("No gradients computed")
        return torch.stack(grads).mean(dim=0)


    def run_one_step(self) -> dict:
        if self.G is None:
            self.setup_attack()
        assert self.G is not None
        assert self.feature_pool is not None

        if self.attacker_endpoint_id == -1 or self.malicious_flow_id == -1:
            raise ValueError("Target not set. Call set_target() first.")

        # Store predictions before injection for side effects (ONLY IF TRACKING)
        if self.track_side_effects:
            before_predictions = self._get_all_predictions(self.G)
        else:
            before_predictions = None

        prev_n_flows = self.G.num_nodes("flow")
        prev_n_dep = self.G.num_edges("depends_on")
        prev_n_lnk = self.G.num_edges("links_to")

        if self.lightweight:
            if self.malicious_flow_destination == -1:
                raise ValueError("Lightweight mode requires malicious_flow_destination to be set")

            target_endpoint = self.malicious_flow_destination
            # In lightweight mode we previously forced gradient=None which caused
            # the selector to fall back to random selection for 'best' mode.
            # Compute gradients when the selection mode benefits from them
            gradient = None
            if self.feature_selection_mode in ["best", "random_same_label"]:
                try:
                    # Use the existing influence calculator to compute gradients for all endpoints
                    # and extract the averaged gradient vector for the target endpoint.
                    sel_ret = self.influence_calc.select_endpoint(
                        self.G,
                        malicious_flow_id=self.malicious_flow_id,
                        attacker_endpoint_id=self.attacker_endpoint_id,
                        target_label=self.label_benign,
                        candidate_endpoints=[target_endpoint],
                        return_gradients=True,
                    )
                    # If gradients returned, sel_ret is (best_ep, best_norm, grad_norms, avg_gradient)
                    if isinstance(sel_ret, tuple) and len(sel_ret) == 4:
                        _, _, _, avg_gradient = sel_ret
                        try:
                            gradient = avg_gradient[int(target_endpoint)].detach().clone().to(self.device)
                        except Exception:
                            gradient = None
                    else:
                        gradient = None
                except Exception as e:
                    logger.warning(f"Could not compute endpoint gradients via influence_calc: {e} -- falling back to random selection")

            feature_idx, selected_features = self._select_feature(gradient=gradient)

            logger.info(f"[Lightweight] Injecting directly to victim endpoint {target_endpoint}, feature_idx={feature_idx}")

            cover_info = inject_cover_flow(
                self.G,
                selected_features,
                feature_idx,
                self.label_benign,
                self.label_malicious,
                self.attacker_endpoint_id,
                target_endpoint,
                pool_indices=self.pool_indices,
                device=self.device,
                verbose=self.verbose,
            )

            if 'cover_flow_id' in cover_info:
                self.injected_flow_ids.append(cover_info['cover_flow_id'])

            step_result = {
                'action': 'add_flow_lightweight',
                'target_endpoint': target_endpoint,
                'covert_destination_endpoint': target_endpoint,
                'attacker_endpoint_id': self.attacker_endpoint_id,
                'malicious_flow_id': self.malicious_flow_id,
                'feature_idx': feature_idx,
                'score': 0.0,
                'cover_info': cover_info,
            }
        else:
            candidate_endpoints = self._build_candidate_set()
            gradient = self._compute_gradient()
            feature_idx, selected_features = self._select_feature(gradient)

            best_e_rel_idx, best_f_idx, score, all_scores = self.influence_calc.compute_joint_influence(
                self.G,
                self.malicious_flow_id,
                self.attacker_endpoint_id,
                self.feature_pool,
                target_label=self.label_benign,
                candidate_endpoints=candidate_endpoints,
                selection_mode=self.feature_selection_mode,
                pool_labels=self.pool_labels,
            )

            target_endpoint = best_e_rel_idx

            _scores_arr = np.array(all_scores, dtype=np.float64)
            _valid_mask = np.isfinite(_scores_arr)
            _valid_ids = np.where(_valid_mask)[0]
            _valid_vals = _scores_arr[_valid_mask]
            _k = min(self.top_k, len(_valid_ids))

            top_k_influences = []
            if _k > 0:
                _top_indices = np.argsort(_valid_vals)[-_k:][::-1]
                top_k_influences = [
                    {'endpoint_id': int(_valid_ids[i]), 'score': float(_valid_vals[i])}
                    for i in _top_indices
                ]

            logger.info(f"Selected pair: Endpoint={target_endpoint}, feature_idx={best_f_idx}, Score={score}")
            logger.info(f"Top-{self.top_k} candidate influences: {top_k_influences}")
            
            ep_degrees = {}
            ep_purities = {}
            for eid in (candidate_endpoints if candidate_endpoints is not None else range(self.G.num_nodes('endpoint'))):
                m = self._calculate_endpoint_metrics(int(eid))
                ep_degrees[int(eid)] = m['degree']
                ep_purities[int(eid)] = m['purity']

            chosen_ep_metrics = self._calculate_endpoint_metrics(target_endpoint)

            sender_ep_id = self.attacker_endpoint_id
            if self.separate_attackers:
                sender_ep_id = self._inject_fresh_sender_endpoint()

            cover_info = self._add_cover_flow(target_endpoint, best_f_idx, sender_endpoint_id=sender_ep_id)

            step_result = {
                'action': 'add_flow',
                'target_endpoint': target_endpoint,
                'covert_destination_endpoint': target_endpoint,
                'attacker_endpoint_id': self.attacker_endpoint_id,
                'sender_endpoint_id': int(sender_ep_id) if sender_ep_id is not None else self.attacker_endpoint_id,
                'malicious_flow_id': self.malicious_flow_id,
                'feature_idx': best_f_idx,
                'score': float(score),
                'all_scores': all_scores,
                'top_k_influences': top_k_influences,
                'ep_degrees': ep_degrees,
                'ep_purities': ep_purities,
                'chosen_ep_degree': chosen_ep_metrics['degree'],
                'chosen_ep_purity': chosen_ep_metrics['purity'],
                'cover_info': cover_info,
            }

        # Integrity check
        curr_n_flows = self.G.num_nodes("flow")
        curr_n_dep = self.G.num_edges("depends_on")
        curr_n_lnk = self.G.num_edges("links_to")

        flow_diff = curr_n_flows - prev_n_flows
        edge_diff = (curr_n_dep + curr_n_lnk) - (prev_n_dep + prev_n_lnk)

        step_result['integrity_flow_diff'] = flow_diff
        step_result['integrity_edge_diff'] = edge_diff

        # Compute side effects (ONLY IF TRACKING)
        if self.track_side_effects:
            after_predictions = self._get_all_predictions(self.G)
            side_effects = self._compute_side_effects(len(self.injected_flow_ids), before_predictions, after_predictions)
            self.side_effects_history.append(side_effects)
            step_result['side_effects'] = side_effects
        else:
            step_result['side_effects'] = {}

        return step_result

    def _inject_fresh_sender_endpoint(self) -> int:
        ep_data = self.G.nodes['endpoint'].data
        ep_features = ep_data['h']
        new_ep_feats = ep_features.mean(dim=0, keepdim=True)

        self.G.add_nodes(1, ntype='endpoint')
        new_ep_id = self.G.num_nodes('endpoint') - 1
        self._ensure_node_capacity(self.G, 'endpoint', new_ep_id)

        self.G.nodes['endpoint'].data['h'][new_ep_id] = new_ep_feats.squeeze(0)

        for k, v in ep_data.items():
            if k == 'h':
                continue
            try:
                if v.dtype == torch.bool:
                    self.G.nodes['endpoint'].data[k][new_ep_id] = False
                else:
                    self.G.nodes['endpoint'].data[k][new_ep_id] = torch.zeros(
                        v.shape[1:], dtype=v.dtype, device=self.device
                    ) if v.ndim > 1 else torch.tensor(0, dtype=v.dtype, device=self.device)
            except Exception:
                pass

        if self.verbose:
            logger.info(f"Injected fresh sender endpoint id={new_ep_id}")

        return new_ep_id

    def _add_cover_flow(self, target_endpoint_id: int, feature_idx: int, sender_endpoint_id: Optional[int] = None):
        if self.G is None:
            raise RuntimeError("Graph G is not initialized.")
        if self.feature_pool is None:
            raise RuntimeError("Feature pool not initialized.")

        if sender_endpoint_id is None:
            sender_endpoint_id = self.attacker_endpoint_id

        selected_features = self.feature_pool[feature_idx]
        cover_info = inject_cover_flow(
            self.G,
            selected_features,
            feature_idx,
            self.label_benign,
            self.label_malicious,
            sender_endpoint_id,
            target_endpoint_id,
            pool_indices=self.pool_indices,
            device=self.device,
            verbose=self.verbose,
        )

        if 'cover_flow_id' in cover_info:
            self.injected_flow_ids.append(cover_info['cover_flow_id'])

        logger.info(
            f"[Attack] Cover flow {cover_info.get('cover_flow_id')} injected: sender={sender_endpoint_id}, target={target_endpoint_id}, pool_idx={feature_idx}"
        )

        return cover_info

    def _configure_initializers(self, graph: Optional[dgl.DGLHeteroGraph]):
        if graph is None:
            return
        for ntype in ('flow', 'endpoint'):
            try:
                graph.set_n_initializer(dgl_init.zero_initializer, ntype=ntype)
            except Exception:
                pass
        for etype in (('flow', 'depends_on', 'endpoint'), ('endpoint', 'links_to', 'flow')):
            try:
                graph.set_e_initializer(dgl_init.zero_initializer, etype=etype)
            except Exception:
                pass

    def _ensure_node_capacity(self, graph: Optional[dgl.DGLHeteroGraph], ntype: str, new_idx: int) -> None:
        if graph is None:
            return
        ensure_node_capacity(graph, ntype, new_idx)

    def _run_victim_on_graph(self, graph: dgl.DGLGraph, flow_id: int) -> Tuple[float, float, list, int]:
        feats = {
            'flow': graph.nodes['flow'].data['h'],
            'endpoint': graph.nodes['endpoint'].data['h'],
        }
        self.victim_model.eval()
        with torch.no_grad():
            with graph.local_scope():
                logits = self.victim_model(graph, feats)
            if isinstance(logits, dict):
                logits = logits['flow']
            mal_logits = logits[flow_id]
            probs = F.softmax(mal_logits, dim=0)
            prob = probs[self.label_malicious].item()
            logit = mal_logits[self.label_malicious].item()
            pred_label = int(torch.argmax(probs).item())
        return prob, logit, mal_logits.tolist(), pred_label

    def evaluate_success(self) -> dict:
        if self.malicious_flow_id == -1:
            return {
                'victim_prob': 0.0, 'surrogate_prob': 0.0,
                'victim_logit': 0.0, 'surrogate_logit': 0.0,
                'victim_logits': [], 'surrogate_logits': [],
                'surrogate_probs_per_model': [],
                'victim_pred_label': self.label_benign,
                'surrogate_pred_label': self.label_benign,
                'surrogate_pred_labels_per_model': [],
                'victim_prob_test': None, 'victim_logit_test': None, 'victim_logits_test': None,
                'victim_pred_label_test': self.label_benign,
            }

        feats = {
            'flow': self.G.nodes['flow'].data['h'],
            'endpoint': self.G.nodes['endpoint'].data['h'],
        }

        victim_prob, victim_logit, victim_logit_tensor, victim_pred_label = self._run_victim_on_graph(self.G, self.malicious_flow_id)

        surr_probs = []
        surr_logits_list = []
        surr_logit_tensors = []
        surr_full_logits_list = []
        surr_pred_labels = []

        with torch.no_grad():
            models = self.surrogate_models or [self.victim_model]
            for model in models:
                model.eval()
                with self.G.local_scope():
                    s_logits = model(self.G, feats)
                if isinstance(s_logits, dict):
                    s_logits = s_logits['flow']

                s_mal_logits = s_logits[self.malicious_flow_id]
                full_logits = s_mal_logits.tolist()
                surr_full_logits_list.append(full_logits)

                s_prob = F.softmax(s_mal_logits, dim=0)
                try:
                    surr_probs.append(s_prob[self.label_malicious].item())
                    surr_logits_list.append(s_mal_logits[self.label_malicious].item())
                    surr_logit_tensors.append(s_mal_logits.tolist())
                    surr_pred_labels.append(int(torch.argmax(s_prob).item()))
                except IndexError:
                    surr_probs.append(0.0)
                    surr_logits_list.append(0.0)
                    surr_logit_tensors.append([])
                    surr_pred_labels.append(self.label_benign)

        surrogate_prob = sum(surr_probs) / len(surr_probs) if surr_probs else 0.0
        surrogate_logit = sum(surr_logits_list) / len(surr_logits_list) if surr_logits_list else 0.0
        surrogate_pred_label = (
            int(round(sum(surr_pred_labels) / len(surr_pred_labels)))
            if surr_pred_labels else self.label_benign
        )
        avg_surr_tensor = (
            np.mean(surr_logit_tensors, axis=0).tolist()
            if surr_logit_tensors and surr_logit_tensors[0]
            else []
        )

        return {
            'victim_prob': victim_prob,
            'victim_logit': victim_logit,
            'victim_logits': victim_logit_tensor,
            'surrogate_prob': surrogate_prob,
            'surrogate_logit': surrogate_logit,
            'surrogate_logits': avg_surr_tensor,
            'surrogate_probs_per_model': surr_probs,
            'victim_pred_label': victim_pred_label,
            'surrogate_pred_label': surrogate_pred_label,
            'surrogate_pred_labels_per_model': surr_pred_labels,
            'surrogate_full_logits': surr_full_logits_list,
            'victim_prob_test': victim_prob,
            'victim_logit_test': victim_logit,
            'victim_logits_test': victim_logit_tensor,
            'victim_pred_label_test': victim_pred_label,
        }

    # ------------------------------------------------------------------
    # Main attack loop
    # ------------------------------------------------------------------

    def attack(self, malicious_flow_id: int, attacker_endpoint_id: int, budget: int = 10, destination_id: int = None) -> List[dict]:
        loaded, completed_steps = self.load_checkpoint()

        if loaded and completed_steps > 0:
            logger.info(f"Resuming attack from step {completed_steps}/{budget}")
            if self.G is None:
                self.setup_attack()
            base_graph = copy.deepcopy(self.G)
            self.G = self.reconstruct_graph_from_history(base_graph)

            if len(self.history) - 1 != len(self.injected_flow_ids):
                logger.warning("Inconsistency detected after reconstruction, starting fresh")
                loaded = False
                completed_steps = 0
                self.history = []
                self.side_effects_history = []
                self.injected_flow_ids = []
                self.set_target(attacker_endpoint_id, malicious_flow_id, destination_id)
                self.setup_attack()
        else:
            logger.info("Starting fresh attack")
            self.set_target(attacker_endpoint_id, malicious_flow_id, destination_id)
            if self.G is None:
                self.setup_attack()
            completed_steps = 0

        if completed_steps >= budget:
            logger.info(f"Attack already completed ({completed_steps}/{budget} steps).")
            return self.history

        init_flow_count = self.G.num_nodes('flow')
        init_edge_count_dep = self.G.num_edges('depends_on')
        init_edge_count_lnk = self.G.num_edges('links_to')

        forced_ep_metrics = self._calculate_endpoint_metrics(attacker_endpoint_id)
        res = self.evaluate_success()

        if not self.history:
            self.history.append({
                'step': 0,
                'target_endpoint': -1,
                'feature_idx': -1,
                'influence_score': 0.0,
                'chosen_ep_degree': -1,
                'chosen_ep_purity': -1.0,
                'forced_ep_degree': forced_ep_metrics['degree'],
                'forced_ep_purity': forced_ep_metrics['purity'],
                'victim_prob_malicious': float(res['victim_prob']),
                'victim_logit_malicious': float(res['victim_logit']),
                'victim_logits': res['victim_logits'],
                'victim_prob_malicious_test': res['victim_prob_test'],
                'victim_logit_malicious_test': res['victim_logit_test'],
                'victim_pred_label': int(res.get('victim_pred_label', self.label_benign)),
                'victim_pred_label_test': int(res.get('victim_pred_label_test', self.label_benign)),
                'surrogate_prob_malicious': float(res['surrogate_prob']),
                'surrogate_logit_malicious': float(res['surrogate_logit']),
                'surrogate_logits': res['surrogate_logits'],
                'surrogate_pred_label': int(res.get('surrogate_pred_label', self.label_benign)),
            })
            self.save_checkpoint(0)

            if int(res.get('victim_pred_label', self.label_malicious)) != self.label_malicious:
                logger.warning(
                    "Step 0 warning: victim already predicts the injected malicious flow as non-malicious "
                    f"(pred={res.get('victim_pred_label')}, expected malicious={self.label_malicious}, "
                    f"p_mal={res['victim_prob']:.6f})."
                )

        logger.info(
            f"Initial State - Victim(G_test): P={res['victim_prob']:.4f}/L={res['victim_logit']:.4f}"
            f" | Surrogate(G_test): P={res['surrogate_prob']:.4f}/L={res['surrogate_logit']:.4f}"
        )

        for i in range(completed_steps, budget):
            try:
                step_res = self.run_one_step()

                res = self.evaluate_success()
                forced_ep_metrics = self._calculate_endpoint_metrics(attacker_endpoint_id)

                cover_info = step_res.get('cover_info', {})
                victim_success = int(res.get('victim_pred_label', self.label_malicious)) != self.label_malicious
                surr_success = int(res.get('surrogate_pred_label', self.label_malicious)) != self.label_malicious

                history_entry = {
                    'step': i + 1,
                    'target_endpoint': int(step_res.get('target_endpoint', -1)),
                    'covert_destination_endpoint': int(step_res.get('covert_destination_endpoint', -1)),
                    'attacker_endpoint_id': int(step_res.get('attacker_endpoint_id', attacker_endpoint_id)),
                    'sender_endpoint_id': int(step_res.get('sender_endpoint_id', attacker_endpoint_id)),
                    'malicious_flow_id': int(step_res.get('malicious_flow_id', malicious_flow_id)),
                    'cover_flow_id': int(cover_info.get('cover_flow_id', -1)),
                    'feature_idx': int(step_res.get('feature_idx', -1)),
                    'cover_pool_idx': cover_info.get('cover_pool_idx', None),
                    'cover_source_flow_id': cover_info.get('cover_source_flow_id', None),
                    'influence_score': float(step_res.get('score', 0.0)),
                    'chosen_ep_degree': int(step_res.get('chosen_ep_degree', -1)),
                    'chosen_ep_purity': float(step_res.get('chosen_ep_purity', -1.0)),
                    'forced_ep_degree': forced_ep_metrics['degree'],
                    'forced_ep_purity': forced_ep_metrics['purity'],
                    'victim_prob_malicious': float(res['victim_prob']),
                    'victim_logit_malicious': float(res['victim_logit']),
                    'victim_logits': res['victim_logits'],
                    'victim_evasion_success': bool(victim_success),
                    'victim_prob_malicious_test': res['victim_prob_test'],
                    'victim_logit_malicious_test': res['victim_logit_test'],
                    'victim_pred_label': int(res.get('victim_pred_label', self.label_benign)),
                    'victim_pred_label_test': int(res.get('victim_pred_label_test', self.label_benign)),
                    'victim_evasion_success_test': victim_success,
                    'surrogate_prob_malicious': float(res['surrogate_prob']),
                    'surrogate_logit_malicious': float(res['surrogate_logit']),
                    'surrogate_logits': res['surrogate_logits'],
                    'surrogate_pred_label': int(res.get('surrogate_pred_label', self.label_benign)),
                    'surrogate_pred_labels_per_model': res.get('surrogate_pred_labels_per_model', []),
                    'surrogate_probs_per_model': res.get('surrogate_probs_per_model', []),
                    'surrogate_evasion_success': bool(surr_success),
                    'top_k_influences': step_res.get('top_k_influences', []),
                    'integrity_flow_diff': step_res.get('integrity_flow_diff', 1),
                    'integrity_edge_diff': step_res.get('integrity_edge_diff', 0),
                }

                # Add side effects summary only if tracking
                if self.track_side_effects:
                    history_entry['side_effects_summary'] = {
                        'n_label_flips': step_res.get('side_effects', {}).get('n_label_flips', 0),
                        'n_neighbor_changes': step_res.get('side_effects', {}).get('n_neighbor_changes', 0),
                        'n_non_neighbor_changes': step_res.get('side_effects', {}).get('n_non_neighbor_changes', 0),
                        'mean_delta_prob': step_res.get('side_effects', {}).get('mean_delta_prob', 0.0),
                    }

                self.history.append(history_entry)
                self.save_checkpoint(i + 1)

                if self.exp_folder:
                    try:
                        with open(self.metrics_path, 'w') as _f:
                            json.dump(self.history, _f, indent=2)
                    except Exception as _e:
                        logger.warning(f"Incremental save failed: {_e}")

                logger.info(
                    f"Step {i+1}/{budget}: Target EP {step_res.get('target_endpoint', 'N/A')}. "
                    f"Victim(G_test): P={res['victim_prob']:.4f}/L={res['victim_logit']:.4f} "
                    f"{'(SUCCESS)' if res['victim_prob'] < 0.5 else ''}"
                )

            except Exception as e:
                logger.exception(f"Attack step failed: {e}")
                self.save_checkpoint(i)
                break

        final_flow_added = self.G.num_nodes('flow') - init_flow_count
        final_edges_added = (self.G.num_edges('depends_on') + self.G.num_edges('links_to')) - (init_edge_count_dep + init_edge_count_lnk)
        logger.info(f"Attack Complete. Total added: {final_flow_added} flows, {final_edges_added} edges.")

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                os.remove(self.checkpoint_path)
                logger.info("Checkpoint removed after successful completion")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint: {e}")

        return self.history

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def plot_results(self, save_path: Optional[str] = None):
        if not self.history:
            logger.warning("No history to plot.")
            return

        steps = [h['step'] for h in self.history]
        v_probs = [h['victim_prob_malicious'] for h in self.history]
        s_probs = [h['surrogate_prob_malicious'] for h in self.history]

        v_margins = [1 - 2 * p for p in v_probs]
        s_margins = [1 - 2 * p for p in s_probs]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        ax1.plot(steps, v_probs, label='Victim (Malicious Prob)', marker='o', color='red')
        ax1.plot(steps, s_probs, label='Surrogate (Malicious Prob)', marker='x', color='blue', linestyle='--')
        ax1.axhline(y=0.5, color='gray', linestyle=':', label='Decision Boundary')
        ax1.set_xlabel('Added Covert Flows')
        ax1.set_ylabel('Probability (Malicious)')
        title = 'Attack Progress: Probability'
        if self.lightweight:
            title += ' (Lightweight Mode)'
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True)

        ax2 = axes[1]
        ax2.plot(steps, v_margins, label='Victim Margin', marker='o', color='green')
        ax2.plot(steps, s_margins, label='Surrogate Margin', marker='x', color='orange', linestyle='--')
        ax2.axhline(y=0.0, color='gray', linestyle=':', label='Decision Boundary')
        ax2.set_xlabel('Added Covert Flows')
        ax2.set_ylabel('Margin (Benign - Malicious)')
        ax2.set_title('Attack Progress: Margin')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    def save_metrics(self, save_path: str):
        if not self.history:
            logger.warning("No history to save.")
            return

        surrogate_names = []
        surrogate_layers = []
        for m in self.surrogate_models:
            surrogate_names.append(m.__class__.__name__)
            surrogate_layers.append(getattr(m, 'n_layers', getattr(m, 'num_layers', None)))

        metadata = {
            'attack_type': 'evasion',
            'attack_version': 'lightweight' if self.lightweight else 'influence_based',
            'nids_model': self.victim_model.__class__.__name__,
            'nids_layers': getattr(self.victim_model, 'n_layers', getattr(self.victim_model, 'num_layers', None)),
            'surrogate_models': surrogate_names,
            'surrogate_layers': surrogate_layers,
            'n_surrogates': len(self.surrogate_models),
            'pool_strategy': self.pool_strategy,
            'pool_size': self.feature_pool.shape[0] if self.feature_pool is not None else 0,
            'pool_metadata': self.pool_metadata,
            'test_mode': self.test_mode,
            'split_config': self.split_config,
            'feature_selection_mode': self.feature_selection_mode,
            'exclude_victim_as_dest': self.exclude_victim_as_dest,
            'separate_attackers': self.separate_attackers,
            'budget': len(self.history) - 1,
            'seed': getattr(self, 'seed', None),
            'surrogate_query_budget': getattr(self, 'surrogate_query_budget', None),
            'transfer_success': self.history[-1].get('victim_evasion_success_test', False) if self.history else False,
            'lightweight': self.lightweight,
            'track_side_effects': self.track_side_effects,
            'malicious_flow_destination': self.malicious_flow_destination,
        }

        compact = []
        for entry in self.history:
            compact_entry = {
                k: v for k, v in entry.items()
                if k not in {'grad_norms', 'cos_scores', 'cover_features', 'all_scores', 'victim_logits', 'surrogate_logits'}
            }
            compact.append(compact_entry)

        full_data = {
            'metadata': metadata,
            'history': compact,
            'side_effects': self.side_effects_history if self.track_side_effects else [],
        }

        try:
            with open(save_path, 'w') as f:
                json.dump(full_data, f, indent=2)
            logger.info(f"Attack metrics saved to {save_path}")
            logger.info(f"  Metadata: nids={metadata['nids_model']} L{metadata['nids_layers']}, "
                        f"surr={surrogate_names} L{surrogate_layers}, "
                        f"feat_sel={metadata['feature_selection_mode']}, "
                        f"lightweight={metadata['lightweight']}, "
                        f"track_side_effects={metadata['track_side_effects']}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")