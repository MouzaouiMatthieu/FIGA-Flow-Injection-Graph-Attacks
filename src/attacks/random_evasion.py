"""
Random Evasion Attack: Baseline that randomly selects destination endpoints and features.
No influence calculation, just random choices.
"""

import copy
import json
import logging
import os
import random
from typing import Dict, List, Optional, Union, Tuple

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.logger import setup_logging
from src.attacks.graph_ops import inject_cover_flow

setup_logging()
logger = logging.getLogger(__name__)


class RandomEvasionAttack:
    """
    Random evasion attack baseline.
    - Randomly selects destination endpoints from candidates
    - Randomly selects features from the feature pool
    - No surrogate model, no influence calculation
    """

    def __init__(
        self,
        G,
        victim_model: torch.nn.Module,
        feature_pool: Optional[torch.Tensor] = None,
        pool_indices: Optional[torch.Tensor] = None,
        pool_labels: Optional[List[int]] = None,
        device: Union[str, torch.device] = "cpu",
        label_benign: int = 0,
        label_malicious: int = 1,
        verbose: bool = True,
        pool_strategy: str = "random",
        pool_mask: str = "test_mask",
        exp_folder: Optional[str] = None,
        test_mode: str = "test_only",
        split_config: str = "",
        feature_selection_mode: str = "random",
        seed: int = 42,
    ):
        self.victim_model = victim_model.to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.label_benign = label_benign
        self.label_malicious = label_malicious
        self.verbose = verbose
        self.pool_strategy = pool_strategy
        self.pool_mask = pool_mask
        self.exp_folder = exp_folder
        self.test_mode = test_mode
        self.split_config = split_config
        self.feature_selection_mode = feature_selection_mode
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Feature pool
        self.feature_pool = feature_pool
        self.pool_indices = pool_indices
        
        # Attack node IDs
        self.attacker_endpoint_id: int = -1
        self.malicious_flow_id: int = -1
        
        self.G = copy.deepcopy(G).to(self.device)
        self.injected_flow_ids: List[int] = []
        self.history: List[dict] = []
        
        # Checkpoint paths
        self.checkpoint_path = None
        self.metrics_path = None
        if self.exp_folder:
            self.checkpoint_path = os.path.join(self.exp_folder, "attack_checkpoint.json")
            self.metrics_path = os.path.join(self.exp_folder, "attack_metrics.json")

    # ------------------------------------------------------------------
    # Checkpoint methods
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int) -> None:
        if not self.checkpoint_path:
            return
        checkpoint = {
            "step": step,
            "history": self.history,
            "injected_flow_ids": self.injected_flow_ids,
            "attacker_endpoint_id": self.attacker_endpoint_id,
            "malicious_flow_id": self.malicious_flow_id,
        }
        try:
            with open(self.checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
            logger.info(f"Checkpoint saved at step {step}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> Tuple[bool, int]:
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return False, 0
        try:
            with open(self.checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            self.history = checkpoint.get("history", [])
            self.injected_flow_ids = checkpoint.get("injected_flow_ids", [])
            self.attacker_endpoint_id = checkpoint.get("attacker_endpoint_id", -1)
            self.malicious_flow_id = checkpoint.get("malicious_flow_id", -1)
            step = len(self.history) - 1
            logger.info(f"Loaded checkpoint: completed {step} steps")
            return True, step
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False, 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_attack(self) -> None:
        self.victim_model.eval()
        
        if self.feature_pool is None:
            raise RuntimeError("Feature pool not provided.")
        
        if isinstance(self.feature_pool, list):
            self.feature_pool = torch.stack(self.feature_pool)
        self.feature_pool = self.feature_pool.to(self.device)
        
        if self.verbose:
            logger.info(f"Random evasion initialized with feature pool size {len(self.feature_pool)}")
            logger.info(f"  pool_strategy={self.pool_strategy}, seed={self.seed}")

    def set_target(self, endpoint_id: int, flow_id: int):
        self.attacker_endpoint_id = endpoint_id
        self.malicious_flow_id = flow_id

    # ------------------------------------------------------------------
    # Candidate set (same as targeted_evasion)
    # ------------------------------------------------------------------

    def _build_candidate_set(self) -> List[int]:
        ep_data = self.G.nodes['endpoint'].data
        num_eps = self.G.num_nodes('endpoint')
        
        mask = None
        if self.pool_mask and self.pool_mask in ep_data:
            mask = ep_data[self.pool_mask]
        elif 'test_mask' in ep_data:
            mask = ep_data['test_mask']
        
        if mask is None:
            valid_idx = torch.arange(num_eps, device=self.device)
        else:
            mask_t = mask.bool().to(self.device)
            valid_idx = torch.nonzero(mask_t, as_tuple=False).squeeze(1)
        
        def apply_filter(attr_name, keep_true=True):
            nonlocal valid_idx
            if attr_name in ep_data:
                attr = ep_data[attr_name].bool().to(self.device)
                mask_local = attr[valid_idx] if keep_true else ~attr[valid_idx]
                valid_idx = valid_idx[mask_local]
        
        apply_filter('is_internal', keep_true=True)
        apply_filter('is_destination', keep_true=True)
        apply_filter('is_source', keep_true=False)
        
        if self.attacker_endpoint_id != -1:
            valid_idx = valid_idx[valid_idx != self.attacker_endpoint_id]
        
        return valid_idx.tolist()

    # ------------------------------------------------------------------
    # Random selection
    # ------------------------------------------------------------------

    def _random_select_endpoint(self, candidates: List[int]) -> int:
        """Randomly select a destination endpoint from candidates."""
        return random.choice(candidates)
    
    def _random_select_feature(self) -> Tuple[int, torch.Tensor]:
        """Randomly select a feature from the pool."""
        idx = random.randint(0, len(self.feature_pool) - 1)
        features = self.feature_pool[idx]
        if self.pool_strategy == "centroid" and hasattr(self, '_centroid_features'):
            # Use centroid if requested (pre-computed)
            idx = 0  # Use centroid
            features = self._centroid_features
        return idx, features
    
    def _compute_centroid(self) -> torch.Tensor:
        """Pre-compute centroid of feature pool if needed."""
        if self.pool_strategy == "centroid":
            self._centroid_features = self.feature_pool.mean(dim=0)
            logger.info(f"Using centroid feature (strategy=centroid)")
            return self._centroid_features
        return None

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def run_one_step(self) -> dict:
        """Execute one random attack step."""
        candidates = self._build_candidate_set()
        if not candidates:
            raise RuntimeError("No valid victim endpoints found.")
        
        # Random selections
        target_ep = self._random_select_endpoint(candidates)
        feature_idx, features = self._random_select_feature()
        
        # Inject the flow
        cover_info = inject_cover_flow(
            self.G,
            features,
            feature_idx,
            self.label_benign,
            self.label_malicious,
            self.attacker_endpoint_id,
            target_ep,
            pool_indices=self.pool_indices,
            device=self.device,
            verbose=self.verbose,
        )
        
        if 'cover_flow_id' in cover_info:
            self.injected_flow_ids.append(cover_info['cover_flow_id'])
        
        logger.info(f"Random step: target={target_ep}, feature_idx={feature_idx}")
        
        return {
            'action': 'add_flow',
            'target_endpoint': target_ep,
            'attacker_endpoint_id': self.attacker_endpoint_id,
            'malicious_flow_id': self.malicious_flow_id,
            'feature_idx': feature_idx,
            'score': 0.0,  # No score for random
            'cover_info': cover_info,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_success(self) -> dict:
        """Check classification of the malicious flow."""
        if self.malicious_flow_id == -1:
            return {'victim_prob': 0.0, 'victim_logit': 0.0}
        
        feats = {
            'flow': self.G.nodes['flow'].data['h'],
            'endpoint': self.G.nodes['endpoint'].data['h'],
        }
        
        self.victim_model.eval()
        with torch.no_grad():
            logits = self.victim_model(self.G, feats)
            if isinstance(logits, dict):
                logits = logits['flow']
            mal_logits = logits[self.malicious_flow_id]
            prob = F.softmax(mal_logits, dim=0)[self.label_malicious].item()
            logit = mal_logits[self.label_malicious].item()
        
        return {
            'victim_prob': prob,
            'victim_logit': logit,
            'victim_success': prob < 0.5,
        }

    # ------------------------------------------------------------------
    # Main attack loop
    # ------------------------------------------------------------------

    def attack(self, malicious_flow_id: int, attacker_endpoint_id: int, budget: int = 10) -> List[dict]:
        """Run random attack."""
        loaded, completed_steps = self.load_checkpoint()
        
        if loaded and completed_steps > 0:
            logger.info(f"Resuming random attack from step {completed_steps}/{budget}")
            # Need to reconstruct graph from history
            base_graph = copy.deepcopy(self.G)
            self.G = self._reconstruct_graph_from_history(base_graph)
        else:
            logger.info("Starting fresh random attack")
            self.set_target(attacker_endpoint_id, malicious_flow_id)
            self.setup_attack()
            self._compute_centroid()  # Pre-compute centroid if needed
            completed_steps = 0
        
        if completed_steps >= budget:
            logger.info(f"Attack already completed ({completed_steps}/{budget} steps).")
            return self.history
        
        res = self.evaluate_success()
        if not self.history:
            self.history.append({
                'step': 0,
                'victim_prob_malicious': res['victim_prob'],
                'victim_logit_malicious': res['victim_logit'],
                'victim_success': res['victim_success'],
            })
            self.save_checkpoint(0)
        
        for i in range(completed_steps, budget):
            try:
                step_res = self.run_one_step()
                res = self.evaluate_success()
                
                self.history.append({
                    'step': i + 1,
                    'target_endpoint': step_res['target_endpoint'],
                    'feature_idx': step_res['feature_idx'],
                    'victim_prob_malicious': res['victim_prob'],
                    'victim_logit_malicious': res['victim_logit'],
                    'victim_success': res['victim_success'],
                })
                
                self.save_checkpoint(i + 1)
                
                if self.exp_folder:
                    try:
                        with open(self.metrics_path, 'w') as f:
                            json.dump(self.history, f, indent=2)
                    except Exception as e:
                        logger.warning(f"Incremental save failed: {e}")
                
                logger.info(f"Step {i+1}/{budget}: victim_prob={res['victim_prob']:.4f} "
                           f"{'(SUCCESS)' if res['victim_success'] else ''}")
                
            except Exception as e:
                logger.exception(f"Attack step failed: {e}")
                break
        
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                os.remove(self.checkpoint_path)
            except Exception:
                pass
        
        return self.history
    
    def _reconstruct_graph_from_history(self, base_graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """Reconstruct graph by replaying injected flows."""
        G = copy.deepcopy(base_graph).to(self.device)
        
        for entry in self.history:
            if entry.get('step', 0) == 0:
                continue
            
            target_ep = entry.get('target_endpoint')
            feature_idx = entry.get('feature_idx')
            
            if target_ep is None or feature_idx is None:
                continue
            
            if feature_idx < len(self.feature_pool):
                features = self.feature_pool[feature_idx]
            else:
                continue
            
            inject_cover_flow(
                G, features, feature_idx,
                self.label_benign, self.label_malicious,
                self.attacker_endpoint_id, target_ep,
                pool_indices=self.pool_indices,
                device=self.device,
                verbose=False,
            )
        
        return G

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def plot_results(self, save_path: Optional[str] = None):
        if not self.history:
            return
        
        steps = [h['step'] for h in self.history]
        probs = [h['victim_prob_malicious'] for h in self.history]
        
        plt.figure(figsize=(8, 5))
        plt.plot(steps, probs, marker='o', color='red', label='Victim (Malicious Prob)')
        plt.axhline(y=0.5, color='gray', linestyle=':', label='Decision Boundary')
        plt.xlabel('Added Covert Flows')
        plt.ylabel('Probability (Malicious)')
        plt.title(f'Random Evasion Attack (pool_strategy={self.pool_strategy})')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def save_metrics(self, save_path: str):
        if not self.history:
            return
        
        metadata = {
            'attack_type': 'evasion',
            'attack_version': 'random_baseline',
            'nids_model': self.victim_model.__class__.__name__,
            'nids_layers': getattr(self.victim_model, 'n_layers', None),
            'pool_strategy': self.pool_strategy,
            'pool_size': len(self.feature_pool) if self.feature_pool is not None else 0,
            'test_mode': self.test_mode,
            'split_config': self.split_config,
            'seed': self.seed,
        }
        
        full_data = {'metadata': metadata, 'history': self.history}
        
        with open(save_path, 'w') as f:
            json.dump(full_data, f, indent=2)
        logger.info(f"Random evasion metrics saved to {save_path}")