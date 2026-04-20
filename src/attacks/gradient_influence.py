import copy
import torch
import torch.nn.functional as F
import logging
import random
from typing import List, Optional, Tuple
import dgl

logger = logging.getLogger(__name__)


class GradientInfluenceCalculator:
    """Two-phase gradient-based influence for joint (WHERE, WHAT) selection.
    
    Phase 1 — WHERE: ∂L/∂h_endpoint → argmax ||grad||
    Phase 2 — WHAT: ∂L/∂h_flow (virtual) + cosine similarity → best feature
    """

    def __init__(self, surrogate_models: List[torch.nn.Module], device: torch.device = None):
        self.surrogate_models = surrogate_models
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _compute_loss(self, logits: torch.Tensor, target_idx: int, target_label: int = 0):
        """Compute loss: maximize logit_benign - logit_malicious for target_label=0."""
        if target_label == 0:  # Want benign classification (evasion)
            # Maximize (logit_benign - logit_malicious) → minimize negative of that
            return -(logits[target_idx, 0] - logits[target_idx, 1])
        else:  # Want malicious classification
            return -(logits[target_idx, 1] - logits[target_idx, 0])

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1 — WHERE (endpoint selection)
    # ─────────────────────────────────────────────────────────────────────────

    def select_endpoint(
        self,
        G: dgl.DGLGraph,
        malicious_flow_id: int,
        attacker_endpoint_id: int,
        target_label: int = 0,
        candidate_endpoints: Optional[List[int]] = None,
        return_gradients: bool = False,
    ) -> Tuple[int, float, List[float]]:
        """Phase 1: select endpoint using gradient w.r.t. endpoint FEATURES."""
        num_endpoints = G.num_nodes("endpoint")

        if candidate_endpoints is None:
            ep_set = set(range(num_endpoints))
        else:
            ep_set = {int(x) for x in candidate_endpoints}
        ep_set.discard(int(attacker_endpoint_id))

        if not ep_set:
            logger.warning("No candidate endpoints")
            return -1, float("-inf"), [0.0] * num_endpoints

        # Store original features
        orig_flow = G.nodes['flow'].data['h'].detach().clone()
        orig_endpoint = G.nodes['endpoint'].data['h'].detach().clone()

        all_gradients = []
        
        try:
            for model in self.surrogate_models:
                model.eval()
                model.zero_grad()
                
                endpoint_feats = orig_endpoint.clone().detach().requires_grad_(True)
                flow_feats = orig_flow.clone().detach().requires_grad_(True)
                
                G.nodes['flow'].data['h'] = flow_feats
                G.nodes['endpoint'].data['h'] = endpoint_feats
                
                try:
                    with torch.set_grad_enabled(True):
                        logits = model(G)
                            
                except Exception as e:
                    logger.warning(f"Surrogate forward failed: {e}")
                    continue

                if isinstance(logits, dict):
                    logits = logits["flow"]
                if malicious_flow_id >= logits.shape[0]:
                    continue

                loss = self._compute_loss(logits, malicious_flow_id, target_label)
                
                if not loss.requires_grad:
                    logger.warning(f"Loss doesn't require grad for {model.__class__.__name__}")
                
                loss.backward(retain_graph=True)
                
                if endpoint_feats.grad is not None:
                    all_gradients.append(endpoint_feats.grad.clone())
                else:
                    logger.warning("Input gradient is None after backward")
                        
                model.zero_grad()
                                
        finally:
            G.nodes['flow'].data['h'] = orig_flow
            G.nodes['endpoint'].data['h'] = orig_endpoint

        if not all_gradients:
            logger.warning("No valid gradients from any surrogate")
            first_ep = next(iter(ep_set))
            return first_ep, 0.0, [0.0] * num_endpoints

        # Average gradients across surrogates
        avg_gradient = torch.stack(all_gradients).mean(dim=0)
        grad_norm = avg_gradient.norm(dim=1)

        mask = torch.zeros(num_endpoints, dtype=torch.bool, device=grad_norm.device)
        for ep_id in ep_set:
            if 0 <= ep_id < num_endpoints:
                mask[ep_id] = True

        grad_norm_masked = grad_norm.clone()
        grad_norm_masked[~mask] = -float("inf")

        best_ep = int(grad_norm_masked.argmax().item())
        best_norm = float(grad_norm[best_ep].item())

        logger.debug("Phase 1 — best_ep=%d  grad_norm=%.6f", best_ep, best_norm)
        if return_gradients:
            return best_ep, best_norm, grad_norm.tolist(), avg_gradient
        else:
            return best_ep, best_norm, grad_norm.tolist()

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2 — WHAT (feature selection) - MODIFIED with random_same_label
    # ─────────────────────────────────────────────────────────────────────────

    def select_feature(
        self,
        G: dgl.DGLGraph,
        malicious_flow_id: int,
        attacker_endpoint_id: int,
        best_endpoint_id: int,
        candidate_features: torch.Tensor,
        target_label: int = 0,
        selection_mode: str = "best",
        pool_labels: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Phase 2: select best feature using gradient w.r.t. virtual flow features.
        
        Args:
            selection_mode: "best", "worst_positive", "random", or "random_same_label"
            pool_labels: Labels for each feature in pool (required for random_same_label)
            target_label: Label we want the flow to be classified as (0=benign, 1=malicious)
        """
        import random
        
        C = candidate_features.shape[0]
        
        # Handle random selection modes
        if selection_mode == "random":
            best_c_idx = random.randint(0, C - 1)
            logger.debug(f"Phase 2 (random) — best_feature={best_c_idx}")
            return best_c_idx, 0.0
        
        elif selection_mode == "random_same_label":
            if pool_labels is None:
                logger.warning("pool_labels not available for random_same_label, falling back to random")
                best_c_idx = random.randint(0, C - 1)
                return best_c_idx, 0.0
            
            # Filter features with the same label as target_label
            same_label_indices = [i for i, lbl in enumerate(pool_labels) if lbl == target_label]
            if not same_label_indices:
                logger.warning(f"No features with label {target_label} in pool, falling back to random")
                best_c_idx = random.randint(0, C - 1)
            else:
                best_c_idx = random.choice(same_label_indices)
            
            logger.debug(f"Phase 2 (random_same_label) — best_feature={best_c_idx} (label={target_label})")
            return best_c_idx, 0.0
        
        # For best and worst_positive, we need gradient computation
        feat_dim = G.nodes["flow"].data["h"].shape[1]
        candidate_features = candidate_features.to(self.device)

        # Create temporary graph with virtual flow node
        G_tmp = copy.deepcopy(G)
        G_tmp.add_nodes(1, ntype="flow")
        virtual_flow_id = G_tmp.num_nodes("flow") - 1

        # Initialize virtual flow features with zeros (will be differentiated)
        virt_feat = torch.zeros(1, feat_dim, device=self.device, requires_grad=True)

        # Detach existing features
        flow_feats_base = G.nodes["flow"].data["h"].detach().clone()
        flow_feats = torch.cat([flow_feats_base, virt_feat], dim=0)
        endpoint_feats = G.nodes["endpoint"].data["h"].detach().clone()

        # Connect virtual flow
        g_vf = torch.tensor([virtual_flow_id], device=self.device)
        e_atk = torch.tensor([attacker_endpoint_id], device=self.device)
        e_dst = torch.tensor([best_endpoint_id], device=self.device)

        G_tmp.add_edges(g_vf, e_dst, etype="depends_on")
        G_tmp.add_edges(e_dst, g_vf, etype="links_to")
        G_tmp.add_edges(g_vf, e_atk, etype="depends_on")
        G_tmp.add_edges(e_atk, g_vf, etype="links_to")

        try:
            G_tmp.nodes["flow"].data["h"] = flow_feats
            G_tmp.nodes["endpoint"].data["h"] = endpoint_feats

            model = self.surrogate_models[0]
            model.eval()
            model.zero_grad()

            logits = model(G_tmp)
            if isinstance(logits, dict):
                logits = logits["flow"]

            loss = self._compute_loss(logits, virtual_flow_id, target_label)
            
            loss.backward(retain_graph=True)
            
            if virt_feat.grad is not None:
                grad_dir = virt_feat.grad.squeeze(0)
            else:
                logger.warning("Virtual flow gradient is None after backward")
                return random.randint(0, C - 1), 0.0

            # We want to decrease loss → move in negative gradient direction
            target_dir = -grad_dir

            # Find most similar feature in pool
            cos_sim = F.cosine_similarity(
                candidate_features,
                target_dir.unsqueeze(0).expand(C, -1),
                dim=1,
            )
            
            # Filter by same label if needed (for best/worst_positive)
            if pool_labels is not None and selection_mode in ["best", "worst_positive"]:
                same_label_mask = torch.tensor([lbl == target_label for lbl in pool_labels],
                                               device=cos_sim.device)
                if same_label_mask.any():
                    cos_sim = cos_sim.clone()
                    cos_sim[~same_label_mask] = -float('inf')
                else:
                    logger.warning(f"No features with label {target_label} in pool for {selection_mode}")
            
            if selection_mode == "best":
                best_c_idx = int(cos_sim.argmax().item())
                best_score = float(cos_sim[best_c_idx].item())
                logger.debug(f"Phase 2 (best) — best_feature={best_c_idx}  cosine_sim={best_score:.6f}")
            elif selection_mode == "worst_positive":
                # Filter positive similarities
                positive_mask = (cos_sim > 0) & torch.isfinite(cos_sim)
                if positive_mask.any():
                    positive_sims = cos_sim[positive_mask]
                    worst_positive_idx = int(torch.argmin(positive_sims).item())
                    positive_indices = torch.where(positive_mask)[0]
                    best_c_idx = int(positive_indices[worst_positive_idx].item())
                    best_score = float(cos_sim[best_c_idx].item())
                    logger.debug(f"Phase 2 (worst_positive) — best_feature={best_c_idx}  cosine_sim={best_score:.6f} (min positive)")
                else:
                    # Fallback to best if no positive similarities
                    best_c_idx = int(cos_sim.argmax().item())
                    best_score = float(cos_sim[best_c_idx].item())
                    logger.warning(f"No positive similarities found, falling back to best: {best_c_idx} (score={best_score:.6f})")
            else:
                raise ValueError(f"Unknown selection_mode: {selection_mode}")
            
            return best_c_idx, best_score

        finally:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Combined method for evasion
    # ─────────────────────────────────────────────────────────────────────────

    def compute_joint_influence(
        self,
        G: dgl.DGLGraph,
        malicious_flow_id: int,
        attacker_endpoint_id: int,
        candidate_features: torch.Tensor,
        target_label: int = 0,
        candidate_endpoints: Optional[List[int]] = None,
        selection_mode: str = "best",
        pool_labels: Optional[List[int]] = None,
    ) -> Tuple[int, int, float, List[float]]:
        """Two-phase gradient-guided (endpoint, feature) selection.
        
        Args:
            selection_mode: "best", "worst_positive", "random", or "random_same_label"
            pool_labels: Labels for each feature (required for random_same_label)
        
        Returns:
            (best_ep, best_c, cos_score, per_ep_grad_norms)
        """
        
        # Phase 1 — WHERE
        best_ep, _, per_ep_grad_norms = self.select_endpoint(
            G,
            malicious_flow_id=malicious_flow_id,
            attacker_endpoint_id=attacker_endpoint_id,
            target_label=target_label,
            candidate_endpoints=candidate_endpoints,
        )

        if best_ep == -1:
            raise ValueError("Phase 1 failed to select a valid endpoint")

        # Phase 2 — WHAT with selection mode
        best_c, cos_score = self.select_feature(
            G,
            malicious_flow_id=malicious_flow_id,
            attacker_endpoint_id=attacker_endpoint_id,
            best_endpoint_id=best_ep,
            candidate_features=candidate_features,
            target_label=target_label,
            selection_mode=selection_mode,
            pool_labels=pool_labels,
        )

        return best_ep, best_c, cos_score, per_ep_grad_norms
