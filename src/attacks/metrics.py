import torch
import logging
import math
import warnings
import contextlib
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)
def compute_margin_evasion_attack(G, model,target_flow_id, device, label_inserted_flow, label_evasion_flow):
    model = model.to(device)
    with torch.no_grad():
        out_target = model(G.to(device))
        if isinstance(out_target, dict):
            out_target = out_target["flow"]
        out_target = out_target[target_flow_id].squeeze()
        # margin = prob(target_class) - prob(source_class)
        # We want margin to increase (become > 0) to succeed
        diff = out_target[label_inserted_flow] - out_target[label_evasion_flow]
        return diff.item()
        
        