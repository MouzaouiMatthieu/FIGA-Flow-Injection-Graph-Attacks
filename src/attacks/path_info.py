from dataclasses import dataclass
from typing import List


@dataclass
class PathInfo:
    """Information about a single path in influence computation."""

    nodes: List[int]
    node_types: List[str]
    length: int
    edge_weights: List[float]
    path_weight: float
    terminal_node: int
    terminal_label: int
    label_weight: float
    contribution: float
    contributes_to: str

    def __str__(self) -> str:
        node_str = " -> ".join(
            [f"{node_id}({node_type})" for node_id, node_type in zip(self.nodes, self.node_types)]
        )
        weights_str = ", ".join([f"{weight:.4f}" for weight in self.edge_weights])
        return (
            f"Path: {node_str}\n"
            f"  Weights: [{weights_str}]\n"
            f"  Path weight: {self.path_weight:.6f}\n"
            f"  Terminal: node {self.terminal_node} (label={self.terminal_label})\n"
            f"  Label weight: {self.label_weight:+.0f}\n"
                        f"  Contribution: {self.contribution:+.6f} -> {self.contributes_to}"
        )
