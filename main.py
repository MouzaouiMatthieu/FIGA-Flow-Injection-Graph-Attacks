import argparse
import json
from pathlib import Path
from typing import Any, Dict

ALLOWED_DATASETS = ("CICIDS2017", "X-IIoTID")
ALLOWED_MODELS = ("SAGE", "GCN", "GAT")
ALLOWED_ATTACKS = ("targeted_evasion", "random_evasion")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evasion attacks on heterogeneous GNN NIDS models.")
    parser.add_argument("--dataset", choices=ALLOWED_DATASETS, required=True)
    parser.add_argument(
        "--path",
        "--dataset_path",
        dest="path",
        type=str,
        default=None,
        help="Dataset root folder (must contain at least raw/ and processed/)",
    )
    parser.add_argument("--classes", choices=["binary", "category"], default="binary")
    parser.add_argument("--attack", choices=ALLOWED_ATTACKS, required=True)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        choices=[42],
        help="Random seed is fixed to 42.",
    )

    parser.add_argument("--target_model", choices=ALLOWED_MODELS, required=True)
    parser.add_argument("--target_model_path", type=str, required=True)
    parser.add_argument("--surrogate_models", nargs="+", choices=ALLOWED_MODELS, default=["SAGE"])
    parser.add_argument("--surrogate_model_paths", nargs="+", default=[])

    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--aggr_type", choices=["mean", "max", "pool", "lstm"], default="mean")
    parser.add_argument("--feature_pool_strategy", choices=["centroid", "clustering", "random", "all"], default="centroid")
    parser.add_argument("--feature_pool_k", type=int, default=None)

    parser.add_argument("--attacker_endpoint_id", type=int, default=0)
    parser.add_argument("--malicious_flow_id", type=int, default=None)
    parser.add_argument("--destination_endpoint_id", type=int, default=None)

    parser.add_argument(
        "--train-frac",
        type=float,
        default=50.0,
        help="Train split size (percent like 50, or fraction like 0.5)",
    )
    parser.add_argument(
        "--surr-frac",
        type=float,
        default=30.0,
        help="Surrogate split size (percent like 30, or fraction like 0.3)",
    )

    parser.add_argument("--output_dir", type=str, default="results/evasion")
    return parser.parse_args(argv)


def _validate_dataset_root(dataset_root: str | None) -> None:
    if dataset_root is None:
        return

    root_path = Path(dataset_root)
    if root_path.suffix in {".pt", ".pth", ".mth"}:
        raise ValueError(
            f"Dataset path must be a folder containing raw/ and processed/, not a file: {dataset_root}"
        )
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"Dataset path does not exist or is not a directory: {dataset_root}")

    missing = [name for name in ("raw", "processed") if not (root_path / name).is_dir()]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Dataset path must contain subfolders raw/ and processed/. Missing: {missing_str}"
        )


def _resolve_checkpoint_path(model_path: str) -> str:
    candidate_path = Path(model_path)

    if candidate_path.is_file():
        return str(candidate_path)

    if not candidate_path.exists() or not candidate_path.is_dir():
        raise ValueError(f"Model path does not exist: {model_path}")

    preferred_files = ["model.pth", "model.pt", "model.mth"]
    for filename in preferred_files:
        resolved = candidate_path / filename
        if resolved.is_file():
            return str(resolved)

    checkpoint_candidates = sorted(
        [
            path
            for extension in ("*.pth", "*.pt", "*.mth")
            for path in candidate_path.glob(extension)
            if path.is_file()
        ]
    )
    if checkpoint_candidates:
        return str(checkpoint_candidates[0])

    raise ValueError(
        f"No checkpoint found in model directory: {model_path}. "
        f"Expected model.pth/model.pt/model.mth or any *.pth/*.pt/*.mth file."
    )


def _normalize_split_fraction(value: float, argument_name: str) -> float:
    if value <= 0:
        raise ValueError(f"{argument_name} must be positive, got {value}")

    normalized = value / 100.0 if value > 1 else value

    if normalized <= 0 or normalized >= 1:
        raise ValueError(
            f"{argument_name} must be in (0,1) or (0,100), got {value}"
        )
    return normalized


def _build_model(model_name: str, in_feats: int, n_classes: int, n_layers: int, num_heads: int, aggr_type: str) -> Any:
    from src.models import HeteroGAT, HeteroGCN, HeteroGraphSAGE

    if model_name == "SAGE":
        return HeteroGraphSAGE(in_feats, 120, 128, n_classes, n_layers=n_layers, aggr_type=aggr_type)
    if model_name == "GCN":
        return HeteroGCN(in_feats, 120, 128, n_classes, n_layers=n_layers)
    if model_name == "GAT":
        return HeteroGAT(in_feats, 120, 128, num_heads=num_heads, num_classes=n_classes, n_layers=n_layers)
    raise ValueError(f"Unsupported model: {model_name}")


def _load_model(
    model_name: str,
    model_path: str,
    in_feats: int,
    n_classes: int,
    n_layers: int,
    num_heads: int,
    aggr_type: str,
    device: Any,
) -> Any:
    import torch

    resolved_model_path = _resolve_checkpoint_path(model_path)
    loaded = torch.load(resolved_model_path, map_location=device)
    if isinstance(loaded, torch.nn.Module):
        model = loaded.to(device)
        model.eval()
        return model

    model = _build_model(model_name, in_feats, n_classes, n_layers, num_heads, aggr_type).to(device)
    if isinstance(loaded, dict):
        state_dict = loaded.get("state_dict", loaded)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported checkpoint format at {resolved_model_path}")
    model.eval()
    return model


def _infer_malicious_flow_id(graph, fallback_label: int = 1) -> int:
    import torch

    labels = graph.nodes["flow"].data["label"]
    if labels.ndim > 1:
        labels = labels.argmax(dim=1)
    test_mask = graph.nodes["flow"].data.get("test_mask", torch.ones_like(labels, dtype=torch.bool))
    candidates = torch.where((labels == fallback_label) & test_mask.bool())[0]
    if len(candidates) == 0:
        candidates = torch.where(labels == fallback_label)[0]
    if len(candidates) == 0:
        raise RuntimeError("No malicious flow found in graph for attack initialization.")
    return int(candidates[0].item())


def _load_surrogates(args: argparse.Namespace, in_feats: int, n_classes: int, device: Any) -> Dict[str, Any]:
    surrogates: Dict[str, Any] = {}
    for idx, model_name in enumerate(args.surrogate_models):
        if idx >= len(args.surrogate_model_paths):
            break
        model_path = args.surrogate_model_paths[idx]
        surrogates[model_name] = _load_model(
            model_name=model_name,
            model_path=model_path,
            in_feats=in_feats,
            n_classes=n_classes,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            aggr_type=args.aggr_type,
            device=device,
        )
    return surrogates


def main(argv: list[str] | None = None) -> None:
    import torch

    from src.attacks import RandomEvasionAttack, TargetedEvasionAttack
    from src.data.feature_pool import get_global_feature_pool
    from src.data.loaders import load_dataset
    from src.utils.logger import setup_logging

    setup_logging()

    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _validate_dataset_root(args.path)

    train_frac = _normalize_split_fraction(args.train_frac, "--train-frac")
    surr_frac = _normalize_split_fraction(args.surr_frac, "--surr-frac")
    seed = 42
    if train_frac + surr_frac >= 1:
        raise ValueError(
            f"train_frac + surr_frac must be < 1. Received {args.train_frac} and {args.surr_frac}."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, _, g_test = load_dataset(
        dataset_name=args.dataset,
        classes=args.classes,
        train_frac=train_frac,
        surr_frac=surr_frac,
        seed=seed,
        path=args.path,
    )

    in_feats = int(g_test.nodes["flow"].data["h"].shape[1])
    flow_labels = g_test.nodes["flow"].data["label"]
    n_classes = int(flow_labels.shape[-1]) if flow_labels.ndim > 1 else int(flow_labels.max().item() + 1)

    victim_model = _load_model(
        model_name=args.target_model,
        model_path=args.target_model_path,
        in_feats=in_feats,
        n_classes=n_classes,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        aggr_type=args.aggr_type,
        device=device,
    )

    malicious_flow_id = args.malicious_flow_id
    if malicious_flow_id is None:
        malicious_flow_id = _infer_malicious_flow_id(g_test)

    feature_pool_obj = get_global_feature_pool(
        dataset_name=args.dataset,
        classes_def=args.classes,
        pool_strategy=args.feature_pool_strategy,
        pool_k=args.feature_pool_k,
        label_filter=0,
        seed=seed,
        dataset_path=args.path,
        train_frac=train_frac,
        surr_frac=surr_frac,
    )

    metrics_path = output_dir / f"{args.dataset}_{args.target_model}_{args.attack}_metrics.json"

    if args.attack == "targeted_evasion":
        surrogates = _load_surrogates(args, in_feats, n_classes, device)
        attack = TargetedEvasionAttack(
            G=g_test,
            victim_model=victim_model,
            surrogate_models=surrogates,
            feature_pool=feature_pool_obj.feature_pool,
            pool_indices=feature_pool_obj.pool_indices,
            device=device,
            pool_strategy=args.feature_pool_strategy,
            pool_k=args.feature_pool_k or 50,
            exp_folder=str(output_dir),
        )
        history = attack.attack(
            malicious_flow_id=malicious_flow_id,
            attacker_endpoint_id=args.attacker_endpoint_id,
            budget=args.budget,
            destination_id=args.destination_endpoint_id,
        )
        attack.save_metrics(str(metrics_path))
    else:
        attack = RandomEvasionAttack(
            G=g_test,
            victim_model=victim_model,
            feature_pool=feature_pool_obj.feature_pool,
            pool_indices=feature_pool_obj.pool_indices,
            device=device,
            pool_strategy=args.feature_pool_strategy,
            exp_folder=str(output_dir),
            seed=seed,
        )
        history = attack.attack(
            malicious_flow_id=malicious_flow_id,
            attacker_endpoint_id=args.attacker_endpoint_id,
            budget=args.budget,
        )
        attack.save_metrics(str(metrics_path))

    summary = {
        "dataset": args.dataset,
        "target_model": args.target_model,
        "attack": args.attack,
        "budget": args.budget,
        "steps_recorded": len(history),
        "metrics_path": str(metrics_path),
    }

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
