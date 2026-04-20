import torch
import config
from data.load_qm9 import load_qm9, PROPERTY_NAMES
from environment.acquisition_env import build_molecule_list
from models.predictor import load_predictor, build_xy

def main():
    _, val_loader, _, stats = load_qm9(batch_size=256)
    val_mols = build_molecule_list(val_loader)
    target_idx = stats["target_idx"]
    X, y = build_xy(val_mols, target_idx, config.DEVICE)

    model = load_predictor("./checkpoints/predictor_baseline.pt", device=config.DEVICE)
    X, y = X.to(model.device), y.to(model.device)

    # Baseline: no features observed
    mask0 = torch.zeros_like(X)
    x0 = torch.cat([mask0, X * mask0], dim=1)
    with torch.no_grad():
        pred0 = model(x0)
    mse_zero = ((pred0 - y) ** 2).mean().item()
    print(f"\nTarget: {PROPERTY_NAMES[target_idx]}")
    print(f"MSE with ZERO features observed: {mse_zero:.4f}\n")

    # Rank single features
    results = []
    for i in range(config.NUM_FEATURES):
        if i == target_idx:
            continue
        mask = torch.zeros_like(X)
        mask[:, i] = 1.0
        xb = torch.cat([mask, X * mask], dim=1)
        with torch.no_grad():
            pred = model(xb)
        mse = ((pred - y) ** 2).mean().item()
        gain = mse_zero - mse
        results.append((PROPERTY_NAMES[i], i, mse, gain))

    results.sort(key=lambda r: r[2])
    print(f"  {'rank':<5} {'feature':<14} {'idx':<5} {'MSE':<10} {'gain vs 0':<10}")
    for rank, (name, i, mse, gain) in enumerate(results, 1):
        print(f"  {rank:<5} {name:<14} {i:<5} {mse:.4f}     {gain:+.4f}")

if __name__ == "__main__":
    main()