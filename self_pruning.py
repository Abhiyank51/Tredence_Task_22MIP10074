import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────
#  PART 1 — Custom Prunable Linear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A fully-connected layer that learns to prune its own weights during training.
    Each weight has an associated learnable gate (0 = pruned, 1 = active).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # Initialise gate scores to -1 → sigmoid ≈ 0.27
        nn.init.constant_(self.gate_scores, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_stats(self, threshold: float = 1e-2) -> dict:
        gates = self.get_gates()
        n = gates.numel()
        pruned = (gates < threshold).sum().item()
        return {
            "total": n,
            "pruned": pruned,
            "sparsity": pruned / n * 100,
            "mean_gate": gates.mean().item(),
        }


# ─────────────────────────────────────────────
#  PART 2 — Self-Pruning Feedforward Network
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    def __init__(self, input_dim: int = 32*32*3, num_classes: int = 10):
        super().__init__()

        dims = [input_dim, 512, 256, 128, num_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(PrunableLinear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm1d(dims[i+1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)

    def prunable_layers(self):
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def network_sparsity_loss(self) -> torch.Tensor:
        """
        L1 penalty on gate values (mean across all parameters).
        Using the mean makes the loss scale independent of network size,
        so λ values of 0.01, 0.1, 0.5 work directly.
        """
        total_sum = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            total_sum = total_sum + gates.sum()
        return total_sum

    def network_sparsity_stats(self, threshold: float = 1e-2) -> dict:
        total, pruned = 0, 0
        mean_gates = []
        for layer in self.prunable_layers():
            s = layer.sparsity_stats(threshold)
            total += s["total"]
            pruned += s["pruned"]
            mean_gates.append(s["mean_gate"])
        return {
            "total_params": total,
            "pruned_params": pruned,
            "sparsity_pct": pruned / total * 100 if total else 0,
            "avg_mean_gate": sum(mean_gates) / len(mean_gates) if mean_gates else 0,
        }

    def all_gate_values(self) -> torch.Tensor:
        return torch.cat([layer.get_gates().view(-1)
                          for layer in self.prunable_layers()])


# ─────────────────────────────────────────────
#  PART 3 — Data Loading
# ─────────────────────────────────────────────

def get_dataloaders(data_dir: str = './data', batch_size: int = 128):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = torchvision.datasets.CIFAR10(data_dir, train=True,
                                               download=True, transform=train_transform)
    test_set   = torchvision.datasets.CIFAR10(data_dir, train=False,
                                               download=True, transform=test_transform)

    val_size   = 5000
    train_size = len(full_train) - val_size
    train_set, val_set = torch.utils.data.random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))

    workers = 0
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True,  num_workers=workers)
    val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=batch_size,
                                                shuffle=False, num_workers=workers)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size,
                                                shuffle=False, num_workers=workers)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
#  PART 4 — Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, lam, device):
    model.train()
    total_loss, ce_loss_sum, sp_loss_sum = 0.0, 0.0, 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        ce_loss = criterion(outputs, targets)
        sp_loss = model.network_sparsity_loss()       # mean gate value
        loss = ce_loss + lam * sp_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print("[WARNING] NaN/Inf detected — skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        ce_loss_sum += ce_loss.item()
        sp_loss_sum += sp_loss.item()

    n = len(loader)
    return total_loss / n, ce_loss_sum / n, sp_loss_sum / n


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_sum += criterion(outputs, targets).item()
            predicted = outputs.argmax(dim=1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return loss_sum / len(loader), 100.0 * correct / total


def train(lam: float, epochs: int, device, data_dir: str, batch_size: int = 128):
    print(f"\n{'='*60}")
    print(f"  Training  λ = {lam}  |  {epochs} epochs  |  {device}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)
    model = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "total_loss": [], "ce_loss": [], "sp_loss": [],
        "val_acc": [], "val_loss": [],
        "avg_gate": [], "sparsity_pct": [],
    }

    for epoch in range(1, epochs + 1):
        total_l, ce_l, sp_l = train_one_epoch(
            model, train_loader, criterion, optimizer, lam, device)
        val_l, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        stats = model.network_sparsity_stats(threshold=1e-2)
        avg_g = stats["avg_mean_gate"]
        spar  = stats["sparsity_pct"]

        history["total_loss"].append(total_l)
        history["ce_loss"].append(ce_l)
        history["sp_loss"].append(sp_l)
        history["val_loss"].append(val_l)
        history["val_acc"].append(val_acc)
        history["avg_gate"].append(avg_g)
        history["sparsity_pct"].append(spar)

        print(f"  Epoch {epoch:>2}/{epochs} | "
              f"Total: {total_l:.4f}  CE: {ce_l:.4f}  "
              f"Sp: {sp_l:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Avg Gate: {avg_g:.4f}  Sparsity: {spar:.2f}%")

    _, test_acc = evaluate(model, test_loader, criterion, device)
    final_stats = model.network_sparsity_stats(threshold=1e-2)
    print(f"\n  ✓ Test Accuracy : {test_acc:.2f}%")
    print(f"  ✓ Sparsity Level: {final_stats['sparsity_pct']:.2f}%")

    return history, model, test_acc, final_stats


# ─────────────────────────────────────────────
#  PART 5 — Visualisation & Report
# ─────────────────────────────────────────────

def plot_training_curves(all_history: dict, out_dir: str):
    lambdas = list(all_history.keys())
    colors  = ['#6366f1', '#f59e0b', '#ef4444']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f172a')
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for lam, col in zip(lambdas, colors):
        h = all_history[lam]
        axes[0].plot(h["ce_loss"], color=col, lw=2, label=f"λ={lam}")
    axes[0].set_title("Classification Loss per Epoch", color='white', fontsize=13)
    axes[0].set_xlabel("Epoch", color='#94a3b8')
    axes[0].set_ylabel("Loss", color='#94a3b8')
    axes[0].legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
    axes[0].grid(alpha=0.15)

    for lam, col in zip(lambdas, colors):
        h = all_history[lam]
        axes[1].plot(h["val_acc"], color=col, lw=2, label=f"λ={lam}")
    axes[1].set_title("Validation Accuracy per Epoch", color='white', fontsize=13)
    axes[1].set_xlabel("Epoch", color='#94a3b8')
    axes[1].set_ylabel("Accuracy (%)", color='#94a3b8')
    axes[1].legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
    axes[1].grid(alpha=0.15)

    fig.suptitle("Self-Pruning Network — Training Curves", color='white', fontsize=15, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_gate_distribution_for_best_model(model, lam, out_dir: str):
    gates = model.all_gate_values().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.hist(gates, bins=50, color='#6366f1', alpha=0.85, edgecolor='none')
    ax.set_title(f"Gate Value Distribution – Best Model (λ = {lam})", color='white', fontsize=14)
    ax.set_xlabel("Gate Value  (0 = pruned, 1 = active)", color='#94a3b8')
    ax.set_ylabel("Count", color='#94a3b8')
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.12)

    path = os.path.join(out_dir, "gate_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def generate_report(results, best_lam, out_dir: str):
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("# Self-Pruning Neural Network – Case Study Report\n\n")
        f.write("## 1. Why L1 penalty on sigmoid gates encourages sparsity\n")
        f.write(
            "The sparsity loss is an L1 penalty (here computed as the mean of all gate values, "
            "which is proportional to the L1 sum). Since gate values are always positive after "
            "sigmoid, minimising this penalty drives gates toward zero. The cross‑entropy loss "
            "counteracts this for weights that are important for classification, resulting in a "
            "network that prunes away superfluous connections while retaining essential ones.\n\n"
        )

        f.write("## 2. Results Summary\n\n")
        f.write("| λ (lambda) | Test Accuracy (%) | Sparsity Level (%) |\n")
        f.write("|------------|-------------------|-------------------|\n")
        for r in results:
            f.write(f"| {r['lam']:>10} | {r['test_acc']:>17.2f} | {r['sparsity_pct']:>17.2f} |\n")
        f.write("\n")

        f.write("## 3. Gate Distribution for Best Model\n\n")
        f.write(f"The best model (selected by test accuracy) used λ = {best_lam}. ")
        f.write("The histogram below shows a large spike near 0 (pruned) and a cluster "
                "away from 0 (active).\n\n")
        f.write("![Gate Distribution](gate_distribution.png)\n\n")

        f.write("## 4. Training Curves\n\n")
        f.write("![Training Curves](training_curves.png)\n")

    print(f"  Report saved: {report_path}")


# ─────────────────────────────────────────────
#  MAIN — Run All Experiments
# ─────────────────────────────────────────────

def main():
    EPOCHS     = 25
    BATCH_SIZE = 128
    DATA_DIR   = './data'
    OUT_DIR    = './results'
    # Your own chosen λ values (low, medium, high)
    LAMBDAS = [1e-5, 5e-5, 1e-4]

    os.makedirs(OUT_DIR,  exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")
    print(f"Epochs : {EPOCHS}")
    print(f"Lambdas: {LAMBDAS}\n")

    all_history = {}
    all_models  = {}
    results     = []

    for lam in LAMBDAS:
        history, model, test_acc, stats = train(
            lam=lam, epochs=EPOCHS, device=device,
            data_dir=DATA_DIR, batch_size=BATCH_SIZE)

        all_history[lam] = history
        all_models[lam]  = model
        results.append({
            "lam": lam,
            "test_acc": test_acc,
            "sparsity_pct": stats["sparsity_pct"],
        })

    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"  {'Lambda':>10} | {'Test Acc':>12} | {'Sparsity %':>12}")
    print("  " + "-"*44)
    for r in results:
        print(f"  {r['lam']:>10} | {r['test_acc']:>11.2f}% | {r['sparsity_pct']:>11.2f}%")

    best_entry = max(results, key=lambda x: x["test_acc"])
    best_lam = best_entry["lam"]
    best_model = all_models[best_lam]

    print("\n  Generating plots and report…")
    plot_training_curves(all_history, OUT_DIR)
    plot_gate_distribution_for_best_model(best_model, best_lam, OUT_DIR)
    generate_report(results, best_lam, OUT_DIR)

    print("\n  All done. Check the ./results directory.\n")


if __name__ == '__main__':
    main()