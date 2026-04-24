# 🧠 Self-Pruning Neural Network with Learnable Gates 
**Tredence Analytics — AI Engineer Case Study Submission** A neural network that learns to compress *itself* during training through differentiable gate parameters—eliminating the need for post-training pruning. 

---

## 📌 Table of Contents 
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Code Architecture & Functions](#code-architecture--functions)
- [Tech Stack](#tech-stack)
- [Visualizations & Dashboard](#visualizations--dashboard)
- [Quick Start](#quick-start)
- [Insights & Trade-off Analysis](#insights--trade-off-analysis)

---

## Overview 
This project implements the **Self-Pruning Neural Network** case study for the Tredence Analytics AI Engineer internship. The core idea is simple but powerful: 

> Instead of pruning a trained model in a separate step, attach a learnable **gate scalar** to every weight. Penalize the model during training for keeping too many gates open. The network learns what to keep—and what to throw away—on its own. 

This repository contains a clean, standalone PyTorch script that handles custom layer creation, dataset loading, training routines, and automatic result visualization.

---

## How It Works 

### The Gated Weight Mechanism 
Every `PrunableLinear` layer replaces the standard PyTorch linear layer with: 

```python
gates         = torch.sigmoid(gate_scores)        # ∈ (0, 1) per weight
pruned_weight = weight * gates                    # element-wise masking
output        = F.linear(x, pruned_weight, bias)  # standard linear op
```

`gate_scores` is a **learnable parameter tensor** with the same shape as `weight`. When a gate collapses to 0, the corresponding weight is silenced—that connection is pruned without ever being removed from the graph. Gradients flow through both `weight` and `gate_scores` automatically. 

### The Loss Function 
```text
Total Loss = CrossEntropyLoss + λ × SparsityLoss 
SparsityLoss = Σ sigmoid(gate_scores) / N_gates    # normalized L1 ∈ (0, 1)
```

- **λ (lambda)** controls the sparsity-accuracy trade-off. 
- The L1 norm on gate values encourages exact zeros (unlike L2 which only shrinks). 
- Normalization by total gate count keeps the loss scale stable across model sizes. 

---

## Project Structure 

```text
TREDENCE_AI/ 
│ 
├── self_pruning_network.py    ← Complete implementation in one standalone script
├── requirements.txt           ← Dependencies (torch, torchvision, matplotlib)
├── data/                      ← Auto-downloaded CIFAR-10 dataset
└── results/                   ← Auto-generated upon execution
    ├── training_curves.png
    ├── gate_distribution.png
    └── report.md
```

---

## Code Architecture & Functions

The script (`self_pruning_network.py`) is modularized into distinct parts for clarity:

### 1. Custom Layers (`PrunableLinear`)
* **`__init__`**: Initializes weights, biases, and the learnable `gate_scores`. Gates are initialized to -1.0 so the sigmoid starts at ~0.27, providing a stable starting point.
* **`forward`**: Applies the sigmoid to gate scores, multiplies them element-wise with the weights, and performs the linear transformation.
* **`sparsity_stats`**: Calculates the percentage of gates that have dropped below a specific threshold (effectively pruned).

### 2. Network Architecture (`SelfPruningNet`)
* **`__init__`**: Assembles a 4-layer feedforward network (`3072 → 512 → 256 → 128 → 10`) using the custom `PrunableLinear` layers.
* **`network_sparsity_loss`**: Computes the mean gate value across the entire network. Normalizing this loss ensures stability regardless of model depth/width.

### 3. Data & Training Pipeline
* **`get_dataloaders`**: Downloads CIFAR-10, applies data augmentation (flips, crops), and splits the data into training, validation, and test sets.
* **`train_one_epoch`**: Core training logic. Calculates Cross Entropy and Sparsity Loss, checks for NaN/Inf anomalies, clips gradients, and steps the Adam optimizer.
* **`evaluate`**: Runs the model in inference mode to calculate accuracy and loss on validation/test sets.
* **`train`**: The master training loop. Manages epochs, coordinates the Cosine Annealing learning rate scheduler, and tracks history metrics.

### 4. Visualization & Reporting
* **`plot_training_curves`**: Generates a side-by-side plot comparing Classification Loss and Validation Accuracy across different λ values.
* **`plot_gate_distribution_for_best_model`**: Creates a histogram of the gate values for the best-performing model, demonstrating the bimodal "pruned vs. active" behavior.
* **`generate_report`**: Automatically outputs a detailed markdown summary of the run.

---

## Tech Stack 

| Technology | Role |
|------------|------|
| **PyTorch** | Neural network framework, custom layers, autograd |
| **Torchvision** | CIFAR-10 dataset loading & image transformations |
| **Matplotlib** | Generating training curves & gate distribution histograms |

---

## Visualizations & Dashboard

### Auto-Generated Graphs (From Script)
Running the standalone script automatically generates visualizations to analyze the network's behavior. 

**Training Curves:** Shows the classification loss and validation accuracy across different sparsity penalties (λ).
![Training Curves](results/training_curves.png)

**Gate Distribution:** Highlights the bimodal nature of the learned gates. A successful run shows a massive spike near `0` (pruned weights) and a smaller cluster of active weights.
![Gate Distribution](results/gate_distribution.png)

### Extended Dashboard (Interactive UI)
*Note: The following images showcase the extended full-stack interactive dashboard built to monitor these experiments in real-time.*

**Dashboard Overview & Metrics:**
![Training Dashboard — Metrics Overview](assets/dashboard_1.png) 

**Layer-wise Sparsity Analysis:**
![Training Dashboard — Layer-wise Sparsity & Gate Distribution](assets/dashboard_2.png) 

---

## Quick Start 

### Prerequisites
- Python 3.10+

### Running the Script

1. Install the required dependencies:
   ```bash
   pip install torch torchvision matplotlib
   ```
2. Execute the standalone training script:
   ```bash
   python self_pruning_network.py
   ```

Outputs will automatically be saved to the `./results/` directory, and the CIFAR-10 dataset will be downloaded to `./data/`. By default, the script tests λ values of `[0.01, 0.1, 0.5]` over 25 epochs.

---

## Insights & Trade-off Analysis 

### The Pareto Frontier 
As λ increases, sparsity increases but accuracy inevitably decays. The relationship traces a Pareto frontier—no single configuration dominates on both axes simultaneously. Finding the "sweet spot" yields significant sparsity with negligible accuracy loss, making it the best practical choice for deployment. 

### Layer-wise Behavior 
Not all layers prune equally. Early layers tend to retain denser connections because they extract low-level features (edges, textures) that are broadly reused. Deeper layers, which encode more task-specific patterns, prune faster under L1 pressure.

### Stable Regularization
The naive implementation of sparsity loss sums the gate values, which breaks down on larger models by producing astronomical loss numbers. This implementation safely normalizes the sparsity loss by calculating the *mean* gate value, allowing the exact same λ values to be used reliably regardless of model scale.
