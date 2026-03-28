# FracCaps — Fractal-Capsule Network for Hierarchical Visual Reasoning

> **99.02% test accuracy on MNIST** · PyTorch ·hybrid architecture 

---

## Overview

FracCaps is a  deep learning architecture that combines **Fractal Neural Networks (FrNN)** and **Capsule Networks (CapsNet)** to address a fundamental gap in visual reasoning: existing models either handle spatial hierarchy *or* scale invariance, but rarely both.

| Architecture | Strength | Weakness |
|---|---|---|
| Standard CNN | Fast, scalable | Loses spatial relationships via pooling |
| CapsNet | Encodes part-whole relationships | Heavy compute, no multi-scale handling |
| FrNN | Multi-scale, self-similar patterns | No relational reasoning between parts |
| **FracCaps** | **Both spatial hierarchy and scale invariance** | — |

---

## Results

### MNIST — Test Set (10,000 samples)

```
Test Accuracy: 99.02%

              precision    recall  f1-score   support

    0 - zero       0.99      1.00      0.99       980
     1 - one       0.99      0.99      0.99      1135
     2 - two       0.98      1.00      0.99      1032
   3 - three       1.00      0.99      0.99      1010
    4 - four       0.99      0.98      0.99       982
    5 - five       0.98      1.00      0.99       892
     6 - six       1.00      0.99      0.99       958
   7 - seven       0.99      0.98      0.99      1028
   8 - eight       1.00      0.99      0.99       974
    9 - nine       0.98      0.98      0.98      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000
```

Achieved using the **Lite config** (reduced depth, half channels, 20 epochs) — full config on CIFAR-10 is in progress.

---

## Architecture

```
Input Image
     │
     ├──── Fine Branch (stride 1)   ──┐
     ├──── Mid Branch  (stride 2)   ──┤  FractalEncoder
     └──── Coarse Branch (stride 4) ──┘
                    │
             [Concat + Upsample]
                    │
          Primary Capsule Layer
          (squash activation, pose vectors)
                    │
       Dynamic Routing by Agreement
       (2–3 iterations, cross-scale voting)
                    │
          Class Capsule Layer
          (capsule lengths → class probs)
                    │
         ┌──────────┴──────────┐
    Classification          Decoder
    (margin loss)      (reconstruction loss)
```

### Key Components

**Fractal Encoder Block**
Each branch runs a recursive two-path structure: a direct conv path and a recursive fractal path, merged via element-wise mean. Drop-path regularisation (p=0.15) is applied during training. Three parallel branches process the input at fine (stride 1), mid (stride 2), and coarse (stride 4) spatial scales, then upsample and concatenate.

**Primary Capsule Layer**
Converts multi-scale feature maps into capsule vectors via depthwise conv + reshape + squash activation. Each capsule encodes both an existence probability (via its norm) and a pose vector (its orientation).

**Dynamic Routing by Agreement**
Sabour et al. (2017) style routing. Primary capsules vote for parent class capsules; routing logits are updated by dot-product agreement. Gradient is detached through routing coefficients on all but the final iteration.

**Decoder (Reconstruction Regulariser)**
A 3-layer MLP reconstructs the input image from the winning class capsule. This forces capsule orientations to encode meaningful pose information rather than class identity.

---

## Hyperparameters

###  Config (this run)

| Parameter | Value | Note |
|---|---|---|
| `FRACTAL_DEPTH` | 2 | Recursive depth per branch |
| `BASE_CHANNELS` | 16 | Feature map width |
| `PRIMARY_CAPS_NUM` | 16 | Number of primary capsule types |
| `PRIMARY_CAPS_DIM` | 8 | Primary capsule vector dimension |
| `CLASS_CAPS_DIM` | 8 | Class capsule vector dimension |
| `ROUTING_ITERATIONS` | 2 | Dynamic routing passes |
| `BATCH_SIZE` | 32 | — |
| `EPOCHS` | 20 | — |
| `LR` | 1e-3 | Adam + cosine annealing |
| `RECON_WEIGHT` | 0.0005 | Reconstruction loss weight |

### Full Config (for CIFAR-10)

| Parameter | Value |
|---|---|
| `FRACTAL_DEPTH` | 3 |
| `BASE_CHANNELS` | 32 |
| `PRIMARY_CAPS_NUM` | 32 |
| `CLASS_CAPS_DIM` | 16 |
| `ROUTING_ITERATIONS` | 3 |
| `BATCH_SIZE` | 64 |
| `EPOCHS` | 50 |

---

## Training Strategy

Three-phase training recommended for best stability:

1. **Pre-train fractal branches independently** with cross-entropy on the base task
2. **Freeze fractal encoders, train primary capsule heads** — establishes stable pose representations
3. **End-to-end fine-tune** with combined loss:
   - Margin loss (classification)
   - Reconstruction loss (decoder regularisation)
   - Scale consistency loss *(open research problem — see below)*

---

## Loss Functions

**Margin Loss** (Sabour et al. 2017):

$$\mathcal{L}_k = T_k \cdot \max(0,\ m^+ - \|v_k\|)^2 + \lambda (1 - T_k) \cdot \max(0,\ \|v_k\| - m^-)^2$$

where $m^+ = 0.9$, $m^- = 0.1$, $\lambda = 0.5$.

**Reconstruction Loss**: MSE between decoder output and normalised input image.

**Combined Loss**: $\mathcal{L} = \mathcal{L}_{margin} + 0.0005 \cdot \mathcal{L}_{recon}$

---

## Open Research Problems

This architecture opens three genuinely unsolved questions:

1. **Cross-scale routing** — How to design an efficient routing algorithm that operates across capsules from 3 different scale spaces without collapsing to a single-scale solution.

2. **Scale consistency loss** — A principled mathematical formulation that penalises disagreement between fine/mid/coarse capsules detecting the same entity. A candidate form:

$$\mathcal{L}_{scale} = \sum_{i \neq j} \| \hat{u}_{k \leftarrow i} - \hat{u}_{k \leftarrow j} \|^2$$

where $\hat{u}_{k \leftarrow i}$ is the predicted vote for parent capsule $k$ from scale branch $i$.

3. **Fractal-capsule weight sharing** — Whether sharing weights between fractal branches helps or hurts capsule pose quality is an open empirical question.

---

## Ideal Use Cases

This architecture is best suited for domains where self-similarity and viewpoint equivariance matter simultaneously:

- **Medical imaging** — tumors, fractures, and tissue textures are self-similar at multiple scales; CapsNet's viewpoint equivariance handles patient positioning variation
- **Remote sensing / satellite imagery** — land cover has fractal properties; capsules handle object pose in aerial views
- **Material science microscopy** — grain boundaries, crystal structures
- **Texture classification** — DTD dataset is a natural benchmark
- **Anomaly detection** — fractal deviation is often the anomaly signal

---

## Installation & Usage

```bash
git clone https://github.com/<your-username>/fraccaps.git
cd fraccaps
pip install torch torchvision matplotlib seaborn scikit-learn tqdm
```

Open `FracCaps_Model.ipynb` for the full config or `FracCaps_Lite.ipynb` for the CPU-friendly version.

```python
# Quick inference
model = FracCaps(cfg).to(device)
model.load_state_dict(torch.load('fraccaps_lite_best.pth'))
model.eval()

lengths, caps, recon = model(img_tensor)
predicted_class = lengths.argmax(dim=1)
```

---

## Experiment Ideas

| Idea | How to implement |
|---|---|
| EM routing | Replace `DynamicRouting` with EM-CapsNet style M/E-step |
| More fractal depth | Increase `FRACTAL_DEPTH` to 4–5 (GPU memory scales) |
| Medical imaging | Replace CIFAR-10 loader with `ImageFolder` dataset |
| Texture classification | DTD dataset — fractal branches are natively suited |
| Scale consistency loss | Auxiliary loss penalising fine/mid/coarse capsule disagreement |
| Attention routing | Replace agreement dot-product with cross-attention |
| Larger backbone | Replace `FractalBranch` entry with a ResNet-18 stem per scale |

---

## References

- Sabour, S., Frosst, N., & Hinton, G. E. (2017). *Dynamic Routing Between Capsules.* NeurIPS.
- Larsson, G., Maire, M., & Shakhnarovich, G. (2016). *FractalNet: Ultra-Deep Neural Networks without Residuals.* ICLR.
- Hinton, G. E., Sabour, S., & Frosst, N. (2018). *Matrix Capsules with EM Routing.* ICLR.

---
