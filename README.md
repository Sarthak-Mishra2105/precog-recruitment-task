# Colored MNIST: Shortcut Learning and Debiasing

A PyTorch study of spurious correlations, interpretability, debiasing, and adversarial robustness using a biased "Colored MNIST" dataset where digit colors correlate with labels.

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup (loads pre-trained models)
python run_all.py --mode load_artifacts

# Or run full pipeline from scratch (~20 min)
python run_all.py --mode run_from_scratch
```

## Repository Structure

```
├── README.md
├── requirements.txt
├── run_all.py                  # Main entry point
├── notebooks/
│   └── cv_task.ipynb           # Jupyter notebook (all tasks)
├── src/
│   ├── data_colored_mnist.py   # Dataset with spurious correlation
│   ├── models.py               # CNN architectures
│   ├── interpretability.py     # Grad-CAM and Activation Maximization
│   ├── debias.py               # Consistency and GRL debiasing
│   └── attacks.py              # Targeted PGD attack
├── artifacts/                  # Saved models and figures
├── scripts/
│   └── smoke_test.py           # Quick sanity check
└── report/
    ├── main.tex                # LaTeX source
    └── report.pdf              # Compiled report
```

## Results

| Model | Val Accuracy | Hard Test Accuracy |
|-------|--------------|-------------------|
| Baseline (Cheater) | ~95% | ~6% |
| Consistency | ~97% | **96.27%** |
| GRL + Augmentation | ~96% | **92.77%** |

Both debiasing methods exceed the 70% target on the hard test set.

### Adversarial Attack Summary
Targeted PGD attack (digit 7 → 3) with ε < 0.05:
- Cheater: 6 steps to success
- Consistency: 12 steps to success
- GRL: 8 steps to success

## Task Overview

0. **Dataset**: 95% color-label correlation in train/val, 0% in hard test
1. **Baseline CNN**: Learns color shortcut (95% val, 6% hard test)
2. **Activation Maximization**: Visualizes learned features (color blobs)
3. **Grad-CAM**: Confirms attention on color, not digit shape
4. **Debiasing**: Color Consistency + GRL achieve >70% hard test
5. **Adversarial Attacks**: Targeted PGD with comparison across models

## Reproducibility

- All experiments use seed 42
- MNIST downloads automatically on first run (~11 MB)
- GPU optional but speeds training 2-5×

## Demo App

Interactive Streamlit demo where you can draw digits and see model predictions + Grad-CAM:

```bash
# Install demo dependencies
pip install -r requirements-demo.txt

# Run the demo
streamlit run demo_app/app.py
```

Features:
- Draw digits on canvas with selectable stroke colors
- Side-by-side predictions from Cheater/Consistency/GRL models
- Grad-CAM visualizations for each model
- Recolor test showing predictions across all 10 colors

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision, numpy, matplotlib, tqdm

