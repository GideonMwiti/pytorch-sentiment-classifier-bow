# Sentiment Classifier with Bag-of-Words (BoW) & Feedforward Neural Network in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Validation Accuracy](https://img.shields.io/badge/Validation%20Accuracy-78.90%25-brightgreen)]()
[![F1-Score](https://img.shields.io/badge/Macro%20F1--Score-78.10%25-blue)]()

An end-to-end, production-grade Natural Language Processing (NLP) implementation of a **Bag-of-Words (BoW) Sentiment Classifier** using PyTorch. 

Trained on the canonical **Stanford Sentiment Treebank (SST-2)** movie review benchmark (6,920 training samples, 872 validation samples), this project covers custom text tokenization, vocabulary construction with frequency pruning, term-frequency vectorization, feedforward neural network classification with Batch Normalization and Dropout, model interpretability via global linear attribution, and real-time sentence inference.

---

## Visual Diagnostic Overview

The 4-panel diagnostic dashboard below illustrates the complete NLP classification pipeline: training and validation BCE loss trajectories, validation accuracy and macro F1 curves, learned top positive and negative word attributions, and the normalized confusion matrix:

![Sentiment BoW Classifier Results](sentiment_bow_classifier_results.png)

---

## Table of Contents

- [Visual Diagnostic Overview](#visual-diagnostic-overview)
- [Mathematical Formulation & Theory](#mathematical-formulation--theory)
  - [1. Vector Space Model & Term-Frequency Normalization](#1-vector-space-model--term-frequency-normalization)
  - [2. Feedforward Neural Network Topology](#2-feedforward-neural-network-topology)
  - [3. Binary Cross-Entropy Loss](#3-binary-cross-entropy-loss)
  - [4. Model Interpretability & Weight Attribution](#4-model-interpretability--weight-attribution)
- [Dataset: Stanford Sentiment Treebank (SST-2)](#dataset-stanford-sentiment-treebank-sst-2)
- [Experimental Results & Real-Time Inference](#experimental-results--real-time-inference)
  - [Validation Performance](#validation-performance)
  - [Interactive Inference Showcase](#interactive-inference-showcase)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Reproduction](#installation--reproduction)
- [Author & License](#author--license)

---

## Mathematical Formulation & Theory

### 1. Vector Space Model & Term-Frequency Normalization

In the classical Bag-of-Words (BoW) vector space model, a text document $d$ is mapped to a sparse vector $\mathbf{x} \in \mathbb{R}^V$, where $V = |\mathcal{V}|$ is the vocabulary size. 

Rather than raw frequency counts which disproportionately penalize longer documents, we apply **$L_1$ Term-Frequency (TF) normalization**:

$$
x_j = \frac{\text{count}(w_j, d)}{\sum_{k=1}^V \text{count}(w_k, d)}, \quad \forall j \in \{1, \dots, V\}
$$

Such that:

$$
\|\mathbf{x}\|_1 = \sum_{j=1}^V x_j = 1
$$

Out-of-vocabulary (OOV) tokens occurring fewer than `min_freq = 2` times are routed to a reserved `<UNK>` index, preserving bounded representation dimensionality.

### 2. Feedforward Neural Network Topology

The model implements a deep 3-stage feedforward architecture equipped with 1D Batch Normalization and aggressive Dropout to curb memorization of sparse high-dimensional n-grams:

```
Text Review String
        │
        ▼ Tokenization & TF-BoW Vectorization
Sparse Feature Vector x ∈ R^5000
        │
        ▼ Linear(5000 -> 128) -> BatchNorm1d(128) -> ReLU -> Dropout(p=0.40)
Hidden Layer 1 h1 ∈ R^128
        │
        ▼ Linear(128 -> 32) -> ReLU -> Dropout(p=0.20)
Hidden Layer 2 h2 ∈ R^32
        │
        ▼ Linear(32 -> 1)
Scalar Logit z ∈ R
        │
        ▼ Sigmoid Activation: σ(z) = 1 / (1 + e^-z)
Predicted Positive Probability P(Y = 1 | x) ∈ (0, 1)
```

- **Total Trainable Parameters**: $644,545$ parameters.

### 3. Binary Cross-Entropy Loss

The network optimizes binary cross-entropy with numerically stable log-sum-exp:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right]
$$

### 4. Model Interpretability & Weight Attribution

To inspect what semantic features the neural network prioritized, we compute the first-order global linear sensitivity:

$$
\mathbf{w}_{\text{eff}} = \mathbf{w}_{\text{out}}^T \mathbf{W}_2 \mathbf{W}_1 \in \mathbb{R}^{1 \times V}
$$

- Tokens with large positive weights ($w_{\text{eff}, j} \gg 0$) act as strong indicators of positive sentiment (*e.g., "sparkling", "remarkable", "enjoyed", "wonderful"*).
- Tokens with large negative weights ($w_{\text{eff}, j} \ll 0$) drive negative sentiment predictions (*e.g., "distasteful", "failure", "repetitive", "poor"*).

---

## Dataset: Stanford Sentiment Treebank (SST-2)

The **SST-2** benchmark is a gold-standard sentiment classification corpus extracted from Rotten Tomatoes movie reviews:

- **Training Set**: $6,920$ labeled sentence reviews.
- **Validation Set**: $872$ labeled sentence reviews.
- **Vocabulary Size**: $5,000$ unique tokens after frequency filtering (`min_freq = 2`).

---

## Experimental Results & Real-Time Inference

### Validation Performance

| Metric | Training Set | Validation Set |
| :--- | :---: | :---: |
| **BCE Loss** | $0.0225$ | $0.8761$ |
| **Accuracy** | $99.39\%$ | **$78.90\%$** |
| **Macro F1-Score** | $99.39\%$ | **$78.10\%$** |

*(Note: For an unigram Bag-of-Words model lacking sequential recurrent or attention mechanisms, an accuracy of ~79% on SST-2 represents the practical upper bound of bag-of-words text representations).*

### Interactive Inference Showcase

| Input Movie Review Sentence | True Sentiment | Predicted Label | Probability $P(\text{Pos})$ | Confidence |
| :--- | :---: | :---: | :---: | :---: |
| *"An absolute cinematic masterpiece with breathtaking performances and stunning visuals."* | Positive | **POSITIVE** | $1.0000$ | **$100.0\%$** |
| *"Completely predictable, dull, and a colossal waste of two hours."* | Negative | **NEGATIVE** | $0.0000$ | **$100.0\%$** |
| *"Hilarious, heartwarming, and deeply moving from start to finish."* | Positive | **POSITIVE** | $1.0000$ | **$100.0\%$** |
| *"A mediocre screenplay dragged down by wooden dialogue and poor pacing."* | Negative | **NEGATIVE** | $0.0000$ | **$100.0\%$** |

---

## Project Directory Structure

```plaintext
13_sentiment_classifier_bow/
│
├── data/                                      # Stanford Sentiment Treebank TSV dataset
│   ├── train.tsv                              # 6,920 training samples
│   └── dev.tsv                                # 872 validation samples
├── sentiment_bow_classifier_results.png       # 4-panel diagnostic visual artifact
├── main.py                                    # Full tokenization, BoW, and NN training pipeline
├── .gitignore                                 # Git ignore configuration
└── README.md                                  # Mathematical documentation & benchmark analysis
```

---

## Installation & Reproduction

### Prerequisites

- Python 3.10+
- PyTorch
- NumPy & Matplotlib

```bash
# Clone the repository
git clone https://github.com/GideonMwiti/pytorch-sentiment-classifier-bow.git
cd pytorch-sentiment-classifier-bow

# Install dependencies
pip install torch numpy matplotlib
```

### Run Training and Inference

Execute the self-contained script:

```bash
python main.py
```

The script will:
1. Load and parse the SST-2 dataset.
2. Build the 5,000-word Bag-of-Words vocabulary.
3. Train the feedforward network across 8 epochs with Cosine Annealing.
4. Extract the learned top positive and negative sentiment lexicon.
5. Execute live inference on novel test sentences.
6. Export the 4-panel diagnostic dashboard `sentiment_bow_classifier_results.png`.

---

## Author & License

This project is part of the **100 PyTorch Projects in 10 Days** challenge.

- **Author**: Gideon Mwiti
- **GitHub**: [@GideonMwiti](https://github.com/GideonMwiti)
- **License**: MIT

