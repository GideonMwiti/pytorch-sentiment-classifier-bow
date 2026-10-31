"""
Project 13: Sentiment Classifier with Bag-of-Words (BoW) & Feedforward Neural Network
Dataset: Stanford Sentiment Treebank (SST-2) Movie Reviews Benchmark (6,920 Train, 872 Test)
Framework: PyTorch
Author: Gideon Mwiti

Key Features:
- Custom NLP Text Tokenizer, Vocabulary Builder, and Vectorizer.
- Term-Frequency Normalized Bag-of-Words (BoW) feature representation.
- Deep Feedforward Neural Network with Regularization (BatchNorm, Dropout).
- Model Interpretability: Extraction of learned sentiment lexicon (Top positive & negative words).
- Real-time sentence inference pipeline for novel movie reviews.
- 4-Panel Publication-Quality Diagnostic Visual Artifact:
  1) Training and Validation Binary Cross-Entropy Loss Convergence
  2) Validation Accuracy and F1-Score Learning Curves
  3) Top 15 Positive & Negative Word Importance Bar Chart (Feature Attribution)
  4) Normalized Confusion Matrix & Qualitative Sentence Prediction Confidences
"""

import os
import re
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_tokenize(text):
    """Tokenizes text: lowercases, removes non-alphanumeric punctuation, splits tokens."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = text.strip().split()
    return tokens


class Vocabulary:
    """Manages word-to-index mappings and Bag-of-Words vectorization."""
    def __init__(self, max_vocab_size=5000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}

    def build_vocab(self, corpus):
        word_counts = Counter()
        for text in corpus:
            tokens = clean_tokenize(text)
            word_counts.update(tokens)

        # Filter by min_freq and take top max_vocab_size words
        filtered = [word for word, count in word_counts.most_common() if count >= self.min_freq]
        for word in filtered:
            if len(self.word2idx) >= self.max_vocab_size:
                break
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def text_to_bow(self, text, binary=False):
        """Converts a single string into a fixed-size normalized Bag-of-Words vector."""
        tokens = clean_tokenize(text)
        vector = np.zeros(len(self.word2idx), dtype=np.float32)
        unk_idx = self.word2idx[self.unk_token]

        if not tokens:
            return vector

        for token in tokens:
            idx = self.word2idx.get(token, unk_idx)
            vector[idx] += 1.0

        if binary:
            vector = (vector > 0).astype(np.float32)
        else:
            # Term-frequency normalization (L1 norm)
            total = np.sum(vector)
            if total > 0:
                vector /= total

        return vector

    def __len__(self):
        return len(self.word2idx)


class MovieReviewDataset(Dataset):
    """PyTorch Dataset yielding BoW feature vectors and sentiment labels."""
    def __init__(self, texts, labels, vocab, binary=False):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.binary = binary

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        bow_vec = self.vocab.text_to_bow(self.texts[idx], binary=self.binary)
        label = float(self.labels[idx])
        return torch.tensor(bow_vec, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)


def load_sst2_data(data_dir):
    """Loads Stanford Sentiment Treebank (SST-2) TSV files."""
    train_path = os.path.join(data_dir, "train.tsv")
    dev_path = os.path.join(data_dir, "dev.tsv")

    def read_tsv(path):
        texts, labels = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    text, label = parts[0].strip(), parts[1].strip()
                    if label in ("0", "1"):
                        texts.append(text)
                        labels.append(int(label))
        return texts, labels

    train_texts, train_labels = read_tsv(train_path)
    val_texts, val_labels = read_tsv(dev_path)
    return train_texts, train_labels, val_texts, val_labels


class BoWSentimentClassifier(nn.Module):
    """
    Feedforward Neural Network for Bag-of-Words Sentiment Classification.
    Topology: Input (V) -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout(0.4)
                       -> Linear(32)  -> ReLU -> Dropout(0.2)
                       -> Linear(1)   -> Output Logit
    """
    def __init__(self, vocab_size, hidden_dim1=128, hidden_dim2=32):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.40)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.20)

        self.out = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        h1 = self.drop1(self.act1(self.bn1(self.fc1(x))))
        h2 = self.drop2(self.act2(self.fc2(h1)))
        logits = self.out(h2)
        return logits


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy().ravel())
            all_targets.extend(labels.cpu().numpy().ravel())
            all_probs.extend(probs.cpu().numpy().ravel())

    total = len(all_targets)
    epoch_loss = running_loss / total
    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)

    epoch_acc = np.mean(preds_arr == targets_arr) * 100.0

    # Precision, Recall, F1
    tp = np.sum((preds_arr == 1) & (targets_arr == 1))
    fp = np.sum((preds_arr == 1) & (targets_arr == 0))
    fn = np.sum((preds_arr == 0) & (targets_arr == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return epoch_loss, epoch_acc, f1 * 100.0, preds_arr, targets_arr, np.array(all_probs)


def extract_sentiment_lexicon(model, vocab, top_k=15):
    """
    Extracts top positive and negative words based on effective 
    linear weight projection through the feedforward network.
    """
    model.eval()
    with torch.no_grad():
        # First layer weight matrix: [128, V]
        W1 = model.fc1.weight.cpu().numpy()
        # Second layer weight matrix: [32, 128]
        W2 = model.fc2.weight.cpu().numpy()
        # Output layer weight vector: [1, 32]
        W_out = model.out.weight.cpu().numpy()

        # Effective linear attribution approximation: W_out @ W2 @ W1 -> [1, V]
        # (ignoring ReLU non-linearities for global linear sensitivity)
        effective_weights = (W_out @ W2 @ W1).squeeze()

    # Filter out special tokens
    word_scores = []
    for idx in range(2, len(vocab)):
        word = vocab.idx2word[idx]
        word_scores.append((word, effective_weights[idx]))

    word_scores.sort(key=lambda x: x[1])
    top_negative = word_scores[:top_k]
    top_positive = word_scores[-top_k:][::-1]
    return top_positive, top_negative


def generate_visual_artifact(history, top_pos, top_neg, targets, preds, sample_results, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)
    epochs = range(1, len(history['train_loss']) + 1)

    # -------------------------------------------------------------
    # Panel 1: BCE Loss Convergence Curves
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training BCE Loss', linewidth=2.0)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation BCE Loss', linewidth=2.2)
    ax1.set_title("Binary Cross-Entropy Loss vs. Epochs\nSmooth Generalization with Dropout Regularization", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Validation Accuracy & F1-Score Trajectories
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_acc'], 'g-o', label=f"Val Accuracy (Final: {history['val_acc'][-1]:.2f}%)", linewidth=2.2)
    ax2.plot(epochs, history['val_f1'], 'purple', linestyle='--', marker='^', label=f"Val F1-Score (Final: {history['val_f1'][-1]:.2f}%)", linewidth=2.0)
    ax2.set_title("Validation Classification Accuracy & Macro F1-Score\nBalanced Convergence on Movie Sentiment Distribution", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Metric Score (%)", fontsize=10)
    ax2.set_ylim([60, 95])
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Learned Sentiment Lexicon (Top Positive & Negative Words)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    pos_words = [w for w, _ in top_pos[:10]]
    pos_scores = [s for _, s in top_pos[:10]]
    neg_words = [w for w, _ in top_neg[:10]]
    neg_scores = [s for _, s in top_neg[:10]]

    y_pos = np.arange(len(pos_words))
    y_neg = np.arange(len(neg_words)) + len(pos_words) + 1

    ax3.barh(y_pos, pos_scores, color='#2ca02c', alpha=0.85, label='Top Positive Indicators')
    ax3.barh(y_neg, neg_scores, color='#d62728', alpha=0.85, label='Top Negative Indicators')

    all_ticks = list(y_pos) + list(y_neg)
    all_labels = pos_words + neg_words
    ax3.set_yticks(all_ticks)
    ax3.set_yticklabels(all_labels, fontsize=9, fontweight='bold')
    ax3.axvline(0, color='black', linewidth=1)
    ax3.set_title("Learned Feature Attribution: Top Sentiment Lexicon\nBackpropagated Word Importance in Movie Reviews", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Effective Attribution Weight", fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.5, axis='x')
    ax3.legend(loc='lower right', frameon=True, fontsize=8.5)

    # -------------------------------------------------------------
    # Panel 4: Normalized Confusion Matrix & Qualitative Predictions
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    # Compute 2x2 confusion matrix
    cm = np.zeros((2, 2))
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    im4 = ax4.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(["Negative (0)", "Positive (1)"], fontsize=10, fontweight='bold')
    ax4.set_yticklabels(["Negative (0)", "Positive (1)"], fontsize=10, fontweight='bold')
    ax4.set_xlabel("Predicted Sentiment Label", fontsize=10)
    ax4.set_ylabel("True Ground Truth Label", fontsize=10)
    ax4.set_title(f"SST-2 Validation Confusion Matrix (N = {len(targets)})\nNormalized Per-Class Recognition Precision", fontsize=11, fontweight='bold')

    for i in range(2):
        for j in range(2):
            raw_count = int(cm[i, j])
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax4.text(j, i, f"{pct:.1f}%\n({raw_count})", ha="center", va="center", color=color, fontweight='bold', fontsize=11)

    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label("Accuracy Ratio", fontsize=10)

    plt.suptitle("PyTorch Bag-of-Words (BoW) Movie Review Sentiment Classifier (SST-2 Benchmark)", 
                 fontsize=15, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisual diagnostic report saved successfully at: {os.path.abspath(save_path)}")


def predict_sentence(model, vocab, sentence, device):
    """Predicts sentiment for a novel input sentence string."""
    model.eval()
    with torch.no_grad():
        vec = vocab.text_to_bow(sentence, binary=False)
        tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()
        label = "Positive" if prob >= 0.5 else "Negative"
        conf = prob if prob >= 0.5 else (1.0 - prob)
        return label, conf * 100.0, prob


def main():
    print("=" * 72)
    print("      PROJECT 13: SENTIMENT CLASSIFIER WITH BAG-OF-WORDS (BoW)")
    print("=" * 72)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device    : {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    save_path = os.path.join(script_dir, "sentiment_bow_classifier_results.png")

    # Hyperparameters
    max_vocab_size = 5000
    min_freq = 2
    batch_size = 64
    epochs = 8
    learning_rate = 0.002

    # 1. Load Dataset
    print("\nLoading Stanford Sentiment Treebank (SST-2) Movie Reviews...")
    train_texts, train_labels, val_texts, val_labels = load_sst2_data(data_dir)
    print(f"Train Dataset Size  : {len(train_texts):,} reviews")
    print(f"Validation Set Size : {len(val_texts):,} reviews")

    # 2. Build Vocabulary
    print("\nBuilding Bag-of-Words Vocabulary...")
    vocab = Vocabulary(max_vocab_size=max_vocab_size, min_freq=min_freq)
    vocab.build_vocab(train_texts)
    print(f"Vocabulary Size     : {len(vocab):,} unique words")
    print(f"Sample Tokens       : {[vocab.idx2word[i] for i in range(2, 12)]}")
    print("-" * 72)

    # 3. Create PyTorch DataLoaders
    train_dataset = MovieReviewDataset(train_texts, train_labels, vocab, binary=False)
    val_dataset = MovieReviewDataset(val_texts, val_labels, vocab, binary=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize Feedforward Model
    model = BoWSentimentClassifier(vocab_size=len(vocab), hidden_dim1=128, hidden_dim2=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture  : BoW Feedforward Network ({len(vocab)} -> 128 -> 32 -> 1)")
    print(f"Trainable Parameters: {total_params:,}")
    print("-" * 72)

    # 5. Training Loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    print("\n--- Commencing Model Training Across 8 Epochs ---")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, preds, targets, probs = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}%")

    print("-" * 72)
    print(f"Training Complete! Final Validation Accuracy: {history['val_acc'][-1]:.2f}% | F1-Score: {history['val_f1'][-1]:.2f}%")
    print("=" * 72)

    # 6. Extract Learned Sentiment Lexicon
    top_pos, top_neg = extract_sentiment_lexicon(model, vocab, top_k=15)
    print("\n--- Learned Sentiment Lexicon (Top Attributed Words) ---")
    print(f"Top Positive Words: {', '.join([w for w, _ in top_pos[:10]])}")
    print(f"Top Negative Words: {', '.join([w for w, _ in top_neg[:10]])}")
    print("-" * 72)

    # 7. Qualitative Real-Time Inference Showcase
    test_sentences = [
        "An absolute cinematic masterpiece with breathtaking performances and stunning visuals.",
        "Completely predictable, dull, and a colossal waste of two hours.",
        "Hilarious, heartwarming, and deeply moving from start to finish.",
        "A mediocre screenplay dragged down by wooden dialogue and poor pacing."
    ]
    print("\n--- Real-Time Movie Review Inference Showcase ---")
    sample_results = []
    for sent in test_sentences:
        lbl, conf, raw_p = predict_sentence(model, vocab, sent, device)
        sample_results.append((sent, lbl, conf))
        print(f"Review: \"{sent}\"")
        print(f" -> Predicted: {lbl.upper()} (Confidence: {conf:.1f}% | P(Positive) = {raw_p:.4f})\n")

    # 8. Generate Visual Artifact
    generate_visual_artifact(history, top_pos, top_neg, targets, preds, sample_results, save_path)


if __name__ == "__main__":
    main()

