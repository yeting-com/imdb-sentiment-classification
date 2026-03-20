"""
Model 2: Fine-Tuning RoBERTa
RoBERTa typically performs better than BERT/DistilBERT on sentiment analysis, serving as a pre-trained model to distinguish from DistilBERT.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def preprocess_text(text: str) -> str:
    """Preprocessing consistent with previous scripts: lowercase, remove HTML, clean special characters"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class RobertaDataset(Dataset):
    """Dataset for RoBERTa: returns input_ids, attention_mask, labels."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels  # None indicates test set
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            out["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return out


def evaluate(model, dataloader, device):
    """Calculate Accuracy, Precision, Recall, F1."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec, f1


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Model 2: Fine-Tuning RoBERTa")
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--test_file", type=str, default="test.csv")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--model_save_path", type=str, default="best_model_roberta.pt")
    parser.add_argument("--submission_path", type=str, default="submission.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load RoBERTa tokenizer and model
    print(f"Loading RoBERTa: {args.model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)

    train_df["text"] = train_df["text"].apply(preprocess_text)
    test_df["text"] = test_df["text"].apply(preprocess_text)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        test_size=args.val_split,
        random_state=args.seed,
        stratify=train_df["label"],
    )

    train_dataset = RobertaDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = RobertaDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_dataset = RobertaDataset(test_df["text"].tolist(), None, tokenizer, args.max_length)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1 = 0.0
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training loss: {train_loss:.4f}")

        acc, prec, rec, f1 = evaluate(model, val_loader, device)
        print(f"Validation - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_f1": best_f1,
                },
                args.model_save_path,
            )
            print(f"✓ Saved best model (F1: {best_f1:.4f})")

    # Load best model for final evaluation and prediction
    # If .pt is saved, can load only state_dict; here for consistency directly use current model (if last epoch didn't update, it's already best)
    if os.path.isfile(args.model_save_path):
        ckpt = torch.load(args.model_save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    print("\nFinal validation set evaluation:")
    acc, prec, rec, f1 = evaluate(model, val_loader, device)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")

    print("\nGenerating test set predictions...")
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy())

    pd.DataFrame({"id": test_df["id"], "label": test_preds}).to_csv(args.submission_path, index=False)
    print(f"✓ Submission file saved: {args.submission_path}")
    print("\nTraining completed")


if __name__ == "__main__":
    main()
