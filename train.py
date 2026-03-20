# -*- coding: utf-8 -*-
"""
Movie Review Binary Classification - DistilBERT Fine-tuning Model
Using HuggingFace Transformers for pre-trained language model fine-tuning
Meets assignment requirement: "Fine-tuning a Pre-trained Language Model"
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import re
from tqdm import tqdm
import os


# ============== Text Preprocessing ==============
def preprocess_text(text):
    """Improved text preprocessing"""
    if pd.isna(text):
        return ""
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove special characters, but keep alphanumeric, spaces, single quotes and basic punctuation
    text = re.sub(r'[^a-z0-9\s\'!?]', ' ', text)
    
    # Compress multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# ============== Parameter Configuration ==============
def get_args():
    p = argparse.ArgumentParser(description="Movie Review Binary Classification - DistilBERT Fine-tuning Model")
    p.add_argument("--train_csv", default="train.csv", help="Training set CSV")
    p.add_argument("--test_csv", default="test.csv", help="Test set CSV")
    
    # Model parameters
    p.add_argument("--model_name", default="distilbert-base-uncased", help="Pre-trained model name")
    p.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    
    # Training parameters (BERT standard hyperparameters)
    p.add_argument("--batch_size", type=int, default=16, help="Batch size (BERT recommends 16 or 32)")
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs (BERT typically needs 3-5 epochs)")
    p.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate (BERT standard: 2e-5)")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    p.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps (0 means no warmup)")
    
    # Others
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    p.add_argument("--random_state", type=int, default=42, help="Random seed")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps (for simulating larger batch size)")
    
    # Output
    p.add_argument("--save_submission", default="submission.csv", help="Prediction results CSV")
    p.add_argument("--save_model", default="best_model.pt", help="Model save path")
    p.add_argument("--save_tokenizer", default="tokenizer", help="Tokenizer save directory")
    p.add_argument("--checkpoint_dir", default="checkpoints", help="Checkpoint save directory")
    
    # Resume training
    p.add_argument("--resume", default=None, help="Resume training from checkpoint (specify checkpoint path)")
    
    return p.parse_args()


# ============== Data Preparation ==============
def prepare_data(tokenizer, texts, labels, max_length):
    """Prepare data: tokenize and convert to tensors"""
    print(f"   Tokenizing {len(texts)} texts...")
    
    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Convert to TensorDataset
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    # Convert to integer labels (0 or 1) for CrossEntropyLoss
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return TensorDataset(input_ids, attention_mask, labels_tensor)


# ============== Training Function ==============
def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=1):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(dataloader, desc="   Training", leave=False)):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Statistics
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Get predictions (use argmax when num_labels=2)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Gradient accumulation: update every accumulation_steps
        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


# ============== Validation Function ==============
def validate(model, dataloader, device, threshold=0.5):
    """
    Validation function
    threshold: Used to adjust prediction threshold (when num_labels=2, represents minimum probability for class 1)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="   Validating", leave=False):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            # Get predictions (use softmax then argmax when num_labels=2)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            # Use threshold: if probability of class 1 > threshold, predict as 1
            predictions = (probs[:, 1] > threshold).long()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


# ============== Main Process ==============
def main():
    args = get_args()
    
    # Set random seed
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_state)
        torch.cuda.manual_seed_all(args.random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    print("=" * 70)
    print("Movie Review Binary Classification - DistilBERT Fine-tuning Model")
    print("=" * 70)
    
    # 1. Load data
    print("\n1. Loading data...")
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    
    # 2. Text preprocessing
    print("\n2. Preprocessing texts...")
    train_df["text"] = train_df["text"].apply(preprocess_text)
    test_df["text"] = test_df["text"].apply(preprocess_text)
    
    X = train_df["text"].tolist()
    y = train_df["label"].tolist()
    
    # Split training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_ratio, random_state=args.random_state, stratify=y
    )
    
    print(f"   Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    # 3. Load Tokenizer and model
    print(f"\n3. Loading DistilBERT tokenizer and model ({args.model_name})...")
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2  # Binary classification: 0 or 1, using CrossEntropyLoss
    )
    model.to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 4. Prepare data
    print("\n4. Preparing datasets...")
    train_dataset = prepare_data(tokenizer, X_train, y_train, args.max_length)
    val_dataset = prepare_data(tokenizer, X_val, y_val, args.max_length)
    test_dataset = prepare_data(tokenizer, test_df["text"].tolist(), [0] * len(test_df), args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 5. Optimizer and scheduler
    print("\n5. Setting up optimizer and scheduler...")
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Calculate total steps
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    
    # Learning rate scheduler (optional warmup)
    if args.warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
        print(f"   Using linear schedule with warmup ({args.warmup_steps} steps)")
    else:
        scheduler = None
        print(f"   No warmup scheduler")
    
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Total training steps: {total_steps}")
    
    # 6. Resume training: load checkpoint (if specified)
    start_epoch = 1
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    if args.resume and os.path.exists(args.resume):
        print(f"\n6. Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 1) + 1
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"   Resumed from epoch {start_epoch}")
        print(f"   Best validation F1 so far: {best_val_f1:.4f}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 7. Training loop
    print("\n7. Training...")
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        if scheduler is not None:
            print(f"   Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        else:
            print(f"   Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.gradient_accumulation_steps
        )
        
        # Validation
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, device)
        
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model (based on F1 Score)
        if val_f1 > best_val_f1 + 0.001:  # Minimum improvement threshold
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"   ✓ New best validation F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
        
        # Save checkpoint (save every epoch for resuming training)
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_val_f1': best_val_f1,
            'best_val_loss': best_val_loss,
            'val_metrics': {
                'accuracy': val_acc,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1,
                'loss': val_loss
            },
            'args': vars(args)  # Save all parameters
        }, checkpoint_path)
        print(f"   Checkpoint saved: {checkpoint_path}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n   Loaded best model with validation F1: {best_val_f1:.4f}")
    
    # 8. Final validation set evaluation
    print("\n8. Final evaluation on validation set...")
    val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, device)
    print(f"   Accuracy: {val_acc:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall: {val_recall:.4f}")
    print(f"   F1 Score: {val_f1:.4f}")
    
    # 9. Test set prediction
    print("\n9. Predicting on test set...")
    model.eval()
    test_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="   Predicting"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # Use argmax to get predicted class
            predictions = torch.argmax(logits, dim=1)
            test_predictions.extend(predictions.cpu().numpy().astype(int))
    
    # 10. Save results
    submission = pd.DataFrame({"id": test_df["id"], "label": test_predictions})
    submission.to_csv(args.save_submission, index=False)
    print(f"   Submission saved to {args.save_submission}")
    
    # Save model and tokenizer
    model.save_pretrained(args.save_model.replace('.pt', ''))
    tokenizer.save_pretrained(args.save_tokenizer)
    print(f"   Model saved to {args.save_model.replace('.pt', '')}")
    print(f"   Tokenizer saved to {args.save_tokenizer}")
    
    # Also save PyTorch format (for compatibility)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_name': args.model_name,
            'max_length': args.max_length,
        },
        'val_metrics': {
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'loss': val_loss
        }
    }, args.save_model)
    print(f"   PyTorch checkpoint saved to {args.save_model}")
    
    # 11. Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Model: DistilBERT ({args.model_name})")
    print(f"Fine-tuning: Yes (Pre-trained Language Model)")
    print(f"Max sequence length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"\nValidation Metrics:")
    print(f"  Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall: {val_recall:.4f}")
    print(f"  F1 Score: {val_f1:.4f}")
    print(f"\nTest predictions: {len(test_predictions)}")
    print("\nSubmission (first 10 rows):")
    print(submission.head(10))


if __name__ == "__main__":
    main()
