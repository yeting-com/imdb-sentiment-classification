# imdb-sentiment-classification
Transformer-based sentiment classification using DistilBERT and RoBERTa

## Project Overview
This project focuses on binary sentiment classification of movie reviews using the IMDB dataset. The goal is to classify reviews as positive or negative based on raw text.

## Dataset
- ~25,000 movie reviews
- Balanced positive and negative labels
- Each sample contains review text and sentiment label

## Approach
The project implements an end-to-end machine learning pipeline:

- Text preprocessing (HTML removal, normalization, cleaning)
- Exploratory data analysis (EDA)
- Tokenization using transformer tokenizers
- Model training and validation

## Models
Two transformer-based models were implemented and compared:

- **DistilBERT**: Lightweight and efficient baseline model
- **RoBERTa**: More powerful model with improved performance

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

## Key Takeaways
- Transformer-based models perform well on text classification tasks
- RoBERTa generally achieves better performance but with higher computational cost
- Proper preprocessing and validation are critical for stable performance

## Files
- `train.py`: DistilBERT training pipeline
- `train_roberta.py`: RoBERTa training pipeline
- `report.pdf`: Full project report with analysis and results

## Tech Stack
- Python
- PyTorch
- HuggingFace Transformers
- scikit-learn
