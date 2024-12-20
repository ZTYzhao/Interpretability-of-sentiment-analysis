import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    DebertaTokenizer, DebertaForSequenceClassification
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    """Dataset class for tokenized data."""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data(post_type_path, confusion_path):
    """Load post-type dataset and confusion dataset."""
    # Load post-type dataset
    post_type_data = pd.read_excel(E:\pythonProject\A Hierarchical Multi-Task Model with Visual Interpretability\Data\post_type_dataset.xlsx)
    post_type_texts = post_type_data['text'].tolist()
    post_type_labels = post_type_data['label'].tolist()

    # Load confusion dataset
    confusion_data = pd.read_excel(E:\pythonProject\A Hierarchical Multi-Task Model with Visual Interpretability\Data\confusion_dataset.xlsx)
    confusion_texts = confusion_data['text'].tolist()
    confusion_labels = confusion_data['label'].tolist()

    return (post_type_texts, post_type_labels), (confusion_texts, confusion_labels)

def train_model(model, dataloader, optimizer, criterion, device):

    model.train()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader, device):

    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, average="weighted"),
        "recall": recall_score(true_labels, predictions, average="weighted"),
        "f1": f1_score(true_labels, predictions, average="weighted")
    }
    return metrics

if __name__ == "__main__":

    post_type_path = r'E:\pythonProject\A Hierarchical Multi-Task Model with Visual Interpretability\Data\post_type_dataset.xlsx'
    confusion_path = r'E:\pythonProject\A Hierarchical Multi-Task Model with Visual Interpretability\Data\confusion_dataset.xlsx'

    (post_type_texts, post_type_labels), (confusion_texts, confusion_labels) = load_data(post_type_path, confusion_path)

    models = {
        'bert': (BertTokenizer.from_pretrained("bert-base-uncased"), BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)),
        'roberta': (RobertaTokenizer.from_pretrained("roberta-base"), RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=6)),
        'albert': (AlbertTokenizer.from_pretrained("albert-base-v2"), AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=6)),
        'deberta': (DebertaTokenizer.from_pretrained("microsoft/deberta-base"), DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=6)),
    }


    tokenizer, model = models['bert']
    dataset = CustomDataset(post_type_texts, post_type_labels, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train and evaluate BERT model
    train_model(model, dataloader, optimizer, criterion, device)
    bert_metrics = evaluate_model(model, dataloader, device)
    print(f"BERT Post-type Metrics: {bert_metrics}")


    tokenizer, model = models['roberta']
    roberta_dataset = CustomDataset(post_type_texts, post_type_labels, tokenizer, max_length=128)
    roberta_dataloader = DataLoader(roberta_dataset, batch_size=2)

    model.to(device)
    train_model(model, roberta_dataloader, optimizer, criterion, device)
    roberta_metrics = evaluate_model(model, roberta_dataloader, device)
    print(f"RoBERTa Post-type Metrics: {roberta_metrics}")

    tokenizer, model = models['albert']
    albert_dataset = CustomDataset(post_type_texts, post_type_labels, tokenizer, max_length=128)
    albert_dataloader = DataLoader(albert_dataset, batch_size=2)

    model.to(device)
    train_model(model, albert_dataloader, optimizer, criterion, device)
    albert_metrics = evaluate_model(model, albert_dataloader, device)
    print(f"ALBERT Post-type Metrics: {albert_metrics}")

    tokenizer, model = models['deberta']
    deberta_dataset = CustomDataset(post_type_texts, post_type_labels, tokenizer, max_length=128)
    deberta_dataloader = DataLoader(deberta_dataset, batch_size=2)

    model.to(device)
    train_model(model, deberta_dataloader, optimizer, criterion, device)
    deberta_metrics = evaluate_model(model, deberta_dataloader, device)
    print(f"DeBERTa Post-type Metrics: {deberta_metrics}")


