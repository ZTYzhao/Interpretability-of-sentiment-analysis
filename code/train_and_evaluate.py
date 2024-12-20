import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer
from data_preprocessing import load_data, CustomDataset
from multitask_model import MultiTaskModel

def train(model, dataloaders, optimizer, criterion, epochs=3, device='cuda'):
    model.to(device)
    for epoch in range(epochs):
        for task in ['confusion', 'post_type']:
            model.train()
            total_loss, total_samples = 0, 0

            for batch in dataloaders[task]['train']:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, task=task)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

            epoch_loss = total_loss / total_samples
            print(f"Epoch {epoch+1} [{task}] Loss: {epoch_loss:.4f}")

def evaluate(model, dataloaders, device='cuda'):
    model.eval()
    results = {}
    for task in ['confusion', 'post_type']:
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dataloaders[task]['val']:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask, task=task)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        results[task] = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average='weighted'),
            "recall": recall_score(all_labels, all_preds, average='weighted'),
            "f1": f1_score(all_labels, all_preds, average='weighted')
        }
    return results

if __name__ == "__main__":
    # Load data
    post_type_path = 'E:\pythonProject\A Hierarchical Multi-Task Model with Visual Interpretability\Data\post_type_dataset.xlsx'
    confusion_path = 'E:\pythonProject\A Hierarchical Multi-Task Model with Visual Interpretability\Data\confusion_dataset.xlsx'
    (post_type_texts, post_type_labels), (confusion_texts, confusion_labels) = load_data(post_type_path, confusion_path)

    # Tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 128

    post_type_dataset = CustomDataset(post_type_texts, post_type_labels, tokenizer, max_length)
    confusion_dataset = CustomDataset(confusion_texts, confusion_labels, tokenizer, max_length)

    dataloaders = {
        'confusion': {
            'train': DataLoader(confusion_dataset, batch_size=16, shuffle=True),
            'val': DataLoader(confusion_dataset, batch_size=16)
        },
        'post_type': {
            'train': DataLoader(post_type_dataset, batch_size=16, shuffle=True),
            'val': DataLoader(post_type_dataset, batch_size=16)
        }
    }

    # Initialize model
    model = MultiTaskModel(num_labels_confusion=3, num_labels_post_type=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Train and evaluate
    train(model, dataloaders, optimizer, criterion)
    results = evaluate(model, dataloaders)
    print(f"Results: {results}")
