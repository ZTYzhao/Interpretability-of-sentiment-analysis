import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """Custom dataset class for tokenization and batching."""
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
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
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
