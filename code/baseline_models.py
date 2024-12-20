from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(post_type_path, confusion_path):
    """Load post-type dataset and confusion dataset."""
    # Load post-type dataset
    post_type_data = pd.read_excel(post_type_path)
    post_type_texts = post_type_data['text'].tolist()
    post_type_labels = post_type_data['label'].tolist()

    # Load confusion dataset
    confusion_data = pd.read_excel(confusion_path)
    confusion_texts = confusion_data['text'].tolist()
    confusion_labels = confusion_data['label'].tolist()

    return (post_type_texts, post_type_labels), (confusion_texts, confusion_labels)

def train_baseline_models(features, labels):
    """Train and evaluate Random Forest and Logistic Regression models."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_metrics = {
        "accuracy": accuracy_score(y_test, rf_preds),
        "precision": precision_score(y_test, rf_preds, average="weighted"),
        "recall": recall_score(y_test, rf_preds, average="weighted"),
        "f1": f1_score(y_test, rf_preds, average="weighted")
    }

    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=500)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    lr_metrics = {
        "accuracy": accuracy_score(y_test, lr_preds),
        "precision": precision_score(y_test, lr_preds, average="weighted"),
        "recall": recall_score(y_test, lr_preds, average="weighted"),
        "f1": f1_score(y_test, lr_preds, average="weighted")
    }

    return rf_metrics, lr_metrics

if __name__ == "__main__":

    post_type_path = r'E:\pythonProject\A Hierarchical Multi-Task Model with Visual Interpretability\Data\post_type_dataset.xlsx'
    confusion_path = r'E:\pythonProject\A Hierarchical Multi-Task Model with Visual Interpretability\Data\confusion_dataset.xlsx'

    (post_type_texts, post_type_labels), (confusion_texts, confusion_labels) = load_data(post_type_path, confusion_path)

    features = [[0] * len(post_type_texts)]  
    rf_metrics, lr_metrics = train_baseline_models(features, post_type_labels)      print(f"Random Forest Metrics: {rf_metrics}")
    print(f"Logistic Regression Metrics: {lr_metrics}")
