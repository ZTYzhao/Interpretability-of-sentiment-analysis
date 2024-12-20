import torch
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from cgan_model import Generator, Discriminator
from train_cgan import train_cgan, generate_augmented_data
from your_model_file import MultiTaskModel  # Import your actual model

class CaptumVisualization:
    def __init__(self, model, tokenizer, epsilon=1.0, learning_rate=2e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.ig = IntegratedGradients(self.model)

    def train(self, train_dataloader):
        self.model.train()
        for batch in train_dataloader:
            inputs = batch['input_ids']
            labels = batch['labels']
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def evaluate(self, test_dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch['input_ids']
                labels = batch['labels']
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        return accuracy, precision, recall, f1

    def visualize_word_importance(self, input_text, target_label):

        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs["input_ids"]


        attributions, delta = self.ig.attribute(input_ids, target=target_label, return_convergence_delta=True)


        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        word_attributions = attributions[0].detach().cpu().numpy()


        word_contributions = [(tokens[i], word_attributions[i]) for i in range(len(tokens))]


        viz.visualize_text([viz.Text(f"Token: {token} Contribution: {attribution}") 
                            for token, attribution in word_contributions])



def evaluate_complete_model():

    post_type_path = 'E:\pythonProject\A Hierarchical Multi-Task Model with Visual Interpretability\Data\post_type_dataset.xlsx'
    confusion_path = 'E:\pythonProject\A Hierarchical Multi-Task Model with Visual Interpretability\Data\confusion_dataset.xlsx'
    (post_type_texts, post_type_labels), (confusion_texts, confusion_labels) = load_data(post_type_path, confusion_path)


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 128
    post_type_dataset = CustomDataset(post_type_texts, post_type_labels, tokenizer, max_length)
    confusion_dataset = CustomDataset(confusion_texts, confusion_labels, tokenizer, max_length)


    generator = Generator()
    discriminator = Discriminator()
    train_cgan(generator, discriminator, DataLoader(confusion_dataset, batch_size=16))
    augmented_data, augmented_labels = generate_augmented_data(generator, DataLoader(confusion_dataset, batch_size=16))


    combined_data = torch.cat([confusion_texts, augmented_data])
    combined_labels = torch.cat([confusion_labels, augmented_labels])

 
    model = MultiTaskModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()


    dataloaders = {
        'confusion': DataLoader(CustomDataset(combined_data, combined_labels, tokenizer, max_length), batch_size=16)
    }
    train_with_pgd_vat(model, dataloaders, optimizer, criterion, pgd=True, vat=False)
    metrics = evaluate(model, dataloaders)
    print(f"Complete Model Metrics: {metrics}")


    captum_viz = CaptumVisualization(model, tokenizer)

 
    input_text = confusion_texts[0]  # Choose a sample from your test set
    target_label = confusion_labels[0]  # Corresponding label for the sample


    captum_viz.visualize_word_importance(input_text, target_label)

