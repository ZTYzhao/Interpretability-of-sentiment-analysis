from cgan_model import Generator, Discriminator
from train_cgan import train_cgan, generate_augmented_data

def evaluate_complete_model():
    # Load data
    post_type_path = 'E:\pythonProject\A Hierarchical Multi-Task Model with Visual Interpretability\Data\post_type_dataset.xlsx'
    confusion_path = 'E:\pythonProject\A Hierarchical Multi-Task Model with Visual Interpretability\Data\confusion_dataset.xlsx'
    (post_type_texts, post_type_labels), (confusion_texts, confusion_labels) = load_data(post_type_path, confusion_path)

    # Tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 128
    post_type_dataset = CustomDataset(post_type_texts, post_type_labels, tokenizer, max_length)
    confusion_dataset = CustomDataset(confusion_texts, confusion_labels, tokenizer, max_length)

    # Initialize cGAN
    generator = Generator()
    discriminator = Discriminator()
    train_cgan(generator, discriminator, DataLoader(confusion_dataset, batch_size=16))
    augmented_data, augmented_labels = generate_augmented_data(generator, DataLoader(confusion_dataset, batch_size=16))

    # Combine augmented and original data
    combined_data = torch.cat([confusion_texts, augmented_data])
    combined_labels = torch.cat([confusion_labels, augmented_labels])

    # Initialize multitask model
    model = MultiTaskModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Train with SAL + cGAN
    dataloaders = {
        'confusion': DataLoader(CustomDataset(combined_data, combined_labels, tokenizer, max_length), batch_size=16)
    }
    train_with_pgd_vat(model, dataloaders, optimizer, criterion, pgd=True, vat=False)
    metrics = evaluate(model, dataloaders)
    print(f"Complete Model Metrics: {metrics}")
