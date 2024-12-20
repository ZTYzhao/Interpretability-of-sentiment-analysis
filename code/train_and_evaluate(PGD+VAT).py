def train_with_pgd_vat(model, dataloaders, optimizer, criterion, pgd=False, vat=False, device='cuda', epochs=3):
    """Train model with optional PGD or VAT."""
    model.to(device)
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            total_loss = 0
            for batch in dataloaders['confusion']:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                # Apply PGD or VAT if enabled
                if pgd:
                    perturbed_inputs = pgd_attack(model, input_ids, labels, optimizer, criterion)
                    logits = model(perturbed_inputs, attention_mask)
                    loss += criterion(logits, labels)
                if vat:
                    loss += vat_loss(model, input_ids, labels, criterion)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, {phase} Loss: {total_loss/len(dataloaders['confusion'])}")
