def pgd_attack(model, inputs, labels, optimizer, criterion, epsilon=0.1, alpha=0.02, num_iter=3):
    """Apply Projected Gradient Descent (PGD) for adversarial training."""
    perturbed_inputs = inputs.clone().detach().requires_grad_(True)
    
    for _ in range(num_iter):
        optimizer.zero_grad()
        logits = model(perturbed_inputs)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Update inputs with gradient
        perturbed_inputs = perturbed_inputs + alpha * perturbed_inputs.grad.sign()
        # Clamp perturbation within allowed range
        perturbed_inputs = torch.clamp(perturbed_inputs, inputs - epsilon, inputs + epsilon).detach().requires_grad_(True)
    
    return perturbed_inputs
