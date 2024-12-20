def vat_loss(model, inputs, labels, criterion, epsilon=1.0):
    """Apply Virtual Adversarial Training (VAT) for robust training."""
    inputs.requires_grad = True
    logits = model(inputs)
    loss = criterion(logits, labels)
    
    # Compute gradient of loss w.r.t. inputs
    grad = torch.autograd.grad(loss, inputs, retain_graph=True)[0]
    perturbation = epsilon * grad / (grad.norm() + 1e-8)
    
    # Add perturbation to inputs
    perturbed_inputs = inputs + perturbation
    perturbed_logits = model(perturbed_inputs)
    
    # Compute consistency loss between original and perturbed logits
    kl_div = torch.nn.KLDivLoss(reduction='batchmean')
    loss_vat = kl_div(torch.nn.functional.log_softmax(perturbed_logits, dim=1),
                      torch.nn.functional.softmax(logits, dim=1))
    return loss_vat
