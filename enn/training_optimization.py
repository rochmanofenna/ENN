import torch

class MetaLearningRateScheduler:
    def __init__(self, base_lr, stability_factor=0.5, entanglement_factor=0.3):
        """
        Initializes the meta-optimized learning rate scheduler.

        Parameters:
        - base_lr: Base learning rate.
        - stability_factor: Weight for stability-based adjustment.
        - entanglement_factor: Weight for entanglement-based adjustment.
        """
        self.base_lr = base_lr
        self.stability_factor = stability_factor
        self.entanglement_factor = entanglement_factor

    def adjust_learning_rate(self, optimizer, neuron_stability, entanglement_strength):
        """
        Adjusts learning rate based on neuron stability and entanglement.

        Parameters:
        - optimizer: Optimizer for the network.
        - neuron_stability: Tensor of neuron stability scores.
        - entanglement_strength: Tensor of entanglement strengths.
        """
        lr_adjustment = self.base_lr * (
            1 
            + self.stability_factor * (1 - torch.mean(neuron_stability)) 
            + self.entanglement_factor * torch.mean(entanglement_strength)
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_adjustment.item()
            
            
def sparse_gradient_aggregation(model, sparsity_mask):
    mask_vec = sparsity_mask.float()
    for param in model.parameters():
        if param.grad is None:
            continue
        g = param.grad
        if g.shape == mask_vec.shape:
            # Gradient has same shape as mask (e.g., both are [num_neurons])
            g *= mask_vec
        elif g.shape[0] == mask_vec.numel():
            # Gradient is [num_neurons, ...], apply mask along first dim
            g *= mask_vec.view(-1, 1)
        elif g.numel() == mask_vec.numel():
            # Gradient is a flat vector of same length, elementwise multiply
            g *= mask_vec.view(-1)
        # else: shapes don’t match mask, skip masking for this param.
    



def gradient_clipping(model, max_norm=1.0):
    """
    Clips gradients to a maximum norm to stabilize training.

    Parameters:
    - model: Neural network model.
    - max_norm: Maximum norm for clipping.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    
