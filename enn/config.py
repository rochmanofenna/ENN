import torch

class Config:
    """
    Central hyper-parameter container for ENN.
    Pass stream_or_path optionally; everything else
    has a sane default so you can do Config().
    """
    def __init__(self, stream_or_path: str = "demo"):
        self.stream_or_path = stream_or_path

        # architecture
        self.num_layers  = 3
        self.num_neurons = 10
        self.num_states  = 5
        self.compressed_dim = 3
        self.input_dim = 5  # Default input feature dimension

        # neuron dynamics
        self.decay_rate      = 0.1
        self.recency_factor  = 0.9
        self.buffer_size     = 5

        # sparsity / gating
        self.importance_threshold = 0.1
        self.sparsity_threshold   = 0
        self.low_power_k          = 3
        self.activation_probability = 0.2
        self.attention_threshold    = 0.6
        self.sparsity_mask = torch.ones(
            self.num_neurons, self.num_states
        )

        # optimization
        self.init_method    = "xavier"
        self.base_lr        = 1e-3
        self.max_grad_norm  = 1.0
        self.epochs         = 114  # Updated for extended training
        self.batch_size     = 32   # Optimal for ENN memory dynamics
        self.priority_threshold = 0.5
        self.l1_lambda      = 1e-4
