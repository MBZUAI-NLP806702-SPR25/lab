from torch import nn
from dataclasses import dataclass
import torch
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute frequencies for rotary position embeddings.

    Args:
        dim: Dimension of the input tensor.
        end: Maximum sequence length.
        theta: Base frequency for rotary embeddings.

    Returns:
        torch.Tensor: Precomputed frequencies, shape (end, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor, shape (batch_size, seq_len, num_heads * head_dim)
        freqs_cis: Complex frequencies, shape (seq_len, head_dim//2)

    Returns:
        torch.Tensor: Rotated tensor of same shape as input
    """
    # x_: shape (batch_size, seq_len, num_heads * head_dim//2, 2)
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)
    x_rotated = x_complex * freqs_cis[: x.shape[1], :]
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)


@dataclass
class MoEConfig:
    """Configuration dataclass for Mixture of Experts model.

    Attributes:
        hidden_size: Dimension of hidden layers.
        intermediate_size: Dimension of intermediate FFN layers.
        num_experts: Total number of experts in the model.
        num_experts_per_token: Number of experts to route each token to.
        expert_capacity_factor: Factor to determine expert capacity.
        router_jitter_noise: Noise added to router logits during training.
        router_dropout: Dropout probability for router.
        expert_dropout: Dropout probability for expert FFN.
        expert_bias: Whether to use bias in expert FFN layers.
        router_bias: Whether to use bias in router linear layer.
        load_balance_coef: Coefficient for load balancing loss.
        activation_fn: Activation function for experts ("gelu" or "selu").
        num_heads: Number of attention heads.
    """

    hidden_size: int = 64
    intermediate_size: int = 256
    num_experts: int = 10
    num_experts_per_token: int = 1
    expert_capacity_factor: float = 1.0
    router_jitter_noise: float = 0.1
    router_dropout: float = 0.1
    expert_dropout: float = 0.1
    expert_bias: bool = True
    router_bias: bool = True
    load_balance_coef: float = 0.001
    activation_fn: str = "gelu"
    num_heads: int = 1


class ExpertFFN(nn.Module):
    """Feed-forward network used as an expert in MoE.

    Architecture:
        Input -> Linear -> Activation -> Dropout -> Linear -> Output

    Args:
        config: Configuration object containing model parameters.

    Shape:
        - Input: (batch_size, seq_len, hidden_size)
        - Output: (batch_size, seq_len, hidden_size)
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.expert_bias
        )
        if config.activation_fn == "gelu":
            self.act_fn = nn.GELU()
        elif config.activation_fn == "selu":
            self.act_fn = nn.SELU()
        else:
            raise ValueError(f"Unsupported activation function: {config.activation_fn}")

        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.expert_bias
        )
        self.dropout = nn.Dropout(config.expert_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert FFN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: Processed tensor of shape (batch_size, seq_len, hidden_size)
        """
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class Router(nn.Module):
    """Routes tokens to experts using learned parameters.

    Args:
        config: Configuration object containing model parameters.

    Attributes:
        router: Linear layer for computing routing logits.
        config: Model configuration object.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.router = nn.Linear(
            config.hidden_size, config.num_experts, bias=config.router_bias
        )

    def _compute_routing_weights(
        self, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights and expert assignments.

        Args:
            router_logits: Raw routing scores, shape (batch_size, seq_len, num_experts)

        Returns:
            tuple:
                - routing_weights: Softmaxed weights, shape (batch_size, seq_len, num_experts_per_token)
                - selected_experts: Expert indices, shape (batch_size, seq_len, num_experts_per_token)
        """
        if self.training and self.config.router_jitter_noise > 0:
            router_logits += (
                torch.randn_like(router_logits) * self.config.router_jitter_noise
            )

        # get top-k experts
        routing_weights, selected_experts = torch.topk(
            router_logits, self.config.num_experts_per_token, dim=-1
        )
        routing_weights = torch.softmax(routing_weights, dim=-1)
        return routing_weights, selected_experts

    def _compute_balance_loss_switch(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss using the Switch Transformer approach.

        Args:
            router_probs: Router probability distribution, shape (batch_size, seq_len, num_experts)

        Returns:
            torch.Tensor: Scalar load balancing loss
        """
        # Calculate f_i: fraction of tokens assigned to each expert
        expert_mask = torch.argmax(router_probs, dim=-1)
        expert_mask = F.one_hot(
            expert_mask, num_classes=self.config.num_experts
        ).float()

        # Calculate mean assignment fraction for each expert
        f_i = expert_mask.mean(dim=[0, 1])  # Shape: (num_experts,)

        # Calculate P_i: mean probability assigned to each expert
        P_i = router_probs.mean(dim=[0, 1])  # Shape: (num_experts,)

        # Compute loss: N * sum(f_i * P_i)
        loss = self.config.num_experts * torch.sum(f_i * P_i)

        return loss * self.config.load_balance_coef

    def _compute_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss using KL divergence.

        Args:
            router_probs: Router probability distribution, shape (batch_size, seq_len, num_experts)

        Returns:
            torch.Tensor: Scalar load balancing loss
        """
        # expert_usage: (num_experts,)
        expert_usage = router_probs.mean(dim=[0, 1])
        target_usage = torch.ones_like(expert_usage) / self.config.num_experts
        balance_loss = torch.sum(
            expert_usage * torch.log(expert_usage / target_usage)
        )  # KL Divergence
        return balance_loss * self.config.load_balance_coef

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route input tokens to experts.

        Args:
            x: Input tensor, shape (batch_size, seq_len, hidden_size)

        Returns:
            tuple:
                - routing_weights: Softmaxed weights, shape (batch_size, seq_len, num_experts_per_token)
                - selected_experts: Expert indices, shape (batch_size, seq_len, num_experts_per_token)
                - aux_loss: Scalar load balancing loss
        """
        router_logits = self.router(x)
        routing_weights, selected_experts = self._compute_routing_weights(router_logits)

        router_probs = torch.softmax(
            router_logits, dim=-1
        )  # Use *all* logits for load balancing

        balance_loss = self._compute_balance_loss_switch(router_probs)
        aux_loss = balance_loss
        return routing_weights, selected_experts, aux_loss


class ExpertLayer(nn.Module):
    """Layer containing multiple experts and routing mechanism.

    Args:
        config: Configuration object containing model parameters.

    Attributes:
        experts: List of expert FFN modules.
        router: Router module for token-to-expert assignment.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [ExpertFFN(config) for _ in range(config.num_experts)]
        )
        self.router = Router(config)

    def _compute_capacity(self, batch_size: int, seq_len: int) -> int:
        """Compute maximum number of tokens per expert.

        Args:
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            int: Maximum number of tokens that can be processed by each expert
        """
        tokens_per_expert = (batch_size * seq_len) / self.config.num_experts # 30 tokens / 6 experts = 5 tokens per expert
        capacity = int(tokens_per_expert * self.config.expert_capacity_factor) # 5 * 1.5 = 7
        return capacity

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input through router and experts.

        Args:
            x: Input tensor, shape (batch_size, seq_len, hidden_size)

        Returns:
            tuple:
                - final_output: Processed tensor, shape (batch_size, seq_len, hidden_size)
                - aux_loss: Load balancing loss scalar
        """
        batch_size, seq_len, _ = x.size()
        capacity = self._compute_capacity(batch_size, seq_len)
        routing_weights, selected_experts, aux_loss = self.router(x)
        final_output = torch.zeros_like(x)

        for expert_idx in range(self.config.num_experts):
            # Initial mask: all tokens assigned to this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            positions = expert_mask.nonzero(as_tuple=False)
            num_selected = positions.size(0)

            if num_selected > capacity:
                # --- Capacity Limiting ---
                batch_indices, seq_indices, k_indices = (
                    selected_experts == expert_idx
                ).nonzero(as_tuple=True)
                expert_weights = routing_weights[batch_indices, seq_indices, k_indices]
                _, sorted_indices = torch.sort(expert_weights, descending=True)
                keep = sorted_indices[:capacity]

                # Create the *new* mask, keeping only the selected tokens
                new_expert_mask = torch.zeros_like(expert_mask)
                new_expert_mask[positions[keep][:, 0], positions[keep][:, 1]] = True

                # Get weights for kept tokens only.
                batch_indices, seq_indices, k_indices = (
                    selected_experts == expert_idx
                ).nonzero(as_tuple=True)
                expert_weights = routing_weights[batch_indices, seq_indices, k_indices][
                    keep
                ]

                # Update expert_mask to the NEW mask!
                expert_mask = new_expert_mask

            else:
                # No capacity limiting needed
                batch_indices, seq_indices, k_indices = (
                    selected_experts == expert_idx
                ).nonzero(as_tuple=True)
                expert_weights = routing_weights[batch_indices, seq_indices, k_indices]

            expert_input = x[expert_mask]  # Use the (potentially updated) mask
            expert_output = self.experts[expert_idx](expert_input)
            final_output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)

        # What's the problem here?
        # One step that I didn't include from the switchtransformer
        # for the discarded one 
        # redistribute randomly the tokens to the experts
        # switch transformer: select the second best exeperts
        # have another expert for discarded token

        return final_output, aux_loss


class MoEBlock(nn.Module):
    """Mixture of Experts block with normalization and residual connection.

    Args:
        config: Configuration object containing model parameters.

    Attributes:
        expert_layer: Main expert routing and processing layer.
        layer_norm: Layer normalization module.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.expert_layer = ExpertLayer(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.layer_norm(x)
        x, aux_loss = self.expert_layer(x)
        x += residual
        return x, aux_loss


class TransformerWithMoE(nn.Module):
    """Transformer layer combining attention and MoE FFN.

    Args:
        config: Configuration object containing model parameters.

    Attributes:
        attention: Multi-head attention module.
        layer_norm1: Pre-attention normalization.
        moe_block: Mixture of experts block.
        layer_norm2: Pre-MoE normalization.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            num_heads=config.num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.moe_block = MoEBlock(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.layer_norm1(x)
        x = apply_rope(x, pos_emb)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        attn_output = attn_output + residual
        x = self.layer_norm2(attn_output)
        x, aux_loss = self.moe_block(x)  # WE FOCUS HERE!
        return x, aux_loss


class TransformerModel(nn.Module):
    """Complete transformer model with mixture of experts.

    Args:
        vocab_size: Size of vocabulary.
        config: Configuration object containing model parameters.
        num_layers: Number of transformer layers.
        max_seq_len: Maximum sequence length.

    Attributes:
        embedding: Token embedding layer.
        transformer_layers: List of transformer layers.
        fc: Final linear projection to vocabulary.
        freqs_cis: Precomputed rotary position embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        config: MoEConfig,
        num_layers: int = 2,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.config = config
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, config.hidden_size)
        self.transformer_layers = nn.ModuleList(
            [TransformerWithMoE(config) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(config.hidden_size, vocab_size)
        self.freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_heads, max_seq_len * 2
        )

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input through the transformer model.

        Args:
            input_ids: Token indices, shape (batch_size, seq_len)

        Returns:
            tuple:
                - logits: Output logits, shape (batch_size, seq_len, vocab_size)
                - total_aux_loss: Sum of load balancing losses
        """
        # x: shape (batch_size, seq_len, hidden_size)
        x = self.embedding(input_ids)
        total_aux_loss = 0
        freqs_cis = self.freqs_cis.to(x.device)

        # Create the causal mask
        batch_size, seq_len = input_ids.size()
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).view(
            1, seq_len, seq_len
        )

        mask = mask.expand(batch_size, -1, -1)  # (batch_size, seq_len, seq_len)

        for layer in self.transformer_layers:
            x, aux_loss = layer(x, freqs_cis, mask)  # Pass the mask to each layer
            total_aux_loss += aux_loss

        return self.fc(x), total_aux_loss
