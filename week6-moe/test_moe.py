from moe_done import MoEConfig, TransformerModel
import torch
from torch import nn


VOCAB_SIZE=200
MAX_SEQ_LEN=10


def test_moe():
    config = MoEConfig(
        hidden_size=128,
        intermediate_size=512,
        num_experts=8,
        num_experts_per_token=2,
        expert_capacity_factor=1.2,
    )

    model = TransformerModel(VOCAB_SIZE, config, num_layers=4, max_seq_len=MAX_SEQ_LEN)
    input_ids = torch.randint(0, VOCAB_SIZE, (4, 10))  # (batch_size, seq_len)
    logits, total_aux_loss = model(input_ids)
    print("Logits shape:", logits.shape)
    print("Total Auxiliary Loss:", total_aux_loss)

    target_ids = torch.randint(0, VOCAB_SIZE, (4, 10))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))
    total_loss = loss + total_aux_loss
    print("Total Loss (Cross-Entropy + Aux):", total_loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
