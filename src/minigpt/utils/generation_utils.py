import torch
from typing import Any

import torch.nn as nn


def text_to_token_ids(text: str, tokenizer: Any) -> torch.Tensor:
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: Any) -> str:
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_sample(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    start_context: str,
) -> str:
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = model.generate(idx=encoded, max_new_tokens=50)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
    model.train()
    return decoded_text
