"""Collection of hidden states from adjacent GPT-2 blocks."""

import torch

from .config import DEVICE, FIT_TOKENS, STATE_CHUNK_SIZE


def _as_tensor(output):
    """Extract the hidden-state tensor from a tensor/tuple/list output."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


@torch.no_grad()
def collect_pair_states(model, text, tokenizer, L,
                        total_tokens=FIT_TOKENS,
                        chunk_size=STATE_CHUNK_SIZE):
    """
    Collect:
      X  = input to block L
      Y1 = output of block L
      Y2 = output of block L+1
    """
    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    ids = ids[:total_tokens]

    X_list = []
    Y1_list = []
    Y2_list = []

    d = model.config.n_embd

    def pre_hook_L(module, inputs):
        h = inputs[0].detach()
        X_list.append(h.reshape(-1, d).cpu().float())

    def post_hook_L(module, inputs, output):
        h = _as_tensor(output).detach()
        Y1_list.append(h.reshape(-1, d).cpu().float())

    def post_hook_L1(module, inputs, output):
        h = _as_tensor(output).detach()
        Y2_list.append(h.reshape(-1, d).cpu().float())

    handles = [
        model.transformer.h[L].register_forward_pre_hook(pre_hook_L),
        model.transformer.h[L].register_forward_hook(post_hook_L),
        model.transformer.h[L + 1].register_forward_hook(post_hook_L1),
    ]

    try:
        for start in range(0, len(ids), chunk_size):
            x = ids[start:start + chunk_size]

            if x.numel() < 2:
                continue

            _ = model(
                x.unsqueeze(0).to(DEVICE),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    X = torch.cat(X_list, dim=0)
    Y1 = torch.cat(Y1_list, dim=0)
    Y2 = torch.cat(Y2_list, dim=0)

    return X, Y1, Y2
