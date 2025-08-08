import torch
from torch.utils._pytree import tree_map

from neural_methods.model.base.ops.utils import ln_fused_l2_bwd, ln_fwd

def scan(f, init, xs, checkpoint_group=0):
    """Mimic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        sub_out_list = []
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            sub_out_list.append(y)
        sub_out = torch.stack(sub_out_list)
        return carry, sub_out

    if checkpoint_group > 0:
        out_list = []
        for k in range(0, num_items, checkpoint_group):
            carry, sub_out = torch.utils.checkpoint.checkpoint(
                scan_fn,
                carry,
                k,
                min(k + checkpoint_group, num_items),
                use_reentrant=False,
            )
            out_list.append(sub_out)
        out = torch.concatenate(out_list, dim=0)
    else:
        carry, out = scan_fn(carry, 0, num_items)

    return carry, out

def compute_mini_batch(
    params_dict,
    inputs,
):
    W1_init = params_dict["W1_states"]
    b1_init = params_dict["b1_states"]
    ttt_norm_weight = params_dict["ttt_norm_weight"]
    ttt_norm_bias = params_dict["ttt_norm_bias"]

    XQ_mini_batch = inputs["XQ"]
    XV_mini_batch = inputs["XV"]
    XK_mini_batch = inputs["XK"]

    eta_mini_batch = inputs["eta"]

    num_heads = XQ_mini_batch.size(1)
    head_dim = XQ_mini_batch.size(-1)

    X1 = XK_mini_batch
    Z1 = X1 @ W1_init + b1_init
    reconstruction_target = XV_mini_batch - XK_mini_batch

    ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
    ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)
    grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

    Attn1 = XQ_mini_batch @ X1.transpose(-2, -1)
    b1_bar = b1_init - eta_mini_batch @ grad_l_wrt_Z1
    Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

    last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
    W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
    b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)

    Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)

    XQW_mini_batch = XQ_mini_batch + Z1_bar

    last_param_dict = {
        "W1_states": W1_last,
        "b1_states": b1_last,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
    }

    return last_param_dict, XQW_mini_batch


def ttt_linear(XK, XQ, XV, eta, ttt_norm_weight, ttt_norm_bias, W1_init, b1_init, checkpoint_group_size):
    init_params_dict = {
        "W1_states": W1_init,
        "b1_states": b1_init,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
    }

    inputs = {
        "XK": XK,
        "XQ": XQ,
        "XV": XV,
        "eta": eta,
    }

    # Reorder such that mini-batch is first dimension for iteration
    inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

    XQW_batch = torch.empty_like(inputs["XK"])

    _, XQW_batch = scan(
        compute_mini_batch,  # Function to iterate over
        init_params_dict,
        inputs,
        checkpoint_group_size,
    )

    return XQW_batch.permute(1, 0, 3, 2, 4)
