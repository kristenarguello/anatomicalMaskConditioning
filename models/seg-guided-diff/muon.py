"""
Muon optimizer + hybrid Muon/AdamW wrapper.

Muon (MomentUm Orthogonalized by Newton-schulz) applies Nesterov momentum
to the gradient and then orthogonalizes the update via a Newton-Schulz
iteration before applying it to the weights. This effectively normalizes
the singular values of the update matrix, giving every direction an equal
step size — similar in spirit to second-order methods but much cheaper.

Muon is only valid for 2-D weight matrices and 4-D Conv weights (reshaped
to 2-D). Biases, LayerNorm/GroupNorm scales, and embedding tables use
AdamW instead. get_muon_and_adamw_params() performs this split.

Reference: https://github.com/KellerJordan/modded-nanogpt
"""

import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Approximate the nearest orthogonal matrix to G via 5 Newton-Schulz steps.
    Works in bfloat16 internally for speed; output is cast back to G.dtype.
    Input must be 2-D with shape (m, n).
    """
    assert G.ndim == 2, f"Expected 2-D tensor, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    # Normalize so the largest singular value is <= 1
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------

class Muon(Optimizer):
    """
    Muon — MomentUm Orthogonalized by Newton-schulz.

    For each parameter p with gradient g:
      1. Accumulate Nesterov momentum: buf = momentum * buf + g
                                       g   = g + momentum * buf   (if nesterov)
      2. Reshape to 2-D if needed (Conv4D → (C_out, C_in*kH*kW))
      3. Orthogonalize: g = NewtonSchulz(g)
      4. Scale by sqrt(max(1, m/n)) to preserve update RMS
      5. Apply: p ← p - lr * g

    Only use Muon for parameters with ndim >= 2 (weight matrices / conv kernels).
    Use AdamW for everything else (biases, norms, embeddings).

    Args:
        params:     iterable of parameters (should all have ndim >= 2)
        lr:         learning rate (default: 1e-4)
        momentum:   Nesterov momentum coefficient (default: 0.95)
        nesterov:   whether to use Nesterov momentum (default: True)
        ns_steps:   number of Newton-Schulz iterations (default: 5)
    """

    def __init__(self, params, lr: float = 1e-4, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.data

                # Initialize momentum buffer
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf.clone()

                # Orthogonalize: reshape 4-D conv weights to 2-D first
                orig_shape = g.shape
                if g.ndim == 4:
                    g = g.reshape(g.size(0), -1)
                if g.ndim == 2:
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    # Scale so update RMS matches a unit-variance Gaussian update
                    g = g * (max(1, g.size(0) / g.size(1)) ** 0.5)
                g = g.reshape(orig_shape)

                p.data.add_(g, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Parameter split helper
# ---------------------------------------------------------------------------

def get_muon_and_adamw_params(model: torch.nn.Module):
    """
    Split model parameters into two lists:
      - muon_params:  ndim >= 2  → weight matrices / conv kernels  → use Muon
      - adamw_params: ndim == 1  → biases, norm scales/shifts       → use AdamW

    Parameters that don't require gradients are skipped.
    """
    muon_params, adamw_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


# ---------------------------------------------------------------------------
# Hybrid Muon + AdamW wrapper
# ---------------------------------------------------------------------------

class HybridMuonAdamW:
    """
    Combines Muon (for weight matrices) and AdamW (for biases / norms).

    Exposes the same interface as a standard optimizer — .step(),
    .zero_grad(), .state_dict(), .load_state_dict() — so it drops in
    to an existing training loop with minimal changes.

    Because Muon does its own gradient manipulation it is not compatible
    with GradScaler.step().  Pass a HybridMuonAdamW to the training loop
    and use step_with_scaler(scaler) instead of scaler.step(optimizer).

    Args:
        muon_params:        parameters for Muon  (from get_muon_and_adamw_params)
        adamw_params:       parameters for AdamW (from get_muon_and_adamw_params)
        muon_lr:            Muon learning rate       (default: 1e-4)
        adamw_lr:           AdamW learning rate      (default: 3e-5)
        adamw_weight_decay: AdamW weight decay       (default: 1e-2)
    """

    def __init__(self, muon_params, adamw_params,
                 muon_lr: float = 1e-4, adamw_lr: float = 3e-5,
                 adamw_weight_decay: float = 1e-2):
        self.muon  = Muon(muon_params,  lr=muon_lr)
        self.adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr,
                                       weight_decay=adamw_weight_decay)

    # ------------------------------------------------------------------
    # Core optimizer interface
    # ------------------------------------------------------------------

    def step(self):
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])

    # ------------------------------------------------------------------
    # AMP / GradScaler integration
    # ------------------------------------------------------------------

    def step_with_scaler(self, scaler: torch.amp.GradScaler):
        """
        Drop-in replacement for scaler.step(optimizer) when using AMP.

        Muon is not compatible with GradScaler.step() because it
        manipulates the raw gradient tensor directly rather than following
        the standard grad → unscale → clip → step flow.

        This method:
          1. Manually unscales Muon parameter gradients.
          2. Steps Muon directly.
          3. Lets scaler.step() handle AdamW normally (it re-uses the
             already-unscaled state tracked by the scaler).
        """
        # Manually unscale Muon grads (scaler only tracks adamw)
        inv_scale = 1.0 / scaler.get_scale()
        for group in self.muon.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.mul_(inv_scale)

        self.muon.step()

        # Let the scaler handle AdamW (unscales + checks for inf/nan)
        scaler.step(self.adamw)
