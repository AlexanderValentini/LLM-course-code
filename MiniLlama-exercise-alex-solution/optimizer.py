from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            #p is a parameter group, which is a dictionary containing the parameters and hyperparameters.
            #Normally when you create an instance of your AdamW optimizer, you pass it the model's parameters. Recall calling model parameters at the EPFL course to check 
            #the similarity of the parameters of two models.
            #This is the same as calling model.parameters() in PyTorch. 
            #So, p is the parameter object. And p.data accesses the raw underlying data tensor of the parameter. 
            #Using p.data is a way to perform direct, in-place modifications on the parameter's values without these modifications being tracked by the autograd system.

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

#                raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]
                if len(state) == 0:
                    # Initialize state variables
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                state["step"] += 1
                #Update the first and second moments of the gradients. This corresponds to the equations with m_t and v_t in the paper.
                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2

                # Bias correction
                # Please note that we are using the "efficient version" given in section 2 of the paper.
                # https://arxiv.org/abs/1412.6980
                #Please read the very last paragraph of the section (before 2.1).

                # Update parameters
                alpha_t = alpha * (1 - beta2 ** state["step"]) ** 0.5 / (1 - beta1 ** state["step"])
                p.data -= alpha_t * state["m"] / (state["v"] ** 0.5 + eps)

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update. This formula comes from the principle of decoupled weight decay. 
                #And originally from simple SGD with weight decay.
                if group["weight_decay"] != 0:
                    p.data -= group["weight_decay"] * alpha * p.data

        return loss