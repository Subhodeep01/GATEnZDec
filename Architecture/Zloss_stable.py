import torch
import torch.nn.functional as F
from torch.distributions import NegativeBinomial

def zinb_nll_stable(x,  # raw counts, (N, G)
                    pi, theta, mu,
                    eps: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable Zero-Inflated Negative Binomial NLL.
    Returns a scalar loss (mean over samples).
    """

    # -- NB log-prob  -------------------------------------------------------
    #  log NB(x | ?, µ) =  lgamma(x+?) – lgamma(?) – lgamma(x+1)
    #                    + ? * log(?)        – ? * log(?+µ)
    #                    + x  * log(µ)        – x  * log(?+µ)
    t1 = torch.lgamma(theta + x) - torch.lgamma(theta) - torch.lgamma(x + 1)
    t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    t3 = x     * (torch.log(mu    + eps) - torch.log(theta + mu + eps))
    log_nb = t1 + t2 + t3                       # (N, G)

    # -- mixture with "extra zero" component --------------------------------
    zero_mask = (x <= 0.0).float()
    log_zero  = torch.log(pi      + (1.0 - pi) * torch.exp(log_nb) + eps)
    log_all   = torch.log(1.0 - pi + eps) + log_nb

    log_prob  = zero_mask * log_zero + (1.0 - zero_mask) * log_all
    nll       = -log_prob.sum(dim=1).mean()     # scalar
    return nll


def sample_zinb(pi, mu, theta):
    """
    Draw a sample from a ZINB for every element in pi/mu/theta.
    Returns an integer tensor with the same shape.
    """
    # Decide which observations are drop-outs
    dropout_mask = torch.bernoulli(pi).bool()          # True ? force zero
    # Build a torch NB distribution (parameterised by total_count=?, probs=p)
    p = mu / (mu + theta)                              # success prob for NB
    nb = NegativeBinomial(total_count=theta, probs=p)
    nb_sample = nb.sample()
    x = torch.where(dropout_mask, torch.zeros_like(nb_sample), nb_sample)
    return x.type_as(mu)