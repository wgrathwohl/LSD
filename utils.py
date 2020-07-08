import os
import math
from numbers import Number
import logging
import torch
from tqdm import tqdm
import numpy as np
import torch.distributions as distributions
import torch.nn as nn

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)


def isnan(tensor):
    return (tensor != tensor)


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()



class HMCSampler(nn.Module):
    def __init__(self, f, eps, n_steps, init_sample, scale_diag=None, covariance_matrix=None, device=None):
        super(HMCSampler, self).__init__()
        self.init_sample = init_sample
        self.f = f
        self.eps = eps
        if scale_diag is not None:
            self.p_dist = distributions.Normal(loc=0., scale=scale_diag.to(device))
        else:
            self.p_dist = distributions.MultivariateNormal(loc=torch.zeros_like(covariance_matrix)[:, 0].to(device),
                                                           covariance_matrix=covariance_matrix.to(device))
        self.n_steps = n_steps
        self.device = device
        self._accept = 0.

    def _grad(self, z):
        return torch.autograd.grad(-self.f(z).sum(), z, create_graph=True)[0]

    def _kinetic_energy(self, p):
        return -self.p_dist.log_prob(p).view(p.size(0), -1).sum(dim=-1)

    def _energy(self, x, p):
        k = self._kinetic_energy(p)
        pot = -self.f(x)
        return k + pot

    def initialize(self):
        x = self.init_sample()
        return x

    def _proposal(self, x, p):
        g = self._grad(x.requires_grad_())
        xnew = x
        gnew = g
        for _ in range(self.n_steps):
            p = p - self.eps * gnew / 2.
            xnew = (xnew + self.eps * p)
            gnew = self._grad(xnew.requires_grad_())
            xnew = xnew#.detach()
            p = p - self.eps * gnew / 2.
        return xnew, p

    def step(self, x):
        p = self.p_dist.sample_n(x.size(0))
        pc = torch.clone(p)
        xnew, pnew = self._proposal(x, p)
        assert (p == pc).all().float().item() == 1.0
        Hnew = self._energy(xnew, pnew)
        Hold = self._energy(x, p)

        diff = Hold - Hnew
        shape = [i if no == 0 else 1 for (no, i) in enumerate(x.shape)]

        accept = (diff.exp() >= torch.rand_like(diff)).to(x).view(*shape)
        x = accept * xnew + (1. - accept) * x
        self._accept = accept.mean()
        return x.detach()

    def sample(self, n_steps):
        x = self.initialize().to(self.device)
        t = tqdm(range(n_steps))
        accepts = []
        for _ in t:
            x = self.step(x)
            t.set_description("Acceptance Rate: {}".format(self._accept))
            accepts.append(self._accept.item())
        accepts = np.mean(accepts)
        if accepts < .4:
            self.eps *= .67
            print("Decreasing epsilon to {}".format(self.eps))
        elif accepts > .9:
            self.eps *= 1.33
            print("Increasing epsilon to {}".format(self.eps))
        return x


if __name__ == "__main__":
    x = torch.randn((1000, 3))
    c = cov(x)
    print(c)
