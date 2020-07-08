import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.optim as optim
import numpy as np
import networks
import argparse
import utils

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from visualize_flow import visualize_transform
import torch.nn.utils.spectral_norm as spectral_norm


device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]


def distance_matrix(x):
    x0, x1 = x[None, :, :], x[:, None, :]
    diff = x0 - x1
    dists = diff.norm(dim=2, p=2)
    return dists


def pairwise_distances(x, y):
    diff = x - y
    dists = diff.norm(dim=1, p=2)
    return dists


def rbf(x, y, h):
    assert x.shape == y.shape
    x_ = x[None, :, :]
    y_ = y[:, None, :]
    return torch.exp((-1 / 2 / h ** 2) * torch.sum(torch.pow(x_-y_, 2), dim=-1))


def poly(x, y, h):
    assert x.shape == y.shape
    x_ = x[None, :, :]
    y_ = y[:, None, :]
    return (1. + (1. / 2. / h) * torch.sum(torch.pow(x_ - y_, 2), dim=-1)) ** (-h)


def pairwise_rbf(x, y, h):
    assert x.shape == y.shape
    return torch.exp((-1 / 2 / h ** 2) * torch.sum(torch.pow(x - y, 2), dim=-1))


def vectorized_t1(sq, K):
    return ((1. - torch.eye(sq.size(0)).to(device)) * K * torch.matmul(sq, sq.t())).sum()


def vectorized_t2(sq, grad_K):
    sum_dK_dx_sq = (sq * grad_K).sum(1)
    return -sum_dK_dx_sq.sum()


def vectorized_t4(x, grad_K):
    eps = torch.randn_like(x)
    eps_T_H = torch.autograd.grad(grad_K, x, grad_outputs=eps, create_graph=True, retain_graph=True)[0]
    trH = (eps_T_H * eps).sum(1)
    return -trH.sum() / 2


def brute_force_ksd(x, bandwidth):
    fx = ebm(x)
    sq = torch.autograd.grad(fx.sum(), x, create_graph=True, retain_graph=True)[0]
    # computation O(n^2)
    K = rbf(x, x, bandwidth)
    bs = x.size(0)

    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    for i in range(bs):
        for j in range(bs):
            grad_K_ij = torch.autograd.grad(K[i, j], x, retain_graph=True, create_graph=True)[0]
            if i != j:
                t1 += K[i, j] * (sq[i] * sq[j]).sum()

                #print(grad_K_ij)
                t2 += (grad_K_ij[j] * sq[i]).sum()
                t3 += (grad_K_ij[i] * sq[j]).sum()

            trace = 0
            for dim in range(x.size(1)):
                v = torch.autograd.grad(grad_K_ij[i, dim], x, retain_graph=True, create_graph=True)[0]
                trace += v[j][dim]
            t4 += trace
    print(t1, t2 + t3, t4)
    return t1, t2, t3, t4


def apply_spectral_norm(module):
    if 'weight' in module._parameters:
        spectral_norm(module)
        print("applying sn")


def pairwise_ksd(ebm, x, y, bandwidth, enc=lambda d: d):
    x.requires_grad_()
    y.requires_grad_()

    # get outputs and scores
    fx = ebm(x)
    fy = ebm(y)
    sqx = torch.autograd.grad(fx.sum(), x, create_graph=True, retain_graph=True)[0]
    sqy = torch.autograd.grad(fy.sum(), y, create_graph=True, retain_graph=True)[0]

    kxy = pairwise_rbf(enc(x), enc(y), bandwidth)

    t1 = (sqx * kxy[:, None] * sqy).sum(-1)

    dkx = torch.autograd.grad(kxy.sum(), x, create_graph=True, retain_graph=True)[0]
    dky = torch.autograd.grad(kxy.sum(), y, create_graph=True, retain_graph=True)[0]

    t2 = (sqx * dky).sum(-1)
    t3 = (dkx * sqy).sum(-1)

    eps = torch.randn_like(x)
    dkxy = torch.autograd.grad(dkx, y, grad_outputs=eps, create_graph=True, retain_graph=True)[0]
    t4 = (dkxy * eps).sum(-1)  # this is constant wrt EBM's parameters
    return t1, t2, t3, t4


def mean_scale(t1, t2, t3, t4):
    vals = t1 + t2 + t3 + t4
    return vals.mean(), vals.std()


def approx_jacobian_trace(fx, x):
    eps = torch.randn_like(fx)
    eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
    tr_dfdx = (eps_dfdx * eps).sum(-1)
    return tr_dfdx


def exact_jacobian_trace(fx, x):
    vals = []
    for i in range(x.size(1)):
        fxi = fx[:, i]
        dfxi_dxi = keep_grad(fxi.sum(), x)[:, i][:, None]
        vals.append(dfxi_dxi)
    vals = torch.cat(vals, dim=1)
    return vals.sum(dim=1)


def fsd(args, model, kernel, x):
    logp_u = model(x)
    sq = keep_grad(logp_u.sum(), x)
    fx = kernel(x)
    sq_fx = (sq * fx).sum(-1)

    if args.exact_trace:
        tr_dfdx = exact_jacobian_trace(fx, x)
    else:
        tr_dfdx = approx_jacobian_trace(fx, x)

    loss = (sq_fx + tr_dfdx).mean()
    return loss


import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from numbers import Number

class Logistic(distributions.Distribution):
    r"""
    Creates a logistic distribution parameterized by
    `loc` and `scale`.
    Example::
        >>> m = Logistic(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # logistic distributed with loc=0 and scale=1
         0.1046
        [torch.FloatTensor of size 1]
    Args:
        loc (float or Tensor or Variable): mean of the distribution (often referred to as mu)
        scale (float or Tensor or Variable): approximately standard deviation of the distribution
            (often referred to as sigma)
    """
    params = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Logistic, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        # eps = _finfo(self.scale).eps
        eps = torch.finfo(self.scale.dtype).eps
        U = self.loc.new(shape).uniform_(eps, 1 - eps)
        return self.loc + self.scale * (U.log() - (-U).log1p())

    def log_prob(self, value):
        #self._validate_log_prob_arg(value)
        x = -(value - self.loc) / self.scale
        return x - self.scale.log() - 2 * x.exp().log1p()

    def entropy(self):
        return self.scale.log() + 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--niters', type=int, default=100001)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--critic_weight_decay', type=float, default=0)
    parser.add_argument('--save', type=str, default='/tmp/test_ksd_ica')
    parser.add_argument('--test_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default="linear_kernel")
                        #choices=["linear_kernel", "all_pairs_kernel", "mle", "nce", "sm"])
    parser.add_argument('--adversarial', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--kernel', type=str, default="rbf", choices=["rbf", "neural"])
    parser.add_argument('--k_iters', type=int, default=5)
    parser.add_argument('--k_dim', type=int, default=2)
    parser.add_argument('--sn', action="store_true")
    parser.add_argument('--r_strength', type=float, default=.0)
    parser.add_argument('--quadratic', action="store_true")
    parser.add_argument('--poly', action="store_true")
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--sqrt', action="store_true")
    parser.add_argument('--scaled', action="store_true")
    parser.add_argument('--static_bandwidth', action="store_true")
    parser.add_argument('--fixed_dataset', action="store_true")
    parser.add_argument('--t_scaled', action="store_true")
    parser.add_argument('--exact_trace', action="store_true")
    parser.add_argument('--both_scaled', action="store_true")
    parser.add_argument('--k_loss_max', type=float, default=10000000000)
    parser.add_argument('--k_hinge', type=float, default=0.0)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--num_const', type=float, default=1e-6)


    args = parser.parse_args()

    # logger
    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

    logger.info(args)

    class ICA(nn.Module):
        def __init__(self, dim, reverse=False):
            super(ICA, self).__init__()
            self.dim = dim
            self.base_dist = Logistic(0, 1)
            self.reverse = reverse
            #while True:
            # resample until stable matrix is chosen
            while True:
                b = np.random.randn(dim, dim)
                c = np.linalg.cond(b)
                if c < dim:
                    break
                else:
                    print("Bad condition number {}".format(c))

            if reverse:
                self._A = nn.Parameter(torch.from_numpy(b).float())
            else:
                self._B = nn.Parameter(torch.from_numpy(b).float())

        @property
        def A(self):
            if self.reverse:
                return self._A
            else:
                return self.B.inverse()

        @property
        def B(self):
            if self.reverse:
                return self._A.inverse()
            else:
                return self._B

        def forward(self, x):
            s = x @ self.B
            return self.base_dist.log_prob(s).sum(1)

        def log_prob(self, x):
            logp_plus_Z = self(x)
            cov = self.B.det().abs().log()
            #cov = self.B.logdet()
            return logp_plus_Z + cov


        def sample(self, n):
            s = self.base_dist.sample((n, self.dim)).to(device)
            x = s @ self.A
            return x

    kernel_net = networks.SmallMLP(args.dim, n_out=args.dim)
    if args.sn:
        kernel_net.apply(apply_spectral_norm)

    np.random.seed(args.seed)
    trueICA = ICA(args.dim, reverse=False)
    modelICA = ICA(args.dim)
    
    logger.info(trueICA.B)
    logger.info(modelICA.B)

    logger.info(trueICA.A)
    logger.info(modelICA.A)

    modelICA.to(device)
    trueICA.to(device)
    kernel_net.to(device)

    fixed_data = trueICA.sample(args.batch_size)

    init_batch = fixed_data
    distances = distance_matrix(init_batch)
    init_mu = init_batch.mean(0)
    init_std = init_batch.std(0)

    log_bandwidth = nn.Parameter(distances.view(-1).median().log())  # for KSD
    log_bandwidth.data *= 0.

    approx_normalizing_const = nn.Parameter(distances.view(-1).median() * 0.)  # for NCE

    logger.info("Median distance is {}, using as bandwidth".format(log_bandwidth.exp().item()))

    optimizer = optim.Adam(list(modelICA.parameters()) + [approx_normalizing_const],
                           lr=args.lr, weight_decay=args.weight_decay, betas=(.5, .9))

    extras = []
    if args.static_bandwidth:
        kernel_optimizer = optim.Adam(list(kernel_net.parameters()), lr=args.lr, betas=(.5, .9))
    else:
        kernel_optimizer = optim.Adam(list(kernel_net.parameters()) + [log_bandwidth], lr=args.lr, betas=(.5, .9),
                                      weight_decay=args.critic_weight_decay)

    if args.kernel == "neural":
        if args.k_dim == 1:
            encoder_fn = lambda x: kernel_net(x)[:, None]
        else:
            encoder_fn = kernel_net
    else:
        encoder_fn = lambda x: x

    time_meter = utils.RunningAverageMeter(0.98)
    loss_meter = utils.RunningAverageMeter(0.98)
    ebm_meter = utils.RunningAverageMeter(0.98)

    def sample_data():
        if args.fixed_dataset:
            inds = list(range(args.batch_size))
            np.random.shuffle(inds)
            inds = torch.from_numpy(inds)
            return fixed_data[inds]
        else:
            return trueICA.sample(args.batch_size)

    best_loss = float('inf')
    modelICA.train()
    end = time.time()
    test_nlls = []
    test_losses = []
    for itr in range(args.niters):
        optimizer.zero_grad()
        kernel_optimizer.zero_grad()

        x = sample_data()
        x = x.to(device)
        x.requires_grad_()

        bandwidth = log_bandwidth.exp()

        if args.mode == "linear_kernel":  # KSD
            # use linear version to get scale estimate
            half_batch = x.size(0) // 2
            x0, x1 = x[:half_batch], x[half_batch:]
            t1, t2, t3, t4 = pairwise_ksd(modelICA, x0, x1, bandwidth, encoder_fn)
            loss, scale = mean_scale(t1, t2 ,t3, t4)
            scale = scale + 1e-3 if (args.k_scaled or args.both_scaled) else scale * 0. + 1.

            if args.adversarial and itr % (args.k_iters + 1) != 0:
                if args.k_scaled or args.both_scaled:
                    (-1. * loss.clamp(-args.k_loss_max, args.k_loss_max) / scale).backward()
                else:
                    (-1. * loss.clamp(-args.k_loss_max, args.k_loss_max) / scale.detach()).backward()
                kernel_optimizer.step()
            else:
                if args.both_scaled:
                    (loss / scale).backward()
                else:
                    (loss / scale.detach()).backward()
                optimizer.step()

            loss_meter.update(loss.item())
            time_meter.update(time.time() - end)

            if itr % args.log_freq == 0:
                log_message = (
                    'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}), Scale: {:.6f}, '
                    'Bandwidth {:.6f} | terms ({:.3f}, {:.3f}, {:.3f}, {:.3f})'.format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, scale,
                        log_bandwidth.exp().item(),
                        t1.mean().item(), t2.mean().item(), t3.mean().item(), t4.mean().item()
                    )
                )
                logger.info(log_message)

        elif args.mode.startswith("functional-"):
            l2 = float(args.mode.split("-")[1])

            logp_u = modelICA(x)
            sq = keep_grad(logp_u.sum(), x)
            fx = kernel_net(x)
            sq_fx = (sq * fx).sum(-1)

            if args.exact_trace:
                tr_dfdx = exact_jacobian_trace(fx, x)
            else:
                tr_dfdx = torch.cat([approx_jacobian_trace(fx, x)[:, None] for _ in range(args.n_samples)], dim=1).mean(dim=1)

            stats = sq_fx + tr_dfdx
            loss = stats.mean()
            l2_penalty = (fx * fx).sum(1).mean() * l2

            if itr % (args.k_iters + 1) != 0:
                if args.t_scaled or args.both_scaled:
                    (-1. * loss / (stats.std() + args.num_const) + l2_penalty).backward()
                else:
                    (-1. * loss + l2_penalty).backward()
                kernel_optimizer.step()
            else:
                if args.both_scaled:
                    (loss / (stats.std() + args.num_const)).backward()
                else:
                    loss.backward()
                optimizer.step()

            loss_meter.update(loss.item())
            time_meter.update(time.time() - end)

            if itr % args.log_freq == 0:
                log_message = (
                    'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'.format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg
                    )
                )
                logger.info(log_message)

        else:  # non-KSD training

            if args.mode == "sm":
                fx = modelICA(x)
                dfdx = torch.autograd.grad(fx.sum(), x, retain_graph=True, create_graph=True)[0]
                eps = torch.randn_like(dfdx)
                epsH = torch.autograd.grad(dfdx, x, grad_outputs=eps, create_graph=True, retain_graph=True)[0]

                trH = (epsH * eps).sum(1)
                norm_s = (dfdx * dfdx).sum(1)

                loss = (trH + .5 * norm_s).mean()

            elif args.mode == "nce":
                noise_dist = distributions.Normal(init_mu, init_std)
                x_fake = noise_dist.sample_n(x.size(0))

                pos_logits = modelICA(x) + approx_normalizing_const - noise_dist.log_prob(x).sum(1)
                neg_logits = modelICA(x_fake) + approx_normalizing_const - noise_dist.log_prob(x_fake).sum(1)

                pos_loss = nn.BCEWithLogitsLoss()(pos_logits, torch.ones_like(pos_logits))
                neg_loss = nn.BCEWithLogitsLoss()(neg_logits, torch.zeros_like(neg_logits))
                loss = pos_loss + neg_loss

            elif args.mode == "mle":
                loss = -modelICA.log_prob(x).mean()

            elif args.mode.startswith("cnce-"):
                eps = float(args.mode.split('-')[1])
                x_pert = x + torch.randn_like(x) * eps
                logits = modelICA(x) - modelICA(x_pert)
                loss = nn.BCEWithLogitsLoss()(logits, torch.ones_like(logits))

            else:
                assert False

            loss.backward(retain_graph=True)
            optimizer.step()

            loss_meter.update(loss.item())
            time_meter.update(time.time() - end)

            if itr % args.log_freq == 0:
                log_message = (
                    'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}), Approx Z {:.6f}'.format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg,
                        approx_normalizing_const.item()
                    )
                )
                logger.info(log_message)

        if itr % args.test_freq == 0 or itr == args.niters:
            modelICA.cpu()
            utils.makedirs(args.save)
            torch.save({
                'args': args,
                'state_dict': modelICA.state_dict(),
                'test_nlls': test_nlls,
                'test_losses': test_losses
            }, os.path.join(args.save, 'checkpt.pth'))
            modelICA.to(device)

        if itr % args.test_freq == 0:
            test_batch = trueICA.sample(args.test_batch_size)

            t = fsd(args, trueICA, kernel_net, test_batch)
            m = fsd(args, modelICA, kernel_net, test_batch)

            K = None

            if K is not None:
                plt.clf()
                plt.hist(K.view(-1).detach().cpu().numpy())
                utils.makedirs("{}/hists/".format(args.save))
                plt.savefig("{}/hists/K_hist_{}.pdf".format(args.save, itr))

            test_losses.append((t, m))
            logger.info("True loss: {}, Model loss: {}".format(t, m))

            t = -trueICA.log_prob(test_batch).mean().item()
            m = -modelICA.log_prob(test_batch).mean().item()

            test_nlls.append((t, m))
            logger.info("True NLL: {}, Model NLL: {}".format(t, m))

            plt.clf()
            t, m = zip(*test_losses)
            plt.plot(t, c='b')
            plt.plot(m, c='r')
            plt.savefig("{}/losses.pdf".format(args.save))

            plt.clf()
            t, m = zip(*test_nlls)
            plt.plot(t, c='b')
            plt.plot(m, c='r')
            plt.savefig("{}/nlls.pdf".format(args.save))

            plt.clf()
            plt.figure(figsize=(4, 4))
            npts = 100
            p_samples = trueICA.sample(10000).detach().cpu().numpy()[:, :2]
            q_samples = modelICA.sample(10000).detach().cpu().numpy()[:, :2]
            if args.dim == 2:
                visualize_transform([p_samples, q_samples], ["data", "model"],
                                    [trueICA, modelICA], ["data", "model"], npts=npts,
                                    low=p_samples.min(), high=p_samples.max())
            else:
                visualize_transform([p_samples, q_samples], ["data", "model"],
                                    [], [], npts=npts,
                                    low=p_samples.min(), high=p_samples.max())
            fig_filename = os.path.join(args.save, 'figs', '{:04d}.png'.format(itr))
            utils.makedirs(os.path.dirname(fig_filename))
            plt.savefig(fig_filename)
            plt.close()

        end = time.time()

    logger.info('Training has finished.')


