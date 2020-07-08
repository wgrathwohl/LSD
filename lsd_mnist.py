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
import time
import torch.nn.utils.spectral_norm as spectral_norm
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as tr
from lsd_test import GaussianBernoulliRBM

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def logit(x, alpha=1e-6):
    x = x * (1 - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1 - x)

def get_data(args):
    if args.data_type == "continuous":
        if args.logit:
            transform = tr.Compose([tr.ToTensor(), lambda x: x * (255. / 256.) + (torch.rand_like(x) / 256.), logit])
        else:
            transform = tr.Compose([tr.ToTensor(), lambda x: x * (255. / 256.) + (torch.rand_like(x) / 256.)])
    else:
        transform = tr.Compose([tr.ToTensor(), lambda x: (x > .5).float()])
    if args.data == "mnist":
        dset_train = tv.datasets.MNIST(root="../data", train=True, transform=transform, download=True)
        dset_test = tv.datasets.MNIST(root="../data", train=False, transform=transform, download=True)
    elif args.data == "fashionmnist":
        dset_train = tv.datasets.FashionMNIST(root="../data", train=True, transform=transform, download=True)
        dset_test = tv.datasets.FashionMNIST(root="../data", train=False, transform=transform, download=True)
    else:
        assert False, "BAD BOI"

    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_test = DataLoader(dset_test, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    return dload_train, dload_test


def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]


def apply_spectral_norm(module):
    if 'weight' in module._parameters:
        spectral_norm(module)
        print("applying sn")


class EBM(nn.Module):
    def __init__(self, net, base_dist=None, learn_base_dist=True):
        super(EBM, self).__init__()
        self.net = net
        if base_dist is not None:
            self.base_mu = nn.Parameter(base_dist.loc, requires_grad=learn_base_dist)
            self.base_logstd = nn.Parameter(base_dist.scale.log(), requires_grad=learn_base_dist)
            self.base_logweight = nn.Parameter(base_dist.scale.mean() * 0., requires_grad=learn_base_dist)
        else:
            self.base_mu = None
            self.base_logstd = None

    def forward(self, x, lp=False):
        if self.base_mu is None:
            bd = 0
        else:
            base_dist = distributions.Normal(self.base_mu, self.base_logstd.exp())
            bd = base_dist.log_prob(x).view(x.size(0), -1).sum(1)
        net = self.net(x)
        if lp:
            return net + bd, net
        else:
            return net + bd

    def sample(self, x_init, l=1., e=.01, n_steps=100, anneal=None):
        x_k = torch.autograd.Variable(x_init, requires_grad=True)
        # sgld
        if anneal == "lin":
            lrs = list(reversed(np.linspace(e, l, n_steps)))
        elif anneal == "log":
            lrs = np.logspace(np.log10(l), np.log10(e))
        else:
            lrs = [l for _ in range(n_steps)]
        for this_lr in lrs:
            f_prime = torch.autograd.grad(self(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += this_lr * f_prime + torch.randn_like(x_k) * e
        final_samples = x_k.detach()
        return final_samples


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['mnist', 'fashionmnist'], type=str, default='mnist')
    parser.add_argument('--niters', type=int, default=100001)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=.5)
    parser.add_argument('--grad_l2', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--save', type=str, default='/tmp/test_ksd')
    parser.add_argument('--load', type=str)
    parser.add_argument('--viz_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--base_dist', action="store_true")
    parser.add_argument('--fixed_base_dist', action="store_true")
    parser.add_argument('--k_iters', type=int, default=1)
    parser.add_argument('--e_iters', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=1000)
    parser.add_argument('--logit', action="store_true")
    parser.add_argument('--data_type', type=str, default="continuous", choices=["continuous", "binary"])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--z_dim', type=int, default=32)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--quadratic', action="store_true")
    parser.add_argument('--data_init', action="store_true")
    parser.add_argument('--full_rank_mass', action="store_true")
    parser.add_argument('--dropout', action="store_true")
    parser.add_argument('--exact_trace', action="store_true")
    parser.add_argument('--t_scaled', action="store_true")
    parser.add_argument('--both_scaled', action="store_true")
    parser.add_argument('--rbm', action="store_true")
    parser.add_argument('--grad_crit', action="store_true")
    parser.add_argument('--e_squared', action="store_true")
    parser.add_argument('--t_squared', action="store_true")
    parser.add_argument('--tanh', action="store_true")
    parser.add_argument('--num_const', type=float, default=1e-6)
    parser.add_argument('--burn_in', type=int, default=2000)
    parser.add_argument('--arch', default='mlp', choices=["mlp", "mlp-large"])


    args = parser.parse_args()
    if args.data == "mnist" or args.data == "fashionmnist":
        args.data_dim = 784
        args.data_shape = (1, 28, 28)

    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(x.clamp(0, 1), p, normalize=False, nrow=sqrt(x.size(0)))

    dload_train, dload_test = get_data(args)

    # logger
    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

    logger.info(args)


    if args.arch == "mlp":
         if args.quadratic:
             net = networks.QuadraticMLP(args.data_dim, n_hid=args.hidden_dim)
         else:
             net = networks.SmallMLP(args.data_dim, n_hid=args.hidden_dim, dropout=args.dropout)

         critic = networks.SmallMLP(args.data_dim, n_out=args.data_dim,
                                   n_hid=args.hidden_dim, dropout=args.dropout)
    elif args.arch == "mlp-large":
        net = networks.LargeMLP(args.data_dim, n_hid=args.hidden_dim, dropout=args.dropout)
        critic = networks.LargeMLP(args.data_dim, n_out=args.data_dim,
                                   n_hid=args.hidden_dim, dropout=args.dropout)
    else:
        assert False


    if args.tanh:
        critic = nn.Sequential(critic, nn.Tanh())

    for x, _ in dload_train:
        init_batch = x.view(x.size(0), -1)
        break

    mu, std = init_batch.mean(), init_batch.std() + 1.
    if args.fixed_base_dist:
        mu = torch.ones_like(mu) * init_batch.mean()
        std = torch.ones_like(std) * init_batch.std()
        print(init_batch.mean(), init_batch.std(), init_batch.min(), init_batch.max())
    base_dist = distributions.Normal(mu, std)

    if args.rbm:
        B = torch.randn((args.data_dim, args.hidden_dim)) / args.hidden_dim
        c = torch.randn((1, args.hidden_dim))
        b = init_batch.mean(0)[None, :]
        ebm = GaussianBernoulliRBM(B, b, c, burn_in=args.burn_in)
    else:
        ebm = EBM(net, base_dist if args.base_dist else None, learn_base_dist=not args.fixed_base_dist)

    if args.load is not None:
        ckpt = torch.load(args.load)
        ebm.load_state_dict(ckpt['ebm_state_dict'])
        critic.load_state_dict(ckpt['critic_state_dict'])

    models = [ebm, critic]

    def cpu():
        for model in models:
            model.cpu()

    def gpu():
        for model in models:
            model.to(device)

    gpu()

    logger.info(ebm)
    logger.info(critic)

    if args.logit:
        init_fn = lambda: logit(torch.rand((args.batch_size, args.data_dim)))
    else:
        init_fn = lambda: torch.rand((args.batch_size, args.data_dim))

    optimizer = optim.Adam(ebm.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(.0, .9))
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(.0, .9))

    time_meter = utils.RunningAverageMeter(0.98)
    loss_meter = utils.RunningAverageMeter(0.98)
    ebm_meter = utils.RunningAverageMeter(0.98)

    best_loss = float('inf')
    ebm.train()
    critic.train()
    end = time.time()

    def stein_stats(distribution, x, critic):
        if args.rbm:
            sq = distribution.score_function(x)
        else:
            logp_u, lp = distribution(x, lp=True)
            sq = keep_grad(logp_u.sum(), x)
        fx = critic(x)
        sq_fx = (sq * fx).sum(-1)

        if args.exact_trace:
            tr_dfdx = exact_jacobian_trace(fx, x)
        else:
            tr_dfdx = torch.cat([approx_jacobian_trace(fx, x)[:, None] for _ in range(args.n_samples)], dim=1).mean(
                dim=1)

        stats = sq_fx + tr_dfdx
        norms = (fx * fx).sum(1)
        grad_norms = (sq * sq).view(x.size(0), -1).sum(1)
        return stats, norms, grad_norms, lp


    static_init = init_fn().to(device)
    itr = 0
    for epoch in range(args.epochs):
        for _itr, (x, _) in enumerate(dload_train):
            x = x.view(x.size(0), -1)
            x = x.to(device)

            optimizer.zero_grad()
            critic_optimizer.zero_grad()

            x.requires_grad_()

            # ebm training
            stats, norms, grad_norms, logp_u = stein_stats(ebm, x, critic)
            loss = stats.mean() + .001 * (logp_u ** 2).mean()
            l2_penalty = norms.mean() * args.l2
            grad_norm_penalty = grad_norms.mean() * args.grad_l2

            cycle_iter = itr % (args.k_iters + args.e_iters)
            if cycle_iter < args.k_iters:
                if args.t_scaled or args.both_scaled:
                    (-1. * loss / (stats.std() + args.num_const) + l2_penalty).backward()
                elif args.t_squared:
                    (stats.var() - (stats ** 2).mean() + l2_penalty).backward()
                else:
                    (-1. * loss + l2_penalty).backward()
                critic_optimizer.step()
            else:
                if args.both_scaled:
                    (loss / (stats.std() + args.num_const) + grad_norm_penalty).backward()
                elif args.e_squared:
                    ((stats ** 2).mean()).backward()
                else:
                    (loss + grad_norm_penalty).backward()
                optimizer.step()

            optimizer.zero_grad()
            critic_optimizer.zero_grad()

            # sampler training
            z = torch.randn((args.batch_size, args.z_dim)).to(device)

            loss_meter.update(loss.item())
            time_meter.update(time.time() - end)

            if itr % args.log_freq == 0:
                log_message = (
                    'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) +/- {:.6f}, logp_u: {} +/- {}'.format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, stats.std(),
                        logp_u.mean().item(), logp_u.std().item()
                    )
                )
                logger.info(log_message)

            if itr % args.save_freq == 0 or itr == args.niters:
                cpu()
                utils.makedirs(args.save)
                torch.save({
                    'args': args,
                    'ebm_state_dict': ebm.state_dict(),
                    'critic_state_dict': critic.state_dict()
                }, os.path.join(args.save, 'checkpt.pth'))
                gpu()

            critic.eval()
            ebm.eval()
            if itr % args.viz_freq == 0 and itr > 0:
                npts = 100
                p_samples = x.view(x.size(0), *args.data_shape)
                pp = "{}/x_p_{}_{}.png".format(os.path.join(args.save, "figs"), epoch, itr)
                utils.makedirs(os.path.dirname(pp))
                if args.logit:
                    plot(pp, torch.sigmoid(p_samples.cpu()))
                else:
                    plot(pp, p_samples.cpu())

                if args.rbm:
                    q_samples = ebm.sample(args.batch_size)
                    q_samples = q_samples.view(q_samples.size(0), 1, 28, 28)
                    pq = "{}/x_q_{}_{}.png".format(os.path.join(args.save, "figs"), epoch, itr)
                    if args.logit:
                        plot(pq, torch.sigmoid(q_samples.cpu()))
                    else:
                        plot(pq, q_samples.cpu())
                else:
                    for e in [.01, .1, .22]:
                        for l in [.1, 1., 10.]:
                            for n in [30, 100, 300]:
                                static_init = init_fn().to(device)
                                q_samples = ebm.sample(static_init, l, e, n)
                                q_samples = q_samples.view(q_samples.size(0), 1, 28, 28)
                                pq = "{}/x_q_{}_{}___{}_{}_{}_rand.png".format(os.path.join(args.save, "figs"),
                                                                          epoch, itr, l, e, n)
                                if args.logit:
                                    plot(pq, torch.sigmoid(q_samples.cpu()))
                                else:
                                    plot(pq, q_samples.cpu())

                x_c = critic(x).view(x.size(0), 1, 28, 28)
                pc = "{}/x_c_{}_{}.png".format(os.path.join(args.save, "figs"), epoch, itr)
                plot(pc, x_c.cpu())
            critic.train()
            ebm.train()

            end = time.time()
            itr += 1

    logger.info('Training has finished.')