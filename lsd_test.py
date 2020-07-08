import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.optim as optim
import numpy as np
import networks
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import torch.nn.utils.spectral_norm as spectral_norm
from tqdm import tqdm


def try_make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

def randb(size):
    dist = distributions.Bernoulli(probs=(.5 * torch.ones(*size)))
    return dist.sample().float()


class GaussianBernoulliRBM(nn.Module):
    def __init__(self, B, b, c, burn_in=2000):
        super(GaussianBernoulliRBM, self).__init__()
        self.B = nn.Parameter(B)
        self.b = nn.Parameter(b)
        self.c = nn.Parameter(c)
        # self.B = B
        # self.b = b
        # self.c = c
        self.dim_x = B.size(0)
        self.dim_h = B.size(1)
        self.burn_in = burn_in

    def score_function(self, x):  # dlogp(x)/dx
        return .5 * torch.tanh(.5 * x @ self.B + self.c) @ self.B.t() + self.b - x

    def forward(self, x):  # logp(x)
        B = self.B
        b = self.b
        c = self.c
        xBc = (0.5 * x @ B) + c
        unden =  (x * b).sum(1) - .5 * (x ** 2).sum(1)# + (xBc.exp() + (-xBc).exp()).log().sum(1)
        unden2 = (x * b).sum(1) - .5 * (x ** 2).sum(1) + torch.tanh(xBc/2.).sum(1)#(xBc.exp() + (-xBc).exp()).log().sum(1)
        print((unden - unden2).mean())
        assert len(unden) == x.shape[0]
        return unden

    def sample(self, n):
        x = torch.randn((n, self.dim_x)).to(self.B)
        h = (randb((n, self.dim_h)) * 2. - 1.).to(self.B)
        for t in tqdm(range(self.burn_in)):
            x, h = self._blocked_gibbs_next(x, h)
        x, h = self._blocked_gibbs_next(x, h)
        return x

    def _blocked_gibbs_next(self, x, h):
        """
        Sample from the mutual conditional distributions.
        """
        B = self.B
        b = self.b
        # Draw h.
        XB2C = (x @ self.B) + 2.0 * self.c
        # Ph: n x dh matrix
        Ph = torch.sigmoid(XB2C)
        # h: n x dh
        h = (torch.rand_like(h) <= Ph).float() * 2. - 1.
        assert (h.abs() - 1 <= 1e-6).all().item()
        # Draw X.
        # mean: n x dx
        mean = h @ B.t() / 2. + b
        x = torch.randn_like(mean) + mean
        return x, h


class Gaussian(nn.Module):
    def __init__(self, mu, std):
        super(Gaussian, self).__init__()
        self.dist = distributions.Normal(mu, std)

    def sample(self, n):
        return self.dist.sample_n(n)

    def forward(self, x):
        return self.dist.log_prob(x).view(x.size(0), -1).sum(1)


class Laplace(nn.Module):
    def __init__(self, mu, std):
        super(Laplace, self).__init__()
        self.dist = distributions.Laplace(mu, std)

    def sample(self, n):
        return self.dist.sample_n(n)

    def forward(self, x):
        return self.dist.log_prob(x).view(x.size(0), -1).sum(1)


def sample_batch(data, batch_size):
    all_inds = list(range(data.size(0)))
    chosen_inds = np.random.choice(all_inds, batch_size, replace=False)
    chosen_inds = torch.from_numpy(chosen_inds)
    return data[chosen_inds]



def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input,
                               grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]


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


class SpectralLinear(nn.Module):
    def __init__(self, n_in, n_out, max_sigma=1.):
        super(SpectralLinear, self).__init__()
        self.linear = spectral_norm(nn.Linear(n_in, n_out))
        self.scale = nn.Parameter(torch.zeros((1,)))
        self.max_sigma = max_sigma

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.scale) * self.max_sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['gaussian-laplace', 'laplace-gaussian',
                                           'gaussian-pert', 'rbm-pert', 'rbm-pert1'], type=str)
    parser.add_argument('--dim_x', type=int, default=50)
    parser.add_argument('--dim_h', type=int, default=40)
    parser.add_argument('--sigma_pert', type=float, default=.02)
    parser.add_argument('--maximize_power', action="store_true")
    parser.add_argument('--maximize_adj_mean', action="store_true")
    parser.add_argument('--val_power', action="store_true")
    parser.add_argument('--val_adj_mean', action="store_true")
    parser.add_argument('--dropout', action="store_true")
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--save', type=str, default='/tmp/test_ksd')

    parser.add_argument('--test_type', type=str, default='mine')



    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0.)

    parser.add_argument('--num_const', type=float, default=1e-6)

    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)



    parser.add_argument('--seed', type=int, default=100001)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_iters', type=int, default=100001)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--test_burn_in', type=int, default=0)
    parser.add_argument('--mode', type=str, default="fs")
    parser.add_argument('--viz_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10000)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--base_dist', action="store_true")
    parser.add_argument('--t_iters', type=int, default=5)
    parser.add_argument('--k_dim', type=int, default=1)
    parser.add_argument('--sn', type=float, default=-1.)

    parser.add_argument('--exact_trace', action="store_true")
    parser.add_argument('--quadratic', action="store_true")
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--both_scaled', action="store_true")
    args = parser.parse_args()
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.test == "gaussian-laplace":
        mu = torch.zeros((args.dim_x,))
        std = torch.ones((args.dim_x,))
        p_dist = Gaussian(mu, std)
        q_dist = Laplace(mu, std)
    elif args.test == "laplace-gaussian":
        mu = torch.zeros((args.dim_x,))
        std = torch.ones((args.dim_x,))
        q_dist = Gaussian(mu, std)
        p_dist = Laplace(mu, std / (2 ** .5))
    elif args.test == "gaussian-pert":
        mu = torch.zeros((args.dim_x,))
        std = torch.ones((args.dim_x,))
        p_dist = Gaussian(mu, std)
        q_dist = Gaussian(mu + torch.randn_like(mu) * args.sigma_pert, std)
    elif args.test == "rbm-pert1":
        B = randb((args.dim_x, args.dim_h)) * 2. - 1.
        c = torch.randn((1, args.dim_h))
        b = torch.randn((1, args.dim_x))

        p_dist = GaussianBernoulliRBM(B, b, c)
        B2 = B.clone()
        B2[0, 0] += torch.randn_like(B2[0, 0]) * args.sigma_pert
        q_dist = GaussianBernoulliRBM(B2, b, c)
    else:  # args.test == "rbm-pert"
        B = randb((args.dim_x, args.dim_h)) * 2. - 1.
        c = torch.randn((1, args.dim_h))
        b = torch.randn((1, args.dim_x))

        p_dist = GaussianBernoulliRBM(B, b, c)
        q_dist = GaussianBernoulliRBM(B + torch.randn_like(B) * args.sigma_pert, b, c)

    # run mah shiiiiit
    if args.test_type == "mine":
        import numpy as np
        data = p_dist.sample(args.n_train + args.n_val + args.n_test).detach()
        data_train = data[:args.n_train]
        data_rest = data[args.n_train:]
        data_val = data_rest[:args.n_val].requires_grad_()
        data_test = data_rest[args.n_val:].requires_grad_()
        assert data_test.size(0) == args.n_test

        critic = networks.SmallMLP(args.dim_x, n_out=args.dim_x, n_hid=300, dropout=args.dropout)
        optimizer = optim.Adam(critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        def stein_discrepency(x, exact=False):
            if "rbm" in args.test:
                sq = q_dist.score_function(x)
            else:
                logq_u = q_dist(x)
                sq = keep_grad(logq_u.sum(), x)
            fx = critic(x)
            if args.dim_x == 1:
                fx = fx[:, None]
            sq_fx = (sq * fx).sum(-1)

            if exact:
                tr_dfdx = exact_jacobian_trace(fx, x)
            else:
                tr_dfdx = approx_jacobian_trace(fx, x)

            norms = (fx * fx).sum(1)
            stats = (sq_fx + tr_dfdx)
            return stats, norms

        # training phase
        best_val = -np.inf
        validation_metrics = []
        test_statistics = []
        critic.train()
        for itr in range(args.n_iters):
            optimizer.zero_grad()
            x = sample_batch(data_train, args.batch_size)
            x = x.to(device)
            x.requires_grad_()

            stats, norms = stein_discrepency(x)
            mean, std = stats.mean(), stats.std()
            l2_penalty = norms.mean() * args.l2


            if args.maximize_power:
                loss = -1. * mean / (std + args.num_const) + l2_penalty
            elif args.maximize_adj_mean:
                loss = -1. * mean + std + l2_penalty
            else:
                loss = -1. * mean + l2_penalty

            loss.backward()
            optimizer.step()

            if itr % args.log_freq == 0:
                print("Iter {}, Loss = {}, Mean = {}, STD = {}, L2 {}".format(itr,
                                                                           loss.item(), mean.item(), std.item(),
                                                                           l2_penalty.item()))

            if itr % args.val_freq == 0:
                critic.eval()
                val_stats, _ = stein_discrepency(data_val, exact=True)
                test_stats, _ = stein_discrepency(data_test, exact=True)
                print("Val: {} +/- {}".format(val_stats.mean().item(), val_stats.std().item()))
                print("Test: {} +/- {}".format(test_stats.mean().item(), test_stats.std().item()))

                if args.val_power:
                    validation_metric = val_stats.mean() / (val_stats.std() + args.num_const)
                elif args.val_adj_mean:
                    validation_metric = val_stats.mean() - val_stats.std()
                else:
                    validation_metric = val_stats.mean()

                test_statistic = test_stats.mean() / (test_stats.std() + args.num_const)



                if validation_metric > best_val:
                    print("Iter {}, Validation Metric = {} > {}, Test Statistic = {}, Current Best!".format(itr,
                                                                                                  validation_metric.item(),
                                                                                                  best_val,
                                                                                                  test_statistic.item()))
                    best_val = validation_metric.item()
                else:
                    print("Iter {}, Validation Metric = {}, Test Statistic = {}, Not best {}".format(itr,
                                                                                                validation_metric.item(),
                                                                                                test_statistic.item(),
                                                                                                best_val))
                validation_metrics.append(validation_metric.item())
                test_statistics.append(test_statistic)
                critic.train()
        best_ind = np.argmax(validation_metrics)
        best_test = test_statistics[best_ind]

        print("Best val is {}, best test is {}".format(best_val, best_test))
        test_stat = best_test * args.n_test ** .5
        threshold = distributions.Normal(0, 1).icdf(torch.ones((1,)) * (1. - args.alpha)).item()
        try_make_dirs(os.path.dirname(args.save))
        with open(args.save, 'w') as f:
            f.write(str(test_stat) + '\n')
            if test_stat > threshold:
                print("{} > {}, rejct Null".format(test_stat, threshold))
                f.write("reject")
            else:
                print("{} <= {}, accept Null".format(test_stat, threshold))
                f.write("accept")

    # baselines
    else:
        import autograd.numpy as np
        #import kgof.goftest as gof
        import mygoftest as gof
        import kgof.util as util
        import kgof.kernel as kernel
        import kgof.density as density
        import kgof.data as kdata

        class GaussBernRBM(density.UnnormalizedDensity):
            """
            Gaussian-Bernoulli Restricted Boltzmann Machine.
            The joint density takes the form
                p(x, h) = Z^{-1} exp(0.5*x^T B h + b^T x + c^T h - 0.5||x||^2)
            where h is a vector of {-1, 1}.
            """

            def __init__(self, B, b, c):
                """
                B: a dx x dh matrix
                b: a numpy array of length dx
                c: a numpy array of length dh
                """
                dh = len(c)
                dx = len(b)
                assert B.shape[0] == dx
                assert B.shape[1] == dh
                assert dx > 0
                assert dh > 0
                self.B = B
                self.b = b
                self.c = c

            def log_den(self, X):
                B = self.B
                b = self.b
                c = self.c

                XBC = 0.5 * np.dot(X, B) + c
                unden = np.dot(X, b) - 0.5 * np.sum(X ** 2, 1) + np.sum(np.log(np.exp(XBC)
                                                                               + np.exp(-XBC)), 1)
                assert len(unden) == X.shape[0]
                return unden

            def grad_log(self, X):
                #    """
                #    Evaluate the gradients (with respect to the input) of the log density at
                #    each of the n points in X. This is the score function.

                #    X: n x d numpy array.
                """
                Evaluate the gradients (with respect to the input) of the log density at
                each of the n points in X. This is the score function.

                X: n x d numpy array.

                Return an n x d numpy array of gradients.
                """
                XB = np.dot(X, self.B)
                Y = 0.5 * XB + self.c
                # E2y = np.exp(2*Y)
                # n x dh
                # Phi = old_div((E2y-1.0),(E2y+1))
                Phi = np.tanh(Y)
                # n x dx
                T = np.dot(Phi, 0.5 * self.B.T)
                S = self.b - X + T
                return S

            def get_datasource(self, burnin=2000):
                return data.DSGaussBernRBM(self.B, self.b, self.c, burnin=burnin)

            def dim(self):
                return len(self.b)

        def job_lin_kstein_med(p, data_source, tr, te, r):
            """
            Linear-time version of the kernel Stein discrepancy test of Liu et al.,
            2016 and Chwialkowski et al., 2016. Use full sample.
            """
            # full data
            data = tr + te
            X = data.data()
            with util.ContextTimer() as t:
                # median heuristic
                med = util.meddistance(X, subsample=1000)
                k = kernel.KGauss(med ** 2)

                lin_kstein = gof.LinearKernelSteinTest(p, k, alpha=args.alpha, seed=r)
                lin_kstein_result = lin_kstein.perform_test(data)
            return {'test_result': lin_kstein_result, 'time_secs': t.secs}

        def job_mmd_opt(p, data_source, tr, te, r, model_sample):
            # full data
            data = tr + te
            X = data.data()
            with util.ContextTimer() as t:
                mmd = gof.QuadMMDGofOpt(p, alpha=args.alpha, seed=r)
                mmd_result = mmd.perform_test(data, model_sample)
            return {'test_result': mmd_result, 'time_secs': t.secs}


        def job_kstein_med(p, data_source, tr, te, r):
            """
            Kernel Stein discrepancy test of Liu et al., 2016 and Chwialkowski et al.,
            2016. Use full sample. Use Gaussian kernel.
            """
            # full data
            data = tr + te
            X = data.data()
            with util.ContextTimer() as t:
                # median heuristic
                med = util.meddistance(X, subsample=1000)
                k = kernel.KGauss(med ** 2)

                kstein = gof.KernelSteinTest(p, k, alpha=args.alpha, n_simulate=1000, seed=r)
                kstein_result = kstein.perform_test(data)
            return {'test_result': kstein_result, 'time_secs': t.secs}

        def job_fssdJ1q_opt(p, data_source, tr, te, r, J=1, null_sim=None):
            """
            FSSD with optimization on tr. Test on te. Use a Gaussian kernel.
            """
            if null_sim is None:
                null_sim = gof.FSSDH0SimCovObs(n_simulate=2000, seed=r)

            Xtr = tr.data()
            with util.ContextTimer() as t:
                # Use grid search to initialize the gwidth
                n_gwidth_cand = 5
                gwidth_factors = 2.0 ** np.linspace(-3, 3, n_gwidth_cand)
                med2 = util.meddistance(Xtr, 1000) ** 2
                print(med2)
                k = kernel.KGauss(med2 * 2)
                # fit a Gaussian to the data and draw to initialize V0
                V0 = util.fit_gaussian_draw(Xtr, J, seed=r + 1, reg=1e-6)
                list_gwidth = np.hstack(((med2) * gwidth_factors))
                besti, objs = gof.GaussFSSD.grid_search_gwidth(p, tr, V0, list_gwidth)
                gwidth = list_gwidth[besti]
                assert util.is_real_num(gwidth), 'gwidth not real. Was %s' % str(gwidth)
                assert gwidth > 0, 'gwidth not positive. Was %.3g' % gwidth
                print('After grid search, gwidth=%.3g' % gwidth)

                ops = {
                    'reg': 1e-2,
                    'max_iter': 40,
                    'tol_fun': 1e-4,
                    'disp': True,
                    'locs_bounds_frac': 10.0,
                    'gwidth_lb': 1e-1,
                    'gwidth_ub': 1e4,
                }

                V_opt, gwidth_opt, info = gof.GaussFSSD.optimize_locs_widths(p, tr,
                                                                             gwidth, V0, **ops)
                # Use the optimized parameters to construct a test
                k_opt = kernel.KGauss(gwidth_opt)
                fssd_opt = gof.FSSD(p, k_opt, V_opt, null_sim=null_sim, alpha=args.alpha)
                fssd_opt_result = fssd_opt.perform_test(te)
            return {'test_result': fssd_opt_result, 'time_secs': t.secs,
                    'goftest': fssd_opt, 'opt_info': info,
                    }

        def job_fssdJ5q_opt(p, data_source, tr, te, r):
            return job_fssdJ1q_opt(p, data_source, tr, te, r, J=5)


        if "rbm" in args.test:
            if args.test_type == "mmd":
                q = kdata.DSGaussBernRBM(np.array(q_dist.B.detach().numpy()),
                                         np.array(q_dist.b.detach().numpy()[0]),
                                         np.array(q_dist.c.detach().numpy()[0]))
            else:
                q = GaussBernRBM(np.array(q_dist.B.detach().numpy()),
                                         np.array(q_dist.b.detach().numpy()[0]),
                                         np.array(q_dist.c.detach().numpy()[0]))
            p = kdata.DSGaussBernRBM(np.array(p_dist.B.detach().numpy()),
                                     np.array(p_dist.b.detach().numpy()[0]),
                                     np.array(p_dist.c.detach().numpy()[0]))
        elif args.test == "laplace-gaussian":
            mu = np.zeros((args.dim_x,))
            std = np.eye(args.dim_x)
            q = density.Normal(mu, std)
            p = kdata.DSLaplace(args.dim_x, scale=1/(2. ** .5))
        elif args.test == "gaussian-pert":
            mu = np.zeros((args.dim_x,))
            std = np.eye(args.dim_x)
            q = density.Normal(mu, std)
            p = kdata.DSNormal(mu, std)

        data_train = p.sample(args.n_train, args.seed)
        data_test = p.sample(args.n_test, args.seed + 1)


        if args.test_type == "fssd":
            result = job_fssdJ5q_opt(q, p, data_train, data_test, r=args.seed)
        elif args.test_type == "ksd":
            result = job_kstein_med(q, p, data_train, data_test, r=args.seed)
        elif args.test_type == "lksd":
            result = job_lin_kstein_med(q, p, data_train, data_test, r=args.seed)
        elif args.test_type == "mmd":
            model_sample = q.sample(args.n_train + args.n_test, args.seed + 2)
            result = job_mmd_opt(q, p, data_train, data_test, args.seed, model_sample)
        print(result['test_result'])
        reject = result['test_result']['h0_rejected']
        try_make_dirs(os.path.dirname(args.save))
        with open(args.save, 'w') as f:
            if reject:
                print("reject")
                f.write("reject")
            else:
                print("accept")
                f.write("accept")


if __name__ == "__main__":
    main()