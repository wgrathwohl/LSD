import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import torch
import torch.distributions as tdist


# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        obs = batch_size
        batch_size = batch_size * 20
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)
        inds = np.random.choice(list(range(batch_size)), obs)
        X = X[inds]
        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        assert False


class Mixture:
    def __init__(self, comps, pi):
        self.pi = tdist.OneHotCategorical(probs=pi)
        self.comps = comps

    def sample(self, n):
        c = self.pi.sample((n,))
        xs = [comp.sample((n,)).unsqueeze(-1) for comp in self.comps]
        xs = torch.cat(xs, -1)
        x = (c[:, None, :] * xs).sum(-1)
        return x

    def logprob(self, x):
        lpx = [comp.log_prob(x) for comp in self.comps]
        lpx = [lp.view(lp.size(0), -1).sum(1).unsqueeze(-1) for lp in lpx]
        lpx = torch.cat(lpx, -1).clamp(-20, 20)
        logpxc = lpx + torch.log(self.pi.probs[None])
        logpx = logpxc.logsumexp(1)
        return logpx


def multi_dim_2d(data_list, rng=None, batch_size=200):
    """
    Creates a mixture of datasets in 2D slices based on inf_train_gen
    """
    samples = []
    for data in data_list:
        samples.append(inf_train_gen(data, rng, batch_size))
    final_samples = np.concatenate(samples, axis=1)
    return final_samples


def gaussian_grid_2d(size=2, std=.25):
    comps = []
    for i in range(size):
        for j in range(size):
            center = np.array([i, j])
            center = torch.from_numpy(center).float()
            comp = tdist.Normal(center, torch.ones((2,)) * std)
            comps.append(comp)

    pi = torch.ones((size**2,)) / (size**2)
    mog = Mixture(comps, pi)
    return mog


if __name__ == "__main__":
    import visualize_flow

    mog = gaussian_grid_2d(4)
    # import matplotlib.pyplot as plt
    x = mog.sample(1000)
    # plt.scatter(x[:, 0].numpy(), x[:, 1].numpy())
    # plt.savefig("/tmp/samp.jpg")
    # # lp = mog.logprob(x)
    # #visualize_flow.visualize_transform(logdensity=mog.logprob)
    # # print(lp.size())
    # # lp = lp.numpy()
    # # plt.hist(lp)
    # # plt.show()
    # plt.figure(figsize=(9, 3))
    # visualize_flow.visualize_transform(logdensity=mog.logprob, npts=800)
    # fig_filename = "/tmp/fig.jpg"
    # plt.savefig(fig_filename)
    # plt.close()

    # x = torch.randn(13, 2, requires_grad=True)
    # l = mog.logprob(x)
    # print(l.size())
    # g = torch.autograd.grad(l.sum(), x)[0]
    # print(g)
    #
    #datas = ["swissroll", "line", "moons"]
    #names = []
    #for name in datas:
    #    names.append(name)
    #    names.append(name)

    #test_multidim = multi_dim_2d(datas)
    #visualize_flow.visualize_slices(x, ["g", "g2"])
