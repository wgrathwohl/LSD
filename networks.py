import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_transpose_3x3(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=1, bias=True)


def conv3x3(in_planes, out_planes, stride=1):
    if stride < 0:
        return conv_transpose_3x3(in_planes, out_planes, stride=-stride)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, out_nonlin=True):
        super(BasicBlock, self).__init__()
        self.nonlin1 = Swish(planes)#nn.ELU()
        self.nonlin2 = Swish(planes)#nn.ELU()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.out_nonlin = out_nonlin

        self.shortcut_conv = None
        if stride != 1 or in_planes != self.expansion * planes:
            if stride < 0:
                self.shortcut_conv = nn.ConvTranspose2d(in_planes, self.expansion*planes,
                                                        kernel_size=1, stride=-stride,
                                                        output_padding=1, bias=True)
            else:
                self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes,
                                               kernel_size=1, stride=stride, bias=True)


    def forward(self, x):
        out = self.nonlin1(self.conv1(x))
        out = self.conv2(out)
        if self.shortcut_conv is not None:
            out_sc = self.shortcut_conv(x)
            out += out_sc
        else:
            out += x
        if self.out_nonlin:
            out = self.nonlin2(out)
        return out


class MNISTResNet(nn.Module):
    def __init__(self, n_channels=64, quadratic=False):
        super().__init__()
        self.proj = nn.Conv2d(1, n_channels, 3, 1, 1)
        downsample = [
            BasicBlock(n_channels, n_channels, 2),
            BasicBlock(n_channels, n_channels, 2)
        ]
        main = [BasicBlock(n_channels, n_channels, 1) for _ in range(6)]
        all = downsample + main
        self.net = nn.Sequential(*all)
        self.energy_linear = nn.Linear(n_channels, 1)
        self.energy_linear2 = nn.Linear(4 * 4 * n_channels, 1)
        self.energy_linear3 = nn.Linear(4 * 4 * n_channels, 1)
        self.quadratic = quadratic

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        input = self.proj(input)
        out = self.net(input)
        if self.quadratic:
            out = out.view(input.size(0), -1)
            return (self.energy_linear(out) * self.energy_linear2(out) + self.energy_linear3(out**2)).squeeze()
        else:
            out = out.view(out.size(0), out.size(1), -1).mean(-1)
            return self.energy_linear(out).squeeze()


class MNISTConvNet(nn.Module):
    def __init__(self, nc=16, quadratic=False):
        super().__init__()
        self.quadratic = quadratic
        self.net = nn.Sequential(
            nn.Conv2d(1, nc, 3, 1, 1),
            Swish(nc), #nn.ELU(),

            nn.Conv2d(nc, nc * 2, 4, 2, 1),
            Swish(nc * 2), #nn.ELU(),

            nn.Conv2d(nc * 2, nc * 2, 3, 1, 1),
            Swish(nc * 2), #nn.ELU(),

            nn.Conv2d(nc * 2, nc * 4, 4, 2, 1),
            Swish(nc * 4), #nn.ELU(),

            nn.Conv2d(nc * 4, nc * 4, 3, 1, 1),
            Swish(nc * 4), #nn.ELU(),

            nn.Conv2d(nc * 4, nc * 8, 4, 2, 1),
            Swish(nc * 8), #nn.ELU(),

            nn.Conv2d(nc * 8, nc * 8, 3, 1, 0),
            Swish(nc * 8), #nn.ELU()
        )
        self.out = nn.Linear(nc * 8, 1)
        self.out2 = nn.Linear(nc * 8, 1)
        self.out3 = nn.Linear(nc * 8, 1)

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        out = self.net(input)
        out = out.squeeze()
        if self.quadratic:
            return (self.out(out) * self.out2(out) + self.out3(out**2)).squeeze()
        else:
            return self.out(out).squeeze()



class MNISTConvNetL(nn.Module):
    def __init__(self, nc=32, quadratic=False):
        super().__init__()
        self.quadratic = quadratic
        self.net = nn.Sequential(
            nn.Conv2d(1, nc, 3, 1, 1),
            nn.LeakyReLU(.2),

            nn.Conv2d(nc, nc * 2, 4, 2, 1),
            nn.LeakyReLU(.2),

            nn.Conv2d(nc * 2, nc * 4, 4, 2, 1),
            nn.LeakyReLU(.2),

            nn.Conv2d(nc * 4, nc * 8, 4, 2, 1),
            nn.LeakyReLU(.2),

            nn.Conv2d(nc * 8, 1, 3, 1, 0),
        )

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        out = self.net(input)
        out = out.squeeze()
        return out


class MNISTConvNetCritic(nn.Module):
    def __init__(self, nc=16):
        super().__init__()
        nef = nc
        self.net = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(1, nef, 3, stride=1, padding=1),
            nn.GroupNorm(4, nef),
            Swish(nef), #nn.ELU(),
            nn.Conv2d(nef, nef, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef),
            Swish(nef), #nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.Conv2d(nef, nef, 3, stride=1, padding=1),
            nn.GroupNorm(4, nef),
            Swish(nef), #nn.ELU(),
            nn.Conv2d(nef, nef * 2, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef * 2),
            Swish(nef * 2), #nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.Conv2d(nef * 2, nef * 4, 5, stride=1, padding=0),
            nn.GroupNorm(4, nef * 4),
            Swish(nef * 4), #nn.ELU(),
            nn.Conv2d(nef * 4, nef * 4, 3, stride=1, padding=1),
            nn.GroupNorm(4, nef * 4),
            Swish(nef * 4), #nn.ELU(),
            # state size. (nef*4) x 3 x 3
            nn.ConvTranspose2d(nef * 4, nef * 2, 5, stride=1, padding=0),
            nn.GroupNorm(4, nef * 2),
            Swish(nef * 2), #nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.ConvTranspose2d(nef * 2, nef, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef),
            Swish(nef), #nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.ConvTranspose2d(nef, nef, 3, stride=1, padding=1),
            nn.GroupNorm(4, nef),
            Swish(nef), #nn.ELU(),
            nn.ConvTranspose2d(nef, 1, 4, stride=2, padding=1),
        )

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        out = self.net(input)
        return out.view(input.size(0), -1)


class MNISTSmallConvNet(nn.Module):
    def __init__(self, nc=64, quadratic=False):
        super().__init__()
        self.quadratic = quadratic
        n_c = 1
        n_f = nc
        l = .2
        self.net = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            Swish(n_f),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            Swish(2 * n_f),
            nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
            Swish(4 * n_f),
            nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
            Swish(8 * n_f),
            nn.Conv2d(n_f * 8, n_f * 8, 3, 1, 0)
        )
        self.out = nn.Linear(n_f * 8, 1)

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        out = self.net(input)
        out = self.out(out.squeeze()).squeeze()
        return out


def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]


class GradModule(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        fx = self.mod(x)
        sq = keep_grad(fx.sum(), x)
        return sq


class MNISTSmallConvNetCritic(nn.Module):
    def __init__(self, nc=64):
        super().__init__()
        n_c = 1
        n_f = nc
        l = .2
        self.net = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            Swish(n_f),  # nn.LeakyReLU(l),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            Swish(2 * n_f),  # nn.LeakyReLU(l),
            nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
            Swish(4 * n_f),  # nn.LeakyReLU(l),
            nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
            Swish(8 * n_f),  # nn.LeakyReLU(l),
            nn.ConvTranspose2d(n_f * 8, n_f * 4, 3, 2, 0),
            Swish(4 * n_f),  # nn.LeakyReLU(l),
            nn.ConvTranspose2d(n_f * 4, n_f * 2, 4, 2, 1),
            Swish(2 * n_f),  # nn.LeakyReLU(l),
            nn.ConvTranspose2d(n_f * 2, n_f, 4, 2, 1),
            Swish(n_f),  # nn.LeakyReLU(l),
            nn.Conv2d(nc, 1, 3, 1, 1)
        )

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        out = self.net(input)
        return out.view(input.size(0), -1)


class MNISTResNetCritic(nn.Module):
    def __init__(self, n_channels=16):
        super().__init__()
        self.proj = nn.Conv2d(1, n_channels, 3, 1)
        downsample = [
            BasicBlock(n_channels, n_channels, 2),
            BasicBlock(n_channels, n_channels, 2)
        ]
        upsample = [
            BasicBlock(n_channels, n_channels, -2),
            BasicBlock(n_channels, n_channels, -2)
        ]
        self.out_conv = nn.Conv2d(n_channels, 1, 3, 1, 1)
        main = [BasicBlock(n_channels, n_channels, 1) for _ in range(12)]
        all = downsample + main + upsample
        self.net = nn.Sequential(*all)

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        input = self.proj(input)
        out = self.net(input)
        return self.out_conv(out).view(input.size(0), -1)


class AnnealLinear(nn.Linear):
    def __init__(self, in_dim, out_dim):
        super(AnnealLinear, self).__init__(in_dim, out_dim)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Swish(nn.Module):
    def __init__(self, dim=-1):
        super(Swish, self).__init__()
        if dim > 0:
            self.beta = nn.Parameter(torch.ones((dim,)))
        else:
            self.beta = torch.ones((1,))

    def forward(self, x):
        if len(x.size()) == 2:
            return x * torch.sigmoid(self.beta[None, :] * x)
        else:
            return x * torch.sigmoid(self.beta[None, :, None, None] * x)


class CouplingBlock(nn.Module):
    def __init__(self, dim_in, dim_h=64, n_layers=2):
        super(CouplingBlock, self).__init__()
        layers = []
        for i in range(n_layers):
            din = dim_in // 2 if i == 0 else dim_h
            layers.append(nn.Linear(din, dim_h))
            layers.append(Swish(dim_h))
        layers.append(nn.Linear(dim_h, dim_in // 2))
        self.net_l = nn.Sequential(*layers)

        layers = []
        for i in range(n_layers):
            din = dim_in // 2 if i == 0 else dim_h
            layers.append(nn.Linear(din, dim_h))
            layers.append(Swish(dim_h))
        layers.append(nn.Linear(dim_h, dim_in // 2))
        self.net_r = nn.Sequential(*layers)

    def forward(self, x):
        x_l, x_r = x[:, :x.size(1) // 2], x[:, x.size(1) // 2:]
        p_r = self.net_l(x_l)
        y_r = x_r + p_r
        p_l = self.net_r(y_r)
        y_l = x_l + p_l
        y = torch.cat([y_l, y_r], 1)
        return y

    def reverse(self, y):
        y_l, y_r = y[:, :y.size(1) // 2], y[:, y.size(1) // 2:]
        p_l = self.net_r(y_r)
        x_l = y_l - p_l
        p_r = self.net_l(x_l)
        x_r = y_r - p_r
        x = torch.cat([x_l, x_r], 1)
        return x


class InvertibleEncoder(nn.Module):
    def __init__(self, n_dims=2, n_blocks=2, n_layers=2, dim_h=64, padding=0):
        super(InvertibleEncoder, self).__init__()
        self.padding = padding
        layers = [CouplingBlock(n_dims + padding, dim_h, n_layers) for _ in range(n_blocks)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.padding > 0:
            z = torch.zeros((x.size(0), self.padding)).to(x)
            xp = torch.cat([x, z], 1)
        else:
            xp = x
        return self.net(xp)


class MLP(nn.Module):
    def __init__(self, n_dims):
        super(MLP, self).__init__()
        self._built = False
        self.net = nn.Sequential(
            nn.Linear(n_dims, 300),
            nn.LeakyReLU(.2),
            nn.Linear(300, 300),
            nn.LeakyReLU(.2),
            nn.Linear(300, 300),
            nn.LeakyReLU(.2),
            nn.Linear(300, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        return out.squeeze()


class QuadraticForm(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(QuadraticForm, self).__init__()
        self.a = nn.Linear(dim_in, dim_out)
        self.b = nn.Linear(dim_in, dim_out)
        self.d = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.a(x) * self.b(x) + self.d(x**2)



class QuadraticMLP(nn.Module):
    def __init__(self, n_dims, n_out=1, n_hid=300, layer=nn.Linear):
        super(QuadraticMLP, self).__init__()
        self._built = False
        self.net = nn.Sequential(
            nn.Linear(n_dims, n_hid),
            Swish(n_hid),
            nn.Linear(n_hid, n_hid),
            Swish(n_hid),
            QuadraticForm(n_hid, n_out)#n_hid),
            #nn.Linear(n_hid, n_out)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        return out.squeeze()



class SmallMLP(nn.Module):
    def __init__(self, n_dims, n_out=1, n_hid=300, layer=nn.Linear, dropout=False):
        super(SmallMLP, self).__init__()
        self._built = False
        if dropout:
            self.net = nn.Sequential(
                layer(n_dims, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_out)
            )
        else:
            self.net = nn.Sequential(
                layer(n_dims, n_hid),
                Swish(n_hid),
                layer(n_hid, n_hid),
                Swish(n_hid),
                layer(n_hid, n_out)
            )
        self.normalized = False

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        out = out.squeeze()
        if self.normalized:
            return out / (out.norm(dim=1, keepdim=True) + 1e-6)
        else:
            return out

class LargeMLP(nn.Module):
    def __init__(self, n_dims, n_out=1, n_hid=300, layer=nn.Linear, dropout=False):
        super(LargeMLP, self).__init__()
        self._built = False
        if dropout:
            self.net = nn.Sequential(
                layer(n_dims, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_out)
            )
        else:
            self.net = nn.Sequential(
                layer(n_dims, n_hid),
                Swish(n_hid),
                layer(n_hid, n_hid),
                Swish(n_hid),
                layer(n_hid, n_hid),
                Swish(n_hid),
                layer(n_hid, n_hid),
                Swish(n_hid),
                layer(n_hid, n_out)
            )
        self.normalized = False

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        out = out.squeeze()
        if self.normalized:
            return out / (out.norm(dim=1, keepdim=True) + 1e-6)
        else:
            return out


class MixedMLP(nn.Module):
    def __init__(self, n_dims, n_out=1, n_hid=300, layer=nn.Linear):
        super(MixedMLP, self).__init__()
        self._built = False
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            Swish(1)
        )
        self.net = nn.Sequential(
            layer(28 * 28 * 8, n_hid),
            Swish(n_hid),
            layer(n_hid, n_hid),
            Swish(n_hid),
            layer(n_hid, n_out)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.net(x)
        out = out.squeeze()
        return out


class BigMLP(nn.Module):
    def __init__(self, n_dims, n_out=1, n_hid=300, layer=nn.Linear, dropout=False):
        super(BigMLP, self).__init__()
        self._built = False
        if dropout:
            self.net = nn.Sequential(
                layer(n_dims, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_out)
            )
        else:
            self.net = nn.Sequential(
                layer(n_dims, n_hid),
                Swish(n_hid),
                layer(n_hid, n_hid),
                Swish(n_hid),
                layer(n_hid, n_hid),
                Swish(n_hid),
                layer(n_hid, n_hid),
                Swish(n_hid),
                layer(n_hid, n_out)
            )
        self.normalized = False

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        out = out.squeeze()
        if self.normalized:
            return out / (out.norm(dim=1, keepdim=True) + 1e-6)
        else:
            return out


class SmallConv(nn.Module):
    def __init__(self, n_dims, n_channels):
        super(SmallConv, self).__init__()
        self._built = False
        self.net = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(.2),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=3),
            nn.LeakyReLU(.2),
            nn.Linear(n_dims * n_channels, 1)
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        out = self.net(x)
        return out.squeeze()


if __name__ == "__main__":
    L = CouplingBlock(10)

    x = torch.randn((13, 10))

    y = L(x)

    x_re = L.reverse(y)

    print((x - x_re).norm())
