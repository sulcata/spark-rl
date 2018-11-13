from abc import abstractmethod
import math

import torch
from torch import nn
from torch.nn import functional as F


def huber_loss(error, delta=1., reduction='elementwise_mean'):
    abs_error = error.abs()
    linear_loss = delta * (abs_error - delta / 2)
    squared_loss = (error ** 2) / 2
    loss = torch.where(abs_error < delta, squared_loss, linear_loss)
    if reduction == 'elementwise_mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    raise ValueError(f'{reduction!r} is not a valid value for reduction')


def quantile_huber_loss(error, quantile, delta=1., reduction='elementwise_mean'):
    return torch.abs(quantile - (error < 0).float()) * huber_loss(error, delta, reduction)


class Sequential(nn.Sequential):
    def forward(self, *args):
        for module in self:
            args = (module(*args),)
        return args[0]

class ResidualBlock(nn.Sequential):
    def forward(self, input):
        return super().forward(input) + input


class NAC(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_hat = nn.Parameter(torch.empty(out_features, in_features))
        self.M_hat = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.in_features)
        self.W_hat.data.uniform_(-stdv, stdv)
        self.M_hat.data.uniform_(-stdv, stdv)

    def forward(self, input):
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return F.linear(input, W)

    def extra_repr(self):
        return f'in_features={self.in_features}, ' \
               f'out_features={self.out_features}'


class NALU(nn.Module):
    def __init__(self, in_features, out_features, space='log', epsilon=1e-7):
        super().__init__()
        if space != 'log' and space != 'asinh':
            raise ValueError(f'{space!r} is not a valid value for space')
        self.in_features = in_features
        self.out_features = out_features
        self.space = space
        self.epsilon = epsilon
        self.W_hat = nn.Parameter(torch.empty(out_features, in_features))
        self.M_hat = nn.Parameter(torch.empty(out_features, in_features))
        self.G = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.in_features)
        self.W_hat.data.uniform_(-stdv, stdv)
        self.M_hat.data.uniform_(-stdv, stdv)

    def forward(self, input):
        g = F.linear(input, self.G).sigmoid()
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        a = F.linear(input, W)
        m = None
        if self.space == 'log':
            m = F.linear(input.abs().add(self.epsilon).log(), W).exp()
        else:
            input_asinh = torch.log(input + torch.sqrt(input ** 2 + 1))
            m = F.linear(input_asinh, W).sinh()
        return g * a + (1 - g) * m

    def extra_repr(self):
        repr = f'in_features={self.in_features}, ' \
               f'out_features={self.out_features}, ' \
               f'space={self.space!r}'
        if self.space == 'log':
            repr += f', epsilon={self.epsilon}'
        return  repr


class Dueling(nn.Module):
    def __init__(self, advantage, value, reduction='mean', distributional=False):
        super().__init__()
        if reduction != 'mean' and reduction != 'max':
            raise ValueError(f'{reduction!r} is not a valid value for reduction')
        self.advantage = advantage
        self.value = value
        self.reduction = reduction
        self.distributional = distributional

    def forward(self, input):
        advantage = self.advantage(input)
        value = self.value(input)
        d = -2 if self.distributional else -1
        reducer = torch.max if self.reduction == 'max' else torch.mean
        return value + (advantage - reducer(advantage, dim=d, keepdim=True))

    def extra_repr(self):
        return f'reduction={self.reduction!r}, ' \
               f'distributional={self.distributional}'


class _DistributionalBase(nn.Module):
    def __init__(self, in_features, out_features, atoms, layer):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.atoms = atoms
        self._fc = layer

    def forward(self, input):
        output = self._fc(input)
        dims = tuple(list(output.shape)[:-1]) + (self.out_features, self.atoms)
        return output.view(*dims)

    def extra_repr(self):
        return f'in_features={self.in_features}, ' \
               f'out_features={self.out_features}, ' \
               f'atoms={self.atoms}'


class Distributional(_DistributionalBase):
    def __init__(self, in_features, out_features, atoms=51, **kwargs):
        layer = nn.Linear(in_features, out_features * atoms, **kwargs)
        super().__init__(in_features, out_features, atoms, layer)


class NoisyDistributional(_DistributionalBase):
    def __init__(self, in_features, out_features, atoms=51, **kwargs):
        layer = NoisyLinear(in_features, out_features * atoms, **kwargs)
        super().__init__(in_features, out_features, atoms, layer)


class IQNFeature(nn.Module):
    def __init__(self, feature, embedding, reduction='product'):
        super().__init__()
        if reduction != 'product' and reduction != 'concat' and reduction != 'residual':
            raise ValueError('{reduction!r} is not a valid value for reduction')
        self.feature = feature
        self.embedding = embedding
        self.reduction = reduction

    def forward(self, input, sample):
        feature = self.feature(input)
        embedding = self.embedding(sample)
        if self.reduction == 'residual':
            return feature * (1 + embedding)
        if self.reduction == 'concat':
            return torch.concat((feature, embedding), dim=-1)
        return feature * embedding

    def extra_repr(self):
        return f'reduction={self.reduction!r}'


class CosineEmbedding(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        i = torch.arange(features, dtype=torch.get_default_dtype())
        self._coeff = math.pi * i

    def forward(self, input):
        return torch.cos(input.unsqueeze(-1) * self._coeff)

    def extra_repr(self):
        return f'features={self.features}'


class NoisyModule(nn.Module):
    @abstractmethod
    def sample_noise(self):
        pass


class NoisyLinear(NoisyModule):
    def __init__(self, in_features, out_features, bias=True,
                 factorized_noise=True, sigma_init=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.include_bias = bias
        self.factorized_noise = factorized_noise
        if sigma_init is not None:
            self.sigma_init = sigma_init
        elif factorized_noise:
            self.sigma_init = 0.5
        else:
            self.sigma_init = 0.017

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        if self.include_bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
            self.register_parameter('bias_epsilon', None)

        self.reset_parameters()

    def _weight(self):
        return self.weight_mu + self.weight_sigma * self.weight_epsilon

    def _bias(self):
        return self.bias_mu + self.bias_sigma * self.bias_epsilon \
               if self.include_bias else None

    def reset_parameters(self):
        mu_range = math.sqrt(1 / self.in_features) if self.factorized_noise \
                   else math.sqrt(3 / self.in_features)
        weight_sigma_init = self.sigma_init / math.sqrt(self.out_features) \
                            if self.factorized_noise else self.sigma_init
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(weight_sigma_init)
        self.weight_epsilon.data.zero_()
        if self.include_bias:
            bias_sigma_init = self.sigma_init / math.sqrt(self.out_features) \
                              if self.factorized_noise else self.sigma_init
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(bias_sigma_init)
            self.bias_epsilon.data.zero_()

    def sample_noise(self):
        if self.factorized_noise:
            noise_out = torch.randn(self.out_features)
            noise_in = torch.randn(self.in_features)
            noise = torch.ger(noise_out, noise_in)
            self.weight_epsilon.copy_(noise.sign() * noise.abs().sqrt())
            if self.include_bias:
                self.bias_epsilon.data.normal_()
        else:
            self.weight_epsilon.data.normal_()
            if self.include_bias:
                self.bias_epsilon.data.normal_()

    def forward(self, input):
        weight = self._weight() if self.training else self.weight_mu
        bias = self._bias() if self.training else self.bias_mu
        return F.linear(input, weight, bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, ' \
               f'out_features={self.out_features}, ' \
               f'bias={self.include_bias}, ' \
               f'factorized_noise={self.factorized_noise}, ' \
               f'sigma_init={self.sigma_init}'


class NoisyBilinear(NoisyModule):
    def __init__(self, in1_features, in2_features, out_features, bias=True,
                 factorized_noise=True, sigma_init=None):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.include_bias = bias
        self.factorized_noise = factorized_noise
        if sigma_init is not None:
            self.sigma_init = sigma_init
        elif factorized_noise:
            self.sigma_init = 0.5
        else:
            self.sigma_init = 0.017

        self.weight_mu = nn.Parameter(torch.empty(out_features, in1_features, in2_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in1_features, in2_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in1_features, in2_features))

        if self.include_bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
            self.register_parameter('bias_epsilon', None)

        self.reset_parameters()

    def _weight(self):
        return self.weight_mu + self.weight_sigma * self.weight_epsilon

    def _bias(self):
        return self.bias_mu + self.bias_sigma * self.bias_epsilon \
               if self.include_bias else None

    def reset_parameters(self):
        p = self.in1_features * self.in2_features
        mu_range = math.sqrt(1 / p) if self.factorized_noise else math.sqrt(3 / p)
        weight_sigma_init = self.sigma_init / math.sqrt(p) \
                            if self.factorized_noise else self.sigma_init
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(weight_sigma_init)
        self.weight_epsilon.data.zero_()
        if self.include_bias:
            bias_sigma_init = self.sigma_init / math.sqrt(self.out_features) \
                              if self.factorized_noise else self.sigma_init
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(bias_sigma_init)
            self.bias_epsilon.data.zero_()

    def sample_noise(self):
        if self.factorized_noise:
            noise_out = torch.randn(self.out_features)
            noise_in1 = torch.randn(self.in1_features)
            noise_in2 = torch.randn(self.in2_features)
            noise = noise_out.view(self.out_features, 1, 1) \
                  * noise_in1.view(1, self.in1_features, 1) \
                  * noise_in2.view(1, 1, self.in2_features)
            self.weight_epsilon.copy_(noise.pow(1 / 3))
            if self.include_bias:
                self.bias_epsilon.data.normal_()
        else:
            self.weight_epsilon.data.normal_()
            if self.include_bias:
                self.bias_epsilon.data.normal_()

    def forward(self, input1, input2):
        weight = self._weight() if self.training else self.weight_mu
        bias = self._bias() if self.training else self.bias_mu
        return F.bilinear(input1, input2, weight, bias)

    def extra_repr(self):
        return f'in1_features={self.in1_features}, ' \
               f'in2_features={self.in2_features}, ' \
               f'out_features={self.out_features}, ' \
               f'bias={self.include_bias}, ' \
               f'factorized_noise={self.factorized_noise}, ' \
               f'sigma_init={self.sigma_init}'
