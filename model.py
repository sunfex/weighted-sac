import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -10
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class BaseQNet(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim):
        super(BaseQNet, self).__init__()

        self.linear1 = nn.Linear(state_dim + act_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1

class BatchQNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, N=2):
        super().__init__()
        self.N = N

        self.qnets = nn.ModuleList([])
        for i in range(N):
            self.qnets.append(BaseQNet(obs_dim, act_dim, hidden_dim))

    def forward(self, states, actions, beta=10):
        out = []
        for i in range(self.N):
            out.append( self.qnets[i](states, actions) )
        out = torch.cat(out, dim=1)
        out = out.min(dim=1)[0][:,None]
        #out = ( out * torch.softmax(out*(-beta), dim=1) ).sum(dim=1)[:,None]
        return out

    def loss(self, states, actions, target_q, wei=None):
        ret = 0.0
        for i in range(self.N):
            if wei is None:
                ret = ret + (self.qnets[i](states, actions) - target_q).pow(2).mean()
            else:
                ret = ret + ((self.qnets[i](states, actions) - target_q).pow(2)*wei).sum()/target_q.size(0)
        return ret
    
    def av_losses(self, states, actions, target_q):
        ret = 0.0
        for i in range(self.N):
            ret = ret + (self.qnets[i](states, actions)- target_q).abs()
        return ret/2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def likeli_and_dist(self, ss, aa):
        mean, log_std = self.forward(ss)
        std = log_std.exp()
        diff = aa-mean

        # log prob
        log_prob = (-log_std -(diff/std).pow(2)/2.0 ).sum(dim=1)
        dist = diff.pow(2).sum(dim=1).sqrt()

        return log_prob, dist

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
