import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, DeterministicPolicy, BatchQNet


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu") 
        self.critic = BatchQNet(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = BatchQNet(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if args.policy == "Gaussian":
            self.alpha = args.alpha
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            self.alpha = 0
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if eval == False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
        return action.cpu().numpy()[0]

    def update_parameters(self, memory, args, traj_len, episode_step):
        # Sample a batch from memory
        if args.mode == 'SAC':
            ss, aa, rewards, ns, msk = memory.sample(args.batch_size)
        elif args.mode == 'ERE':
            ss, aa, rewards, ns, msk = memory.ERE_sample(args.batch_size, episode_step)
        elif args.mode == 'ERE2':
            ss, aa, rewards, ns, msk = memory.ERE2_sample(args.batch_size)
        elif args.mode == 'EREo':
            ss, aa, rewards, ns, msk = memory.ERE_sample(args.batch_size, episode_step, K=traj_len)
        elif args.mode == 'HAR':
            ss, aa, rewards, ns, msk = memory.HAR_sample(args.batch_size)
        
        if 'numpy' in str(type(ss)) or ss.device.type=='cpu':
            ss = torch.FloatTensor(ss).to(self.device)
            ns = torch.FloatTensor(ns).to(self.device)
            aa = torch.FloatTensor(aa).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
            msk = torch.FloatTensor(msk).to(self.device).unsqueeze(1)

        # likelihoods
        with torch.no_grad():
            log_likeli, dist = self.policy.likeli_and_dist(ss, aa)
        log_likeli, dist = log_likeli.mean(), dist.mean()
        

        # soft next q value
        with torch.no_grad():
            na, n_log_pi, _ = self.policy.sample(ns)
            min_qf_next_target = self.critic_target(ns, na) - self.alpha * n_log_pi
            next_q_value = rewards + msk * self.gamma * (min_qf_next_target)
        
        # Q-function loss
        qf_loss = self.critic.loss(ss, aa, next_q_value)
        
        # deterministic policy gradient
        pi, log_pi, _ = self.policy.sample(ss)
        min_qf_pi = self.critic(ss, pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # update        
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        soft_update(self.critic_target, self.critic, self.tau)
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        #entropy = -log_pi.mean().item()
        return log_likeli, dist

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

