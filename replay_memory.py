import numpy as np
import random
import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        # traj
        self.tlen = 0
        self.tcount = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            #self.buffer.append(None)
            self.buffer.append( (state, action, reward, next_state, done) )
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        '''
        uniform sampling
        '''
        batch = random.sample(self.buffer, batch_size)
        #ind = np.random.choice(len(self.buffer), batch_size, replace=False)
        #batch = [self.buffer[i] for i in ind]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
     
    def swor_exp(self, w, K):
        '''
        fast algorithm for sampling without replacement
        https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/
        '''

        E = -np.log(np.random.uniform(0,1, size=len(w)))
        E /= w
        return np.argpartition(E,K)[:K]

    def ERE_sample(self, batch_size, episode_step, eta=0.996, cmin=5000, K=None):
        if len(self.buffer)<batch_size:
            batch = self.buffer
        else:
            if K is None:
                '''
                set by args.mode == ERE 

                If K is not given, need to estimate trajectory length.
                This makes it possible to update whenever getting a new (s,a,ns,r,msk)
                
                self.tlen is estimated by self.update_trajlen(tlen, talp = 0.9)
                '''
                tlen = self.tlen if self.tcount>0 else 100.0
            else:
                '''
                set by args.mode == EREo
                
                If K is given, there is no need to estimate trajectory length.
                This is the original version of ERE at https://arxiv.org/abs/1906.04009
                '''
                tlen = K
            ck = max(len(self.buffer)*(eta**(episode_step*1000/tlen)), cmin)
            ck = min(len(self.buffer), int(ck)) # upper capped by buffer length
            ck = max(batch_size, ck) # lower capped by batch size

            ind = np.random.choice(ck, batch_size, replace=False) 
            ind = len(self.buffer)-1 - ind
            ind = np.mod( ind + self.position, len(self.buffer) )
            batch = [self.buffer[i] for i in ind]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def ERE2_sample(self, batch_size, eta=0.996, cmin=5000):
        '''
        The approximated ERE at Proposition 1 of the paper.
        This shows ERE is equivalent to a non-uniform weighting.
        '''
        if len(self.buffer) <= max(batch_size, cmin):
            if len(self.buffer) <= batch_size:
                batch = self.buffer
            else:
                ind = np.random.choice(len(self.buffer), batch_size, replace=False)
                batch = [self.buffer[i] for i in ind]
        else:
            eta = eta**1000
            N = len(self.buffer)
            ii = np.arange(1,N+1) 
            base = 1/np.maximum(max(cmin, N*eta), ii) - 1/N
            lump = np.log(cmin/N/eta)/cmin
            wei = base + max(lump,0)*(ii<=cmin)
            ind = self.swor_exp(wei, batch_size)
            #ind = np.random.choice(len(self.buffer), batch_size, replace=True, p=(wei/sum(wei)))
            ind = N-1 - ind
            batch = [self.buffer[i] for i in ind]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def HAR_sample(self, batch_size):
        '''
        1/age weighting strategy.
        The name derives from the concept of harmonic series.
        '''
        if len(self.buffer)<batch_size:
            batch = self.buffer
        else:
            #tlen = self.tlen if self.tcount>0 else 100.0
            N = len(self.buffer)
            wei = 1./np.arange(1, N+1) #- 1/N
            ind = N-1 - self.swor_exp(wei, batch_size)
            #ind = np.mod( ind + self.position, len(self.buffer) )
            batch = [self.buffer[i] for i in ind]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
     
    def to_torch(self, device, ss, aa, rewards, ns, msk):
        ss = torch.FloatTensor(ss).to(device)
        ns = torch.FloatTensor(ns).to(device)
        aa = torch.FloatTensor(aa).to(device)
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
        msk = torch.FloatTensor(msk).to(device).unsqueeze(1)
        return ss, aa, rewards, ns, msk    
    
    def update_trajlen(self, tlen, talp = 0.9):
        '''
        Estimate trajectory length using exponential averaging over historical 
        lengths with parameter talp.
        '''
        self.tlen = tlen*(1-talp) + self.tlen*(1-talp**self.tcount)*talp
        self.tcount += 1
        self.tlen = self.tlen/(1-talp**self.tcount)
    
