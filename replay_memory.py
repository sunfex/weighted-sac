import numpy as np
import random
import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
        # torch buffer
        self.tc_buffer = []
        self.tc_size = 0

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
    
    def PER_sample(self, args, device, policy, target_q, q, b1=0.6, b2=0.6):
        '''
        prioritized experience replay
        '''
        
        # copy tc array
        if self.tc_size < len(self.buffer):
            if self.tc_size==0:
                ss, aa, rewards, ns, msk = map(np.stack, zip(*self.buffer))
                ss, aa, rewards, ns, msk = self.to_torch(device, ss, aa, rewards, ns, msk)
                self.tc_buffer = torch.empty((self.capacity,ss.size(1)*2+aa.size(1)+2),
                                             dtype=torch.float32, device=device )
                self.tc_buffer[:len(self.buffer),:] = torch.cat([ss,aa,rewards,ns,msk], dim=1)
                self.tc_size = len(self.buffer)
            else:
                num = len(self.buffer)-self.tc_size
                ss, aa, rewards, ns, msk = map(np.stack, zip(*(self.buffer[-num:])))
                ss, aa, rewards, ns, msk = self.to_torch(device, ss, aa, rewards, ns, msk)
                self.tc_buffer[self.tc_size:self.tc_size+num,:] = torch.cat([ss,aa,rewards,ns,msk], dim=1)
                self.tc_size += num
                
        batch_size = args.batch_size
        ind = np.random.choice(self.tc_size, min(self.tc_size, batch_size*100), 
                               replace=False)
        err = []
        chunk = 4
        chunk_size = int(len(ind)/chunk + 0.99)
        for i in range(chunk):
            subind = ind[i*chunk_size:min(len(ind), (i+1)*chunk_size)]
            ss, aa, rewards, ns, msk = self.split_tc(self.tc_buffer[subind])
            with torch.no_grad():
                na, n_log_pi, _ = policy.sample(ns)
                min_qf_next_target = target_q(ns, na) - args.alpha * n_log_pi
                next_q_value = rewards + msk * args.gamma * (min_qf_next_target)
                err.append( q.av_losses(ss, aa, next_q_value) + 1e-5 )
        err = torch.cat(err, dim=0)
        e_wei = err.pow(b1)
        e_wei = e_wei / e_wei.sum()

        subind = np.random.choice(len(ind), batch_size, replace=False, p=e_wei.view(-1).cpu().numpy())
        wei = (1./(len(ind)*e_wei[subind])).pow(b2)
        
        ss, aa, rewards, ns, msk = self.split_tc(self.tc_buffer[ind[subind]])
        return wei, ss, aa, rewards, ns, msk
     
    def to_torch(self, device, ss, aa, rewards, ns, msk):
        ss = torch.FloatTensor(ss).to(device)
        ns = torch.FloatTensor(ns).to(device)
        aa = torch.FloatTensor(aa).to(device)
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
        msk = torch.FloatTensor(msk).to(device).unsqueeze(1)
        return ss, aa, rewards, ns, msk    
    
    def split_tc(self, arr):
        s0, a0, r0, ns0, m0 = self.buffer[0]
        dims = np.array([0, len(s0),len(a0),1,len(ns0),1]).cumsum()
        out = []
        for i in range(5):
            out.append(arr[:, dims[i]:dims[i+1]])
        return out


    def update_trajlen(self, tlen, talp = 0.9):
        '''
        Estimate trajectory length using exponential averaging over historical 
        lengths with parameter talp.
        '''
        self.tlen = tlen*(1-talp) + self.tlen*(1-talp**self.tcount)*talp
        self.tcount += 1
        self.tlen = self.tlen/(1-talp**self.tcount)
    
