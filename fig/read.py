import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

read_dir = '../result/'
algs = ['SAC','EREo','ERE2','HAR']
#envs = ['HalfCheetah','Ant','Humanoid']
envs = ['HalfCheetah']
name_map = {'SAC':'Uniform weight', 'EREo':'ERE', 'EREe':'ERE', 'ERE2':'ERE_apx', 'HAR':'1/age weight'}

duration = 5000
totalsteps = 1000000

indices = duration*np.arange(1,totalsteps//duration+1)
for idx, env in enumerate(envs):
    plt.figure(idx)
    ms = [] ## average of an algorithm
    for alg in algs:
        files = [f for f in os.listdir(read_dir) if re.search(alg+'_'+env+'-v2', f)]
        means = np.zeros((len(indices),len(files)))
        cum_var = np.zeros(len(indices))
        
        for idx, f in enumerate(files):
            d = np.genfromtxt(read_dir + f, delimiter=',')
            if d[0,0]!=duration:
                # x_indices do not align with the designated one
                # avoid interpolation error
                d[-1,0] = totalsteps
                if d[0,0]>duration: d[0,0] = duration

                fmean = interp1d(d[:,0], d[:,1])
                fstd = interp1d(d[:,0], d[:,2])
                d = np.concatenate([fmean(indices)[:,None], fstd(indices)[:,None]], axis=1)
            else:
                d = d[:, 1:]
            means[:,idx] = d[:,0]
            cum_var += d[:,1]**2
        
        if len(files)>1:
            std = np.sqrt( cum_var/len(files) + means.var(axis=1) )
        else:
            std = np.sqrt( cum_var )
        #std = np.sqrt( means.var(axis=1) )
        mean = means.mean(axis=1)
        plt.plot(indices, mean, linewidth=1, label=name_map[alg])
        plt.fill_between(indices, mean-std, mean+std, alpha=0.5)
        ms.append( mean.mean() )
    ## legend ordering
    ms = np.array(ms)
    order = np.argsort(-ms) ## decresing order
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    plt.legend(handles, labels)

    ## plotting
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.title(env+'-v2')
    plt.savefig(env+'.pdf')
