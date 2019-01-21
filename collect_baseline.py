# Collect entropy-based reward policies.

# Changed from using all-1 reward to init to one-hot at: 2018_11_30-10-00

# python collect_baseline.py --env="MountainCarContinuous-v0" --T=200 --train_steps=400 --episodes=300 --epochs=50 --exp_name=test

import os
import time
from datetime import datetime
import logging

import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import spline
from scipy.stats import norm

import gym
from gym.spaces import prng

from cart_entropy_policy import CartEntropyPolicy
import utils
import curiosity
import plotting

import torch
from torch.distributions import Normal
import random

from itertools import islice

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def moving_averages(values, size):
    for selection in window(values, size):
        yield sum(selection) / size

args = utils.get_args()

Policy = CartEntropyPolicy

def grad_ent(pt):
    
    if args.grad_ent:
        grad_p = -np.log(pt)
        grad_p[grad_p > 100] = 1000
        return grad_p

    eps = 1/np.sqrt(utils.total_state_space)
    return 1/(pt + eps)

# Get the initial zero-state for the env.
def init_state(env):
    if env == "Pendulum-v0":
        return [np.pi, 0] 
    elif env == "MountainCarContinuous-v0":
        return [-0.50, 0]

# Main loop of maximum entropy program. Iteratively collect 
# and learn T policies using policy gradients and a reward function 
# based on entropy.
def collect_entropy_policies(env, epochs, T, MODEL_DIR):

    reward_fn = np.zeros(shape=(tuple(utils.num_states)))

    # set initial state to base, motionless state.
    seed = []
    if args.env == "Pendulum-v0":
        env.env.state = [np.pi, 0]
        seed = env.env._get_obs()
    elif args.env == "MountainCarContinuous-v0":
        env.env.state = [-0.50, 0]
        seed = env.env.state

    reward_fn[tuple(utils.discretize_state(seed))] = 1

    running_avg_p = np.zeros(shape=(tuple(utils.num_states)))
    running_avg_ent = 0
    window_running_avg_p = np.zeros(shape=(tuple(utils.num_states)))
    window_running_avg_ent = 0

    running_avg_p_baseline = np.zeros(shape=(tuple(utils.num_states)))
    running_avg_ent_baseline = 0
    window_running_avg_p_baseline = np.zeros(shape=(tuple(utils.num_states)))
    window_running_avg_ent_baseline = 0

    baseline_entropies = []
    baseline_ps = []
    entropies = []
    ps = []

    average_entropies = []
    average_ps = []

    running_avg_entropies = []
    running_avg_ps = []

    running_avg_entropies_baseline = []
    running_avg_ps_baseline = []

    window_running_avg_ents = []
    window_running_avg_ps = []
    window_running_avg_ents_baseline = []
    window_running_avg_ps_baseline = []

    policies = []
    initial_state = init_state(args.env)

    for i in range(epochs):

        # Learn policy that maximizes current reward function.
        policy = Policy(env, args.gamma, args.lr, utils.obs_dim, utils.action_dim) 
        policy.learn_policy(reward_fn, initial_state=initial_state, episodes=args.episodes, train_steps=args.train_steps)
        policies.append(policy)

        if args.save_models:
            policy.save(MODEL_DIR + 'model_' + str(i) + '.pt')


        # Get next distribution p by executing pi for T steps.
        p_videos = 'cmp_videos/%sp_%d/'% (MODEL_DIR, i) 
        p = policy.execute(T, initial_state=initial_state, render=args.record, video_dir=p_videos)
        
        a = 10 # average over this many rounds
        baseline_videos = 'cmp_videos/%sbaseline_%d/'% (MODEL_DIR, i) # note that MODEL_DIR has trailing slash
        entropy_videos = 'cmp_videos/%sentropy_%d/'% (MODEL_DIR, i)
        p_baseline = policy.execute_random(T, render=False, video_dir=baseline_videos) # args.episodes?
        round_entropy_baseline = scipy.stats.entropy(p_baseline.flatten())
        for av in range(a - 1):
            next_p_baseline = policy.execute_random(T)
            p_baseline += next_p_baseline
            # print(scipy.stats.entropy(next_p_baseline.flatten()))
            round_entropy_baseline += scipy.stats.entropy(next_p_baseline.flatten())
        p_baseline /= float(a)
        round_entropy_baseline /= float(a) # running average of the entropy
        
        # note: the entropy is p_baseline is not the same as the computed avg entropy
        # print("baseline compare:")
        # print(round_entropy_baseline) # running average
        # print(scipy.stats.entropy(p_baseline.flatten())) # entropy of final

        # reward_fn = grad_ent(p)

        round_entropy = scipy.stats.entropy(p.flatten())
        entropies.append(round_entropy)
        baseline_entropies.append(round_entropy_baseline)
        ps.append(p)
        baseline_ps.append(p_baseline)

        # Execute the cumulative average policy thus far.
        # Estimate distribution and entropy.
        average_p, round_avg_ent, initial_state = \
            curiosity.execute_average_policy(env, policies, T, initial_state=initial_state, avg_runs=a, render=False, video_dir=entropy_videos)

        # If in pendulum, set velocity to 0 with some probability
        if args.env == "Pendulum-v0" and random.random() < 0.3:
            initial_state[1] = 0

        reward_fn = grad_ent(average_p)

        print(average_p)
        print("!  --------  !")
        print(reward_fn)

        average_ps.append(average_p)
        average_entropies.append(round_avg_ent) 
        
        # Update running average.
        window = 5
        if (i < window): # add normally
            window_running_avg_ent = window_running_avg_ent * (i)/float(i+1) + round_avg_ent/float(i+1)
            window_running_avg_p = window_running_avg_ent * (i)/float(i+1) + average_p/float(i+1)
            window_running_avg_ent_baseline = window_running_avg_ent_baseline * (i)/float(i+1) + round_entropy_baseline/float(i+1)
            window_running_avg_p_baseline = window_running_avg_p_baseline * (i)/float(i+1) + p_baseline/float(i+1)

        else:
            window_running_avg_ent = window_running_avg_ent + round_avg_ent/float(window) - average_entropies[i-5]/float(window)
            window_running_avg_p = window_running_avg_p + average_p/float(window) - average_ps[i-5]/float(window)
            
            window_running_avg_ent_baseline = window_running_avg_ent_baseline + round_entropy_baseline/float(window) - baseline_entropies[i-5]/float(window)
            window_running_avg_p_baseline = window_running_avg_p_baseline + p_baseline/float(window) - baseline_ps[i-5]/float(window)
        

        running_avg_ent = running_avg_ent * (i)/float(i+1) + round_avg_ent/float(i+1)
        running_avg_p = running_avg_p * (i)/float(i+1) + average_p/float(i+1)
        running_avg_entropies.append(running_avg_ent)
        running_avg_ps.append(running_avg_p)     

        # Update baseline running averages.
        running_avg_ent_baseline = running_avg_ent_baseline * (i)/float(i+1) + round_entropy_baseline/float(i+1)
        running_avg_p_baseline = running_avg_p_baseline * (i)/float(i+1) + p_baseline/float(i+1)
        running_avg_entropies_baseline.append(running_avg_ent_baseline)
        running_avg_ps_baseline.append(running_avg_p_baseline) 

        window_running_avg_ents.append(window_running_avg_ent)
        window_running_avg_ps.append(window_running_avg_p)
        window_running_avg_ents_baseline.append(window_running_avg_ent_baseline)
        window_running_avg_ps_baseline.append(window_running_avg_p_baseline)

        print("p=")
        print(p)
        print("..........")
        print("round_entropy = %f" % (round_entropy))

        print("---------------------")

        print("average_p =") 
        print(average_p)

        print("..........")

        print("round_avg_ent[%d] = %f" % (i, round_avg_ent))
        print("running_avg_ent = %s" % running_avg_ent)
        print("window_running_avg_ent = %s" % window_running_avg_ent)

        print("..........")

        print("round_entropy_baseline[%d] = %f" % (i, round_entropy_baseline))
        print("running_avg_ent_baseline = %s" % running_avg_ent_baseline)
        print("window_running_avg_ent_baseline = %s" % window_running_avg_ent_baseline)
        # print("running_avg_p_baseline =") 
        # print(running_avg_p_baseline)

        print("----------------------")

        plotting.heatmap(running_avg_p, average_p, i)

    # plotting.smear_lines(running_avg_ps, running_avg_ps_baseline)
    plotting.running_average_entropy(running_avg_entropies, running_avg_entropies_baseline)
    plotting.running_average_entropy_window(window_running_avg_ents, window_running_avg_ents_baseline, window)
    # plotting.difference_heatmap(running_avg_ps, running_avg_ps_baseline)

    indexes = [1,5,10,20]
    # print('which indexes?')
    # for i in range(4):
    #     idx = input("index :")
    #     indexes.append(int(idx))
    plotting.heatmap4(running_avg_ps, running_avg_ps_baseline, indexes)

    return policies


def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100)

    # Make environment.
    env = gym.make(args.env)
    # TODO: limit acceleration (maybe also speed?) for Pendulum.
    if args.env == "Pendulum-v0":
        env.env.max_speed = 8
        env.env.max_torque = 1
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    TIME = datetime.now().strftime('%Y_%m_%d-%H-%M')
    MODEL_DIR = 'models-' + args.env + '/models_' + TIME + '/'

    if args.save_models:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        # save metadata from the run. 
        with open(MODEL_DIR + "metadata", "w") as metadata:
            metadata.write("args: %s\n" % args)
            metadata.write("num_states: %s\n" % str(utils.num_states))
            metadata.write("state_bins: %s\n" % utils.state_bins)

    plotting.FIG_DIR = 'figs/' + args.env + '/'
    plotting.model_time = args.exp_name + '/'
    if not os.path.exists(plotting.FIG_DIR+plotting.model_time):
        os.makedirs(plotting.FIG_DIR+plotting.model_time)

    policies = collect_entropy_policies(env, args.epochs, args.T, MODEL_DIR)

    exploration_policy = curiosity.average_policies(env, policies)
    
    # Final policy:
    # average_p, _, _ = curiosity.execute_average_policy(env, policies, args.T)
    # overall_avg_ent = scipy.stats.entropy(average_p.flatten())
    # print('*************')
    # print(np.reshape(average_p, utils.space_dim))
    # print("overall_avg_ent = %f" % overall_avg_ent)

    env.close()

    print("DONE")

if __name__ == "__main__":
    main()


