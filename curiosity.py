# experimenting with curiosity exploration method.
# Code derived from: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

# example command setting args in utils.py
# python curiosity.py  --models_dir=models-MountainCarContinuous-v0/models_2018_11_28-17-45/ --env="MountainCarContinuous-v0"
# python curiosity.py  --models_dir=models-Pendulum-v0/models_2018_11_29-09-48/ --env="Pendulum-v0"

import os
import sys
import time

import random
import numpy as np
import scipy.stats
import gym
from gym.spaces import prng
from gym import wrappers

import torch
from torch.distributions import Categorical

from cart_entropy_policy import CartEntropyPolicy
import utils

args = utils.get_args()

Policy = CartEntropyPolicy

# Average the weights of all the policies. Use to intialize a new Policy object.
def average_policies(env, policies):
    state_dict = policies[0].state_dict()
    for i in range(1, len(policies)):
        for k, v in policies[i].state_dict().items():
            state_dict[k] += v

    for k, v in state_dict.items():
        state_dict[k] /= float(len(policies))
     # obtain average policy.
    average_policy = Policy(env, args.gamma, args.lr, utils.obs_dim, utils.action_dim)
    average_policy.load_state_dict(state_dict)

    return average_policy


def select_action(probs):
    m = Categorical(probs)
    action = m.sample()
    if (action.item() == 1):
        return [0]
    elif (action.item() == 0):
        return [-1]
    return [1]

def load_from_dir(directory):
    policies = []
    files = os.listdir(directory)
    for file in sorted(files):
        if (file == "metadata"):
            print("skipping: " + file)
            continue
        policy = torch.load(directory + file)
        policies.append(policy)
    return policies

def get_obs(state):
    if utils.args.env == "Pendulum-v0":
        theta, thetadot = state
        return np.array([np.cos(theta), np.sin(theta), thetadot])
    elif utils.args.env == "MountainCarContinuous-v0":
        return np.array(state)

    # unroll for T steps and compute p
def execute_policy_internal(env, T, policies, state, render):
    random_T = np.floor(random.random()*T)
    p = np.zeros(shape=(tuple(utils.num_states)))
    random_initial_state = []

    for t in range(T):
        # Compute average probability over action space for state.
        probs = torch.tensor(np.zeros(shape=(1,utils.action_dim))).float()
        var = torch.tensor(np.zeros(shape=(1,utils.action_dim))).float()
        for policy in policies:
            prob = policy.get_probs(state)
            probs += prob
        probs /= len(policies)
        action = select_action(probs)
        
        state, reward, done, _ = env.step(action)
        p[tuple(utils.discretize_state(state))] += 1
        if (t == random_T and not render):
            random_initial_state = env.env.state

        if render:
            env.render()
        if done:
            break # env.reset()

    p /= float(T)
    return p, random_initial_state

# run a simulation to see how the average policy behaves.
def execute_average_policy(env, policies, T, initial_state=[], avg_runs=1, render=False, video_dir=''):
    
    average_p = np.zeros(shape=(tuple(utils.num_states)))
    avg_entropy = 0
    random_initial_state = []

    last_run = avg_runs-1
    for i in range(avg_runs):
        if len(initial_state) == 0:
            initial_state = env.reset()

        # NOTE: this records ONLY the final run. 
        if video_dir != '' and render and i == last_run:
            wrapped_env = wrappers.Monitor(env, video_dir)
            wrapped_env.unwrapped.reset_state = initial_state
            state = wrapped_env.reset()
            # state = get_obs(state)
            
            p, random_initial_state = execute_policy_internal(wrapped_env, T, policies, state, True)
            average_p += p
            avg_entropy += scipy.stats.entropy(average_p.flatten())

        else:
            env.env.reset_state = initial_state
            state = env.reset()
            # state = get_obs(state)
            p, random_initial_state = execute_policy_internal(env, T, policies, state, False)
            average_p += p
            avg_entropy += scipy.stats.entropy(average_p.flatten())

    env.close()
    average_p /= float(avg_runs)

    avg_entropy /= float(avg_runs) # running average of the entropy 
    entropy_of_final = scipy.stats.entropy(average_p.flatten())

    # print("compare:")
    # print(avg_entropy) # running entropy
    # print(entropy_of_final) # entropy of the average distribution

    return average_p, avg_entropy, random_initial_state


def average_p_and_entropy(env, policies, T, avg_runs=1, render=False, video_dir=''):
    exploration_policy = average_policies(env, policies)
    average_p = exploration_policy.execute(T, render=render, video_dir=video_dir)
    average_ent = scipy.stats.entropy(average_p.flatten())
    for i in range(avg_runs-1):
        p = exploration_policy.execute(T)
        average_p += p
        average_ent += scipy.stats.entropy(p.flatten())
        # print(scipy.stats.entropy(p.flatten()))
    average_p /= float(avg_runs) 
    average_ent /= float(avg_runs) # 
    entropy_of_final = scipy.stats.entropy(average_p.flatten())

    # print("entropy compare: ")
    # print(average_ent) # running entropy
    # print(entropy_of_final) # entropy of final

    # return average_p, avg_entropy
    return average_p, average_ent

def main():
    # Suppress scientific notation.
    np.set_printoptions(suppress=True)

    # Make environment.
    env = gym.make(args.env)
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    # Set up experiment variables.
    T = 1000
    avg_runs = 10

    policies = load_from_dir(args.models_dir)
    
    times = []
    entropies = []

    x_dist_times = []
    x_distributions = []

    v_dist_times = []
    v_distributions = []
    
    for t in range(1, len(policies)):

        average_p, avg_entropy = average_p_and_entropy(policies[:t], avg_runs)
        
        print('---------------------')
        print("Average policies[:%d]" % t)
        print(average_p)
        print(avg_entropy)

    # obtain global average policy.
    exploration_policy = average_policies(env, policies)
    average_p = exploration_policy.execute(T)
   
    print('*************')
    print(average_p)

    env.close()
        

if __name__ == "__main__":
    main()





