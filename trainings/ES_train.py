from __future__ import absolute_import, division, print_function

import os
import math
import numpy as np
import torch
import torch.legacy.optim as legacyOptim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable
from envs import create_env
from models.ES import ES
import matplotlib.pyplot as plt
import argparse


def ES_train(env_name):
    # set parameters as namespace object and give them values
    args = argparse.Namespace()
    args.env_name = env_name
    args.lr = 0.3 # learnig rate
    args.lr_decay = 1 #learning rate decay
    args.sigma = 0.05 # noise standard deviation
    args.n = 40 # batch size (even number)
    args.max_episode_length = 100000 # maximum length of an episode 
    args.max_gradient_updates = 100000
    args.restore = '' # restore checkpoint
    args.variable_ep_len = False # Change max episode length during training
    args.silent = False # Prints during training

    env = create_env(args.env_name, client_id="ES1",remotes=1) # Local docker container
    chkpt_dir = 'checkpoints/%s/' % args.env_name
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    synced_model = ES(env.observation_space.shape[0],env.action_space)
    for param in synced_model.parameters():
        param.requires_grad = False
    if args.restore:
        state_dict = torch.load(args.restore)
        synced_model.load_state_dict(state_dict)
    train_loop(args, synced_model, env, chkpt_dir)

def do_rollouts(args, models, random_seeds, return_queue, env, are_negative):
    """
    For each model, do a rollout. Supports multiple models per thread but
    don't do it -- it's inefficient (it's mostly a relic of when I would run
    both a perturbation and its antithesis on the same thread).
    """
    all_returns = []
    all_num_frames = []
    for model in models:
        cx = Variable(torch.zeros(1, 256))
        hx = Variable(torch.zeros(1, 256))
        state = env.reset()
        state = torch.from_numpy(state)
        this_model_return = 0
        this_model_num_frames = 0
        # Rollout
        for step in range(args.max_episode_length):
            logit, (hx, cx) = model(
                (Variable(state.unsqueeze(0), volatile=True),
                    (hx, cx)))
            prob = F.softmax(logit)
            action = prob.max(1)[1].data.numpy()
            state, reward, done, _ = env.step(action[0, 0])
            this_model_return += reward
            this_model_num_frames += 1
            if done:
                break
            state = torch.from_numpy(state)

        all_returns.append(this_model_return)
        all_num_frames.append(this_model_num_frames)
    return_queue.put((random_seeds, all_returns, all_num_frames, are_negative))


def perturb_model(args, model, random_seed, env):
    """
    Modifies the given model with a pertubation of its parameters,
    as well as the negative perturbation, and returns both perturbed
    models.
    """
    new_model = ES(env.observation_space.shape[0],
                   env.action_space)
    anti_model = ES(env.observation_space.shape[0],
                    env.action_space)
    new_model.load_state_dict(model.state_dict())
    anti_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (k, v), (anti_k, anti_v) in zip(new_model.es_params(),
                                        anti_model.es_params()):
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(args.sigma*eps).float()
        anti_v += torch.from_numpy(args.sigma*-eps).float()
    return [new_model, anti_model]

optimConfig = []
averageReward = []
maxReward = []
minReward = []
episodeCounter = []


def gradient_update(args, synced_model, returns, random_seeds, neg_list,
                    num_eps, num_frames, chkpt_dir, unperturbed_results):
    def fitness_shaping(returns):
        """
        A rank transformation on the rewards, which reduces the chances
        of falling into local optima early in training.
        """
        sorted_returns_backwards = sorted(returns)[::-1]
        lamb = len(returns)
        shaped_returns = []
        denom = sum([max(0, math.log(lamb/2 + 1, 2) -
                         math.log(sorted_returns_backwards.index(r) + 1, 2))
                     for r in returns])
        for r in returns:
            num = max(0, math.log(lamb/2 + 1, 2) -
                      math.log(sorted_returns_backwards.index(r) + 1, 2))
            shaped_returns.append(num/denom + 1/lamb)
        return shaped_returns

    def unperturbed_rank(returns, unperturbed_results):
        nth_place = 1
        for r in returns:
            if r > unperturbed_results:
                nth_place += 1
        rank_diag = ('%d out of %d (1 means gradient '
                     'is uninformative)' % (nth_place,
                                             len(returns) + 1))
        return rank_diag, nth_place

    batch_size = len(returns)
    assert batch_size == args.n
    assert len(random_seeds) == batch_size
    shaped_returns = fitness_shaping(returns)
    rank_diag, rank = unperturbed_rank(returns, unperturbed_results)
    print('Episode num: %d\n'
          'Average reward: %f\n'
          'Variance in rewards: %f\n'
          'Max reward: %f\n'
          'Min reward: %f\n'
          'Batch size: %d\n'
          'Max episode length: %d\n'
          'Sigma: %f\n'
          'Learning rate: %f\n'
          'Total num frames seen: %d\n'
          'Unperturbed reward: %f\n'
          'Unperturbed rank: %s\n\n'  %
          (num_eps, np.mean(returns), np.var(returns), max(returns),
          min(returns), batch_size,
          args.max_episode_length, args.sigma, args.lr, num_frames,
          unperturbed_results, rank_diag))

    averageReward.append(np.mean(returns))
    episodeCounter.append(num_eps)
    maxReward.append(max(returns))
    minReward.append(min(returns))

    pltAvg, = plt.plot(episodeCounter, averageReward, label='average')
    pltMax, = plt.plot(episodeCounter, maxReward,  label='max')
    pltMin, = plt.plot(episodeCounter, minReward,  label='min')

    plt.ylabel('rewards')
    plt.xlabel('episode num')
    plt.legend(handles=[pltAvg, pltMax,pltMin])

    fig1 = plt.gcf()

    plt.draw()
    fig1.savefig('graph.png', dpi=100)

    # For each model, generate the same random numbers as we did
    # before, and update parameters. We apply weight decay once.
    for i in range(args.n):
        np.random.seed(random_seeds[i])
        multiplier = -1 if neg_list[i] else 1
        reward = shaped_returns[i]
        for k, v in synced_model.es_params():
            eps = np.random.normal(0, 1, v.size())
            v += torch.from_numpy(args.lr/(args.n*args.sigma) *
                                    (reward*multiplier*eps)).float()
    args.lr *= args.lr_decay

    torch.save(synced_model.state_dict(),
               os.path.join(chkpt_dir, 'latest.pth'))
    return synced_model

def generate_seeds_and_models(args, synced_model, env):
    """
    Returns a seed and 2 perturbed models
    """
    np.random.seed()
    random_seed = np.random.randint(2**30)
    two_models = perturb_model(args, synced_model, random_seed, env)
    return random_seed, two_models


def train_loop(args, synced_model, env, chkpt_dir):
    def flatten(raw_results, index):
        notflat_results = [result[index] for result in raw_results]
        return [item for sublist in notflat_results for item in sublist]
    print("Num params in network %d" % synced_model.count_parameters())
    num_eps = 0
    total_num_frames = 0
    for _ in range(args.max_gradient_updates):
        processes = []
        return_queue = mp.Queue()
        all_seeds, all_models = [], []
        # Generate a perturbation and its antithesis
        for j in range(int(args.n/2)):
            random_seed, two_models = generate_seeds_and_models(args,
                                                                synced_model,
                                                                env)
            # Add twice because we get two models with the same seed
            all_seeds.append(random_seed)
            all_seeds.append(random_seed)
            all_models += two_models
        assert len(all_seeds) == len(all_models)
        # Keep track of which perturbations were positive and negative
        # Start with negative true because pop() makes us go backwards
        is_negative = True
        # Add all peturbed models to the queue
        while all_models:
            perturbed_model = all_models.pop()
            seed = all_seeds.pop()
            p = mp.Process(target=do_rollouts, args=(args,
                                                     [perturbed_model],
                                                     [seed],
                                                     return_queue,
                                                     env,
                                                     [is_negative]))
            p.start()
            processes.append(p)
            is_negative = not is_negative
        assert len(all_seeds) == 0
        # Evaluate the unperturbed model as well
        p = mp.Process(target=do_rollouts, args=(args, [synced_model],
                                                 ['dummy_seed'],
                                                 return_queue, env,
                                                 ['dummy_neg']))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()
        raw_results = [return_queue.get() for p in processes]
        seeds, results, num_frames, neg_list = [flatten(raw_results, index)
                                                for index in [0, 1, 2, 3]]
        # Separate the unperturbed results from the perturbed results
        _ = unperturbed_index = seeds.index('dummy_seed')
        seeds.pop(unperturbed_index)
        unperturbed_results = results.pop(unperturbed_index)
        _ = num_frames.pop(unperturbed_index)
        _ = neg_list.pop(unperturbed_index)

        total_num_frames += sum(num_frames)
        num_eps += len(results)
        synced_model = gradient_update(args, synced_model, results, seeds,
                                       neg_list, num_eps, total_num_frames,
                                       chkpt_dir, unperturbed_results)
        if args.variable_ep_len:
            args.max_episode_length = int(2*sum(num_frames)/len(num_frames))
        