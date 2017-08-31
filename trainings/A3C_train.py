import os
from multiprocessing import Process
import argparse
import time
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable

from envs import create_env
from models.A3C import ActorCritic

def A3C_train(env_name, num_process):
    """Train Actor to Critic model in separate process so Flask is responsive"""
    p = Process(target=train_model, args=(env_name,num_process))
    p.start()


def train_model(env_name, num_processes):
    """Start training of the model"""
    os.environ['OMP_NUM_THREADS'] = '1' #Use one thread per process

    # set parameters as namespace object and give them values
    args = argparse.Namespace()
    args.seed = 1 # default should be possible set on web page
    args.env_name = env_name
    args.num_processes = num_processes
    args.lr = 0.0001 # learning rate
    args.gamma = 0.99 # Discount factor for rewards
    args.tau = 1.0 # GAE parameter
    args.num_steps = 20 # Number of forward steps
    args.max_episode_length = 10000 # maximum length of an episode 
    args.no_shared = False # Use shared model
    

    torch.manual_seed(args.seed)

    env = create_env(env_name, client_id="A3C1",remotes=1) # Local docker container
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    # Not using Adam optimisation for now
    optimizer = None

    processes = []

    p = mp.Process(target=test, args=(num_processes, args, shared_model))
    p.start()
    processes.append(p)

    for rank in range(0, num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


optimConfig = []
averageReward = []
maxReward = []
minReward = []
episodeCounter = []

def get_rewards():
    """Return current histogram"""
    return {'averageReward':averageReward, 'maxReward':maxReward, 'minReward':minReward, 'episodeCounter':minReward}


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = create_env(args.env_name, client_id="A3C1", remotes=1)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            value, logit, (hx, cx) = model(
                (Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()


def test(rank, args, shared_model):
    torch.manual_seed(args.seed + rank)

    env = create_env(args.env_name, client_id="A3C1",remotes=1)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        value, logit, (hx, cx) = model(
            (Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)