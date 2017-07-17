import gym
import universe # register Universe environments into Gym
from envs import create_env

def start_game(model, env_name):
    """Play game with saved model if any otherwise play random"""
    env = create_env(env_name, client_id="play1",remotes=1) # Local docker container
    observation_n = env.reset()
    done_n = False
    i = 0
    while not done_n:
        i+=1
        # agent which presses the Up arrow 60 times per second
        action_n = [[('KeyEvent', 'ArrowUp', True)] for _ in observation_n]
        observation_n, reward_n, done_n, info = env.step(action_n)
        print('Loop run: %d\n' % i)