import gym
import universe
from envs import create_env
from multiprocessing import Process
import time
from universe.spaces.vnc_event import keycode

def start_game(model, env_name):
    """regular Python process, not using torch"""
    #p = Process(target=play_game, args=(model,env_name))
    #p.start()
    # Don't wait with join, respond to user
#def play_game(model, env_name):
#    """Play game with saved model if any otherwise play random"""
    env = create_env(env_name, client_id="play1",remotes=1) # Local docker container
    max_game_length = 10000
    state = env.reset()
    reward_sum = 0
    start_time = time.time()
    for step in range(max_game_length ):
        state, reward, done, _ = env.step( ['up' for i in range(60)]) #keep pressing up, 60 times in minute
        reward_sum += reward
        env.render() #render environment
        if done:
            print("Time {}, game reward {}, game length {}".format(
                time.strftime("%Hh %Mm %Ss"),
                reward_sum,
                time.gmtime(time.time() - start_time)))
            break