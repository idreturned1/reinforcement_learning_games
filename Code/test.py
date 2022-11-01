# Test Agent

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic

import time
from collections import deque
import csv
from datetime import datetime

def initializeFile(filename):
        with open(filename + str(".csv"),'a+',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['Reward'])
            writer.writerow([])
    
def writeMeta(filename,reward):
        with open(filename + str(".csv"),'a+',newline='') as f:
            writer=csv.writer(f)
            writer.writerow([reward])

datim = (str(datetime.now())[:-10]).replace(':','_') 
filename = datim 
initializeFile(filename)

# Making the test agent (won't update the model but will just use the shared model to explore)
def test(rank, params, shared_model,counter):
    torch.manual_seed(params.seed + rank) # asynchronizing the test agent
    env = create_atari_env(params.env_name, video=True) # running an environment with a video
    env.seed(params.seed + rank) # asynchronizing the environment
    model = ActorCritic(env.observation_space.shape[0], env.action_space) # creating one model
    model.eval() # putting the model in "eval" model because it won't be trained
    state = env.reset() # getting the input images as numpy arrays
    state = torch.from_numpy(state) # converting them into torch tensors
    reward_sum = 0 # initializing the sum of rewards to 0
    done = True # initializing done to True
    start_time = time.time() # getting the starting time to measure the computation time
    actions = deque(maxlen=100) 
    episode_length = 0 # initializing the episode length to 0
    while True: # repeat
        episode_length += 1 # incrementing the episode length by one
        if done: # synchronizing with the shared model (same as train.py)
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()
        with torch.no_grad():
            value, action_values, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(action_values, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()
        state, reward, done, _ = env.step(action[0,0])
        done = done or episode_length >= params.max_episode_length
        reward_sum += reward
        if done: # printing the results at the end of each part
            print("Time {}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
            writeMeta(filename,reward_sum)
            reward_sum = 0 # reinitializing the sum of rewards
            episode_length = 0 # reinitializing the episode length
            actions.clear() # reinitializing the actions
            state = env.reset() # reinitializing the environment
            time.sleep(60) # doing a one minute break to let the other agents practice (if the game is done)
        state = torch.from_numpy(state) # new state and we continue
