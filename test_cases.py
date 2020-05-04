import numpy as np
from tqdm import trange
import copy
import matplotlib.pyplot as plt
import time

from ressim_enviroment import resSimEnv_v0, resSimEnv_v1

# print('test 1: const action, const k')
# # test case for const actions in env- Ressim-v0
# t0=time.time()
# env = resSimEnv_v0(5, n_steps=1, dt=1e-3)
# n_action = env.action_space.n
# n_episodes = 3
# n_steps_ep = 1
# cumRewards = []
# for i in range(n_episodes):
#     state = env.reset()
#     cumR = 0.0
#     for j in range(n_steps_ep):
#         # env.render(i,j)
#         state, reward, done, info = env.step(12)
#         cumR += reward
#     cumRewards.append(cumR)
#     print('Episode: '+str(i+1)+', reward:'+str(round(cumR,1)))
# print('time taken for test: '+str(round(time.time() - t0))+' secs')

# print('test 2: const action, random k')
# # test case for random k in env- Ressim-v0
# t0=time.time()
# env = resSimEnv_v0(5, n_steps=60, dt=1e-3, k_type='random')
# n_action = env.action_space.n
# n_episodes = 10
# n_steps_ep = 10
# cumRewards = []
# for i in range(n_episodes):
#     state = env.reset()
#     cumR = 0.0
#     for j in range(n_steps_ep):
#         # env.render(i,j)
#         state, reward, done, info = env.step(12)
#         cumR += reward
#     cumRewards.append(cumR)
#     print('Episode: '+str(i+1)+', reward:'+str(round(cumR,1)))
# print('time taken for test: '+str(round(time.time() - t0))+' secs')

# print('test 3: random action, random k')
# # test case for random actions in env- Ressim-v0
# t0=time.time()
# env = resSimEnv_v0(5, n_steps=60, dt=1e-3, k_type='random')
# n_action = env.action_space.n
# n_episodes = 10
# n_steps_ep = 10
# cumRewards = []
# for i in range(n_episodes):
#     state = env.reset()
#     cumR = 0.0
#     for j in range(n_steps_ep):
#         # env.render(i,j)
#         state, reward, done, info = env.step(np.random.randint(0,n_action))
#         cumR += reward
#     cumRewards.append(cumR)
#     print('Episode: '+str(i+1)+', reward:'+str(round(cumR,1)))
# print('time taken for test: '+str(round(time.time() - t0))+' secs')

print('test 4: unstable action for py')
# test case for erronous actions in env- Ressim-v0
action_list = [[4,23,0], [9,4,5], [15,20,4], [20, 1, 0], [20,15,15], [4,9,15]]
t0=time.time()
env = resSimEnv_v0(5, n_steps=60, dt=1e-3)
n_action = env.action_space.n
n_episodes = 6
n_steps_ep = 3
cumRewards = []
for i in range(n_episodes):
    state = env.reset()
    cumR = 0.0
    for j in range(n_steps_ep):
        # env.render(i,j)
        state, reward, done, info = env.step( action_list[i][j] )
        cumR += reward
    cumRewards.append(cumR)
    print('Episode: '+str(i+1)+', reward:'+str(round(cumR,1)))
print('time taken for test: '+str(round(time.time() - t0))+' secs')