import numpy as np
from tqdm import trange
import copy
import matplotlib.pyplot as plt

from ressim_enviroment import resSimEnv_v0, resSimEnv_v1

# # test case for random actions in env- Ressim-v0
# env = resSimEnv_v0(5, n_steps=60, dt=1e-3, k=50.0)
# n_action = env.action_space.n
# n_episodes = 1
# n_steps = 10
# cumRewards = []
# for i in range(n_episodes):
#     state = env.reset()
#     cumR = 0.0
#     for j in range(n_steps):
#         env.render(i,j)
#         action = np.random.randint(low=0, high=n_action)
#         state, reward, done, info = env.step(12)
#         cumR += reward
#     cumRewards.append(cumR)
#     print('Episode: '+str(i+1)+', reward:'+str(round(cumR,1)))



# # test case for random actions in env- Ressim-v1
# env = resSimEnv_v1(10, n_steps=60)
# n_action = env.action_space.n
# n_episodes = 5
# n_steps = 4
# cumRewards = []
# for i in range(n_episodes):
#     state = env.reset()
#     cumR = 0.0
#     for j in range(n_steps):
#         env.render(i,j)
#         action = np.random.randint(low=0, high=n_action)
#         state, reward, done, info = env.step(action)
#         cumR += reward
#     cumRewards.append(cumR)
#     print('Episode: '+str(i+1)+', reward:'+str(round(cumR,1)))

# solution for 100 step problem in ResSim-v1
n_step_ = 50
dt_ = 1e-3
mu_w_ = 1.0
mu_o_ = 2.0
lx_ = 1.0
ly_ = 1.0
nx_ = 10
ny_ = 10
phi_ = 0.1
k_ = 1
env = resSimEnv_v1(11,nx = nx_, ny=ny_, lx=lx_, ly=ly_ , n_steps=n_step_, dt=dt_, mu_w = mu_w_, mu_o = mu_o_, phi=phi_, k=k_)
n_action = env.action_space.n
N = 20
best_actions, best_rewards = [], []
for i in range(N):
    _ = env.reset()
    for a in best_actions:
        _,_,_,_ = env.step(a)
    max_ = -np.Inf
    for j in range(n_action):
        env_trial = copy.deepcopy(env)
        _,reward,_,_ = env_trial.step(j)
        print('step: '+str(i+1)+', action: '+str(j)+', reward: '+str(round(reward, 1)))
        if reward > max_:
            max_ = reward
            action = j
    best_actions.append(action)
    best_rewards.append(max_)
    print('step: '+str(i+1)+', best_action: '+str(action)+', best_reward: '+str(round(max_,1)))

cum_reward = np.cumsum(best_rewards)
output_table = np.stack((best_actions, best_rewards, cum_reward), axis=1)
np.savetxt('test_ressim_v1.csv', output_table, delimiter=',', header='Best Action, Best Reward, Cumumulative Reward')

# data = np.loadtxt('Ressim_Env/test_ressim_v1.csv', delimiter=',', skiprows=1)
# cum_reward = data[:,2]

# plt.clf()
# plt.plot(cum_reward )
# plt.grid('True')
# plt.xlabel('steps')
# plt.ylabel('maximum reward')
# plt.title('ResSim-v1 global maxima')# solution for 100 step problem in ResSim-v1
# plt.savefig('Ressim_Env/ressim_V1_global maxima.png')

# # solution for 100 step problem in ResSim-v0
# env = resSimEnv_v0(5, n_steps=60, dt=1e-3)
# n_action = env.action_space.n
# N = 100
# best_actions, best_rewards = [], []
# for i in range(N):
#     _ = env.reset()
#     for a in best_actions:
#         _,_,_,_ = env.step(a)
#     max_ = -np.Inf
#     for j in range(n_action):
#         env_trial = copy.deepcopy(env)
#         _,reward,_,_ = env_trial.step(j)
#         # print('step: '+str(i+1)+', action: '+str(j)+', reward: '+str(round(reward, 1)))
#         if reward > max_:
#             max_ = reward
#             action = j
#     best_actions.append(action)
#     best_rewards.append(max_)
#     print('step: '+str(i+1)+', best_action: '+str(action)+', best_reward: '+str(round(max_,1)))

# output_table = np.stack((best_actions, best_rewards), axis=1)
# np.savetxt('Ressim_Env/test_ressim_v0.csv', output_table, delimiter=',', header='Best Action, Best Reward')

# data = np.loadtxt('Ressim_Env/test_ressim_v0.csv', delimiter=',', skiprows=1)
# cum_reward = np.cumsum(data[:,1])

# plt.clf()
# plt.plot(cum_reward )
# plt.grid('True')
# plt.xlabel('steps')
# plt.ylabel('maximum reward')
# plt.title('ResSim-v0 global maxima')
# plt.savefig('Ressim_Env/ressim_V0_global maxima.png')

# # all combinations of actions for N step problem in ResSim-v0
# from itertools import permutations
# env = resSimEnv_v0(5, n_steps=60, dt=1e-3)
# n_action = env.action_space.n
# N = 3
# a = permutations(range(n_action), N)
# count = 0
# for i in a:
#     if count>11951:
#         _ = env.reset()
#         for j in range(N):
#             _,_,_,_ = env.step(i[j])
#             print(str(i)+'->'+str(j))
#     count+=1