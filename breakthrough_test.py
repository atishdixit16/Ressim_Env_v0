import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from ressim_enviroment import resSimEnv_v1
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
n_episodes = 11
n_ep_steps = 1
cumRewards = []
for i in range(n_episodes):
    state = env.reset()
    cumR = 0.0
    sw_p1 = [env.get_sw(0,0)]
    sw_p2 = [env.get_sw(-1,0)]
    for j in trange(n_ep_steps):
        # env.render(i,j)
        action = i
        state, reward, done, info = env.step(action)
        sw_p1.append(env.get_sw(0,0))
        sw_p2.append(env.get_sw(-1,0))
        cumR += reward
    cumRewards.append(cumR)
    print('Action: '+str(action)+', reward:'+str(round(cumR,1)))


# plt.clf()
# x = range(n_ep_steps+1)
# plt.plot(x, sw_p1, x, sw_p2 )
# plt.grid('True')
# plt.xlabel('steps')
# plt.ylabel('water saturation at producer')
# plt.hlines(y=0.8, xmin=0, xmax=n_ep_steps, linestyles='dashed')
# plt.title('Environment: ResSim-v1\nnstep='+str(n_step_)+'; dt='+str(dt_))
# plt.legend(('Sw_producer_1', 'Sw_producer_2'))
# plt.show()