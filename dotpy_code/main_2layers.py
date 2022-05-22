#!/usr/bin/env python3
"""
Assignment Submission for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Ambar Qadeer
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_smoothed_scores_wip, save_data
from agent_2layers import reinforce_agent

plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams.update({'font.size': 14})


# set hyperparameters here

# learnrate = 0.001
# gam = 0.99
# runs = 8
# ep_num = 2000
# save_data_cadence = 500
# smoothen_over = 11
# if save_data_cadence>ep_num:
#     print("WARNING : data will not get saved if save cadence is more than ep_num")
#
# # bookkeeping
# fname = "reinforce_v1_lr"+"{:.4f}".format(learnrate)[-4:]+"_g"+"{:.4f}".format(gam)[-4:]+"_runs{:1d}_eps".format(runs)+str(ep_num)


# testing
learnrate = 0.001
gam = 0.99
runs = 3
ep_num = 20
save_data_cadence = 10
smoothen_over = 3
if save_data_cadence>ep_num:
    print("WARNING : data will not get saved if save cadence is more than ep_num")
# bookkeeping
fname = "reinforce_v1_lr"+"{:.4f}".format(learnrate)[-4:]+"_g"+"{:.4f}".format(gam)[-4:]+"_runs{:1d}_eps".format(runs)+str(ep_num)

if __name__ == "__main__":

    score_stack = []
    # run loop
    for run_num in range(runs):

        print("\n\n\n\nbeginning run number {}".format(run_num+1))

        # initialise environment
        env = gym.make('CartPole-v1')

        # initialize agent
        agent_007 = reinforce_agent(lr=learnrate, gamma=gam, n_actions=env.action_space.n, n_states=env.observation_space.shape)

        # list to maintain score
        score_hist = []
        score_stack.append(score_hist)

        # episode loop
        for ep in range(ep_num):
            done = False
            score = 0
            s = env.reset()

            # step loop
            while not done:
                a = agent_007.get_act(s)
                s_next, reward, done, _ = env.step(a)
                agent_007.remember(s, a, reward)
                s = s_next
                score += reward
            #                 env.render()

            # append score to score list
            score_hist.append(score)

            # make agent learn
            agent_007.learn()
            avg_score = np.mean(score_hist[-100:])
            print('episode: ', ep, 'score: %.1f' % score,
                  ', average score %.1f' % avg_score)

            if (ep + 1) % save_data_cadence == 0:
                score_stack[-1] = score_hist
                save_data(score_stack, ep_num, ep, fname, sav_win = smoothen_over)

# plot results
plot_smoothed_scores_wip(score_stack, sav_window = smoothen_over, avg_plot = 1, save = 1, fname = fname)
