#!/usr/bin/env python3
"""
Assignment Submission for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Ambar Qadeer
"""

"""
convenience functions ( some functions were borrowed from documents provided by the course team )
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window
    '''
    return savgol_filter(y, window, poly)


def save_data(data, ep_num, ep, fname = "demo", sav_win = 11):
    '''
    save data from numpy array into a csv file
    '''
    if ep_num > ep:
        temp = data[-1]
        temp = np.pad(temp, (0, ep_num - len(temp)), mode='constant', constant_values= (np.nan) )
        data[-1] = temp

    np.savetxt(fname+".csv", data, delimiter=",")
    print("data saved in file {}.csv".format(fname))

    plot_smoothed_scores_wip(data, sav_window = sav_win, avg_plot = 0, save = 1, fname = fname)

# def plot_smoothed_scores(data, save=1, fname="demo.pdf"):
#     '''
#     obsolete plot function, not used in latest plots
#     '''
#     for run, scores in enumerate(data):
#         avg_score_hist = smooth(scores, smoothen_over)
#         plt.plot(avg_score_hist, label="run " + str(run + 1))
#     plt.xlabel("Episodes")
#     plt.ylabel("Rewards")
#     avg = np.nanmean(data, axis=0)
#     plt.plot(avg, label="average over " + str(len(data)) + " runs")
#     plt.legend()
#     if save:
#         plt.savefig(fname)
#     plt.show()


def read_data(fname):
    '''
    read data from file into a numpy array
    '''
    data = np.loadtxt(fname+".csv", delimiter = "," )
    return data

def plot_smoothed_scores_wip(data, sav_window = 11, avg_plot=1, save=1, fname="demo"):
    '''
    Plot incomplete score_stacks (deals with nans errors in savgol filters)
    '''
    success = 0

    # first plot
    fig, axs = plt.subplots(1, 1, figsize=(14, 7))
    for run, scores in enumerate(data):
        try:
            avg_score_hist = smooth(scores, sav_window)
            axs.plot(avg_score_hist, label="run " + str(run + 1))
            success = 1
        except Exception as e:
            print(str(e) + " for run {}".format(run + 1))

    if success:
        # set labels
        axs.set_xlabel("Episodes")
        axs.set_ylabel("Rewards")

        # plot legends
        axs.legend()

        # save plot
        if save:
            plt.tight_layout()
            fig.savefig(fname + "_runs.pdf")

        # plot legends
        axs.legend()

        # display fig
        fig.show()

    if avg_plot:
        print("working on average plot")
        # calculate stdev and avg of all runs and plot
        avg = np.nanmean(data, axis=0)
        std = np.std(data, axis=0)

        # second plot
        fig, axs = plt.subplots(1, 1, figsize=(14, 7))
        axs.fill_between(range(len(avg)), smooth(avg - std, sav_window), smooth(avg + std, sav_window), alpha=0.2)
        axs.plot(avg, label="average over " + str(len(data)) + " runs")

        # set labels
        axs.set_xlabel("Episodes")
        axs.set_ylabel("Rewards")

        # plot legends
        axs.legend()

        # save plot
        if save:
            plt.tight_layout()
            fig.savefig(fname + "_average.pdf")

        # display fig
        fig.show()
