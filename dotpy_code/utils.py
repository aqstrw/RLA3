#!/usr/bin/env python3
"""
Assignment Submission for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Ambar Qadeer
"""

"""
convenience functions
"""

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''
    return savgol_filter(y, window, poly)


def save_data(data, ep_num, ep, lr, gamma):
    if ep_num > ep:
        temp = data[-1]
        temp = np.pad(temp, (0, ep_num - len(temp)), mode='constant', constant_values=(np.nan))
        data[-1] = temp
    fname = "reinforce_v1_lr" + "{:.4f}".format(lr)[-4:] + "_g" + "{:.4f}".format(gamma)[-4:] + "_runs{:1d}_eps".format(
        runs) + str(ep_num) + ".csv"
    np.savetxt(fname, data, delimiter=",")
    print("data saved in file {}".format(fname))
    if ep_num == ep:
        plot_smoothed_scores(score_stack)


def plot_smoothed_scores(data, save=1, fname="demo.pdf"):
    for run, scores in enumerate(data):
        avg_score_hist = smooth(scores, smoothen_over)
        plt.plot(avg_score_hist, label="run " + str(run + 1))
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    avg = np.nanmean(data, axis=0)
    plt.plot(avg, label="average over " + str(len(data)) + " runs")
    plt.legend()
    if save:
        plt.savefig(fname)
    plt.show()

def read_data(fname):
    data = np.loadtxt(fname, delimiter = "," )
    return data

def plot_smoothed_scores_wip(data, save = 1, fname = "demo.pdf"):
    '''
    Plot incomplete score_stacks (deals with nans errors in savgol filters)
    '''
    for run, scores in enumerate(data):
        try:
            avg_score_hist = smooth(scores,smoothen_over)
            plt.plot(avg_score_hist, label = "run "+str(run+1))
        except Exception as e:
            print(str(e)+"for run {}".format(run+1))
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    avg = np.nanmean(data, axis = 0)
    std = np.std(data, axis = 0)
#     print(std.shape)
#     print(avg.shape)
    plt.fill_between(range(len(avg)),smooth(avg-std, smoothen_over), smooth(avg+std, smoothen_over), alpha = 0.2)
    plt.plot(avg, label = "average over "+str(len(data))+" runs")
    plt.legend()
    if save:
        plt.savefig(fname)
    plt.show()