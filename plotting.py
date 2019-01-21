import os

import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import spline
from scipy.stats import norm
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg') # matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D

import utils
args = utils.get_args()

# By default, the plotter saves figures to the directory where it's executed.
FIG_DIR = ''
model_time = ''

def get_next_file(directory, model_time, ext, dot=".png"):
    i = 0
    fname = directory + model_time + ext
    while os.path.isfile(fname):
        fname = directory + model_time + ext + str(i) + dot
        i += 1
    return fname

def running_average_entropy(running_avg_entropies, running_avg_entropies_baseline):
    fname = get_next_file(FIG_DIR, model_time, "running_avg", ".png")
    plt.figure()
    plt.plot(np.arange(len(running_avg_entropies)), running_avg_entropies)
    plt.plot(np.arange(len(running_avg_entropies_baseline)), running_avg_entropies_baseline)
    plt.legend(["Entropy", "Random"])
    plt.xlabel("t")
    plt.ylabel("Running average entropy of cumulative policy")
    plt.title("Policy Entropy over Time")
    plt.savefig(fname)

def running_average_entropy_window(window_running_avg_ents, window_running_avg_ents_baseline, window):
    fname = get_next_file(FIG_DIR, model_time, "running_avg_window", ".png")
    plt.figure()
    plt.plot(np.arange(len(window_running_avg_ents)), window_running_avg_ents)
    plt.plot(np.arange(len(window_running_avg_ents_baseline)), window_running_avg_ents_baseline)
    plt.legend(["Entropy", "Random"])
    plt.xlabel("t")
    plt.ylabel("Running avg entropy")
    plt.title("Policy entropy over time, window = %d" % window)
    plt.savefig(fname)
 
def smear_lines(running_avg_ps, running_avg_ps_baseline):
    # want to plot the running_avg_p x_distribution over time
    plt.figure(3)

    smear_x = plt.subplot(211)
    smear_x.set_xlabel('t')
    smear_x.set_ylabel('Policy distribution over x')

    smear_v = plt.subplot(212)
    smear_v.set_xlabel('t')
    smear_v.set_ylabel('Policy distribution over v')

    for t in range(len(running_avg_ps)):
        running_avg_p = running_avg_ps[t]
        running_avg_p_baseline = running_avg_ps_baseline[t]
        x_distribution = np.sum(running_avg_p, axis=1)
        x_distribution_baseline = np.sum(running_avg_p_baseline, axis=1)
        v_distribution = np.sum(running_avg_p, axis=0)
        v_distribution_baseline = np.sum(running_avg_p_baseline, axis=0)

        # x data
        states = np.arange(x_distribution.shape[0])
        alphas = x_distribution.flatten()

        states_baseline = np.arange(x_distribution_baseline.shape[0])
        alphas_baseline = x_distribution_baseline.flatten()

        ls = np.linspace(0, x_distribution.shape[0], 100)
        estimate = np.interp(ls, states, alphas)
        estimate_baseline = np.interp(ls, states_baseline, alphas_baseline)

        colors = np.zeros((len(estimate),4))
        colors[:, 3] = estimate

        colors_baseline = np.zeros((len(estimate_baseline), 4))
        colors_baseline[:,0] = 1
        colors_baseline[:,3] = estimate_baseline

        smear_x.scatter(t*np.ones(shape=(len(estimate),1)), ls, color=colors)
        # smear_x.scatter(t*np.ones(shape=(len(estimate_baseline), 1)), ls, color=colors_baseline)

        # v data
        states = np.arange(v_distribution.shape[0])
        alphas = v_distribution.flatten()
        ls = np.linspace(0, v_distribution.shape[0], 100)
        estimate = np.interp(ls, states, alphas)

        colors = np.zeros((len(estimate),4))
        colors[:, 3] = estimate

        smear_v.scatter(t*np.ones(shape=(len(estimate),1)), ls, color=colors)

    fname = get_next_file(FIG_DIR, model_time, "running_avg_xv_distrs_smear_lines", ".png")
    plt.savefig(fname)

def smear_dots(running_avg_ps):
     # want to plot the running_avg_p x_distribution over time
    plt.figure()
    ax_x = plt.subplot(211)
    ax_v = plt.subplot(212)

    ax_x.set_xlabel('t')
    ax_v.set_xlabel('t')
    ax_x.set_ylabel('Policy distribution over x')
    ax_v.set_ylabel('Policy distribution over v')

    for t in range(len(running_avg_ps)):
        running_avg_p = running_avg_ps[t]
        x_distribution = np.sum(running_avg_p, axis=1)
        v_distribution = np.sum(running_avg_p, axis=0)

        alphas_x = x_distribution
        colors_x = np.zeros((x_distribution.shape[0],4))
        colors_x[:, 3] = alphas_x

        alphas_v = v_distribution
        colors_v = np.zeros((v_distribution.shape[0],4))
        colors_v[:, 3] = alphas_v

        ax_x.scatter(t*np.ones(shape=x_distribution.shape), x_distribution, color=colors_x)
        ax_v.scatter(t*np.ones(shape=v_distribution.shape), v_distribution, color=colors_v)
    fname = get_next_file(FIG_DIR, model_time, "running_avg_xv_distrs_smear_dot", ".png")
    plt.savefig(fname)

def heatmap(running_avg_p, avg_p, i):
    # Create running average heatmap.
    plt.figure()
    min_value = np.min(np.ma.log(running_avg_p))
    plt.imshow(np.ma.log(running_avg_p).filled(min_value), interpolation='spline16', cmap='Blues')

    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("v")
    if (args.env == "MountainCarContinuous-v0"):
        plt.ylabel("x")
    else:
        plt.ylabel(r"$\Theta$")
    # plt.title("Policy distribution at step %d" % i)
    running_avg_heatmap_dir = FIG_DIR + model_time + '/' + 'running_avg' + '/'
    if not os.path.exists(running_avg_heatmap_dir):
        os.makedirs(running_avg_heatmap_dir)
    fname = running_avg_heatmap_dir + "heatmap_%02d.png" % i
    plt.savefig(fname)

    # Create episode heatmap.
    plt.figure()
    min_value = np.min(np.ma.log(avg_p))
    plt.imshow(np.ma.log(avg_p).filled(min_value), interpolation='spline16', cmap='Blues')

    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("v")
    if (args.env == "MountainCarContinuous-v0"):
        plt.ylabel("x")
    else:
        plt.ylabel(r"$\Theta$")

    # plt.title("Policy distribution at step %d" % i)
    avg_heatmap_dir = FIG_DIR + model_time + '/' + 'avg' + '/'
    if not os.path.exists(avg_heatmap_dir):
        os.makedirs(avg_heatmap_dir)
    fname = avg_heatmap_dir + "heatmap_%02d.png" % i
    plt.savefig(fname)


def heatmap4(running_avg_ps, running_avg_ps_baseline, indexes=[0,1,2,3]):
    plt.figure()
    row1 = [plt.subplot(241), plt.subplot(242), plt.subplot(243), plt.subplot(244)]
    row2 = [plt.subplot(245), plt.subplot(246), plt.subplot(247), plt.subplot(248)]

    # min_value = np.min(np.ma.log(running_avg_ps))
    # min_value_baseline = np.min(np.ma.log(running_avg_ps_baseline))
    # min_value = np.minimum(min_value, min_value_baseline)

    # TODO: colorbar for the global figure
    for idx, ax in zip(indexes,row1):
        min_value = np.min(np.ma.log(running_avg_ps[idx]))
        ax.imshow(np.ma.log(running_avg_ps[idx]).filled(min_value), interpolation='spline16', cmap='Blues')
        ax.set_title("Epoch %d" % idx)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    
    for idx, ax in zip(indexes,row2):
        min_value = np.min(np.ma.log(running_avg_ps_baseline[idx]))
        ax.imshow(np.ma.log(running_avg_ps_baseline[idx]).filled(min_value), interpolation='spline16', cmap='Oranges')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    plt.tight_layout()
    fname = get_next_file(FIG_DIR, model_time, "time_heatmaps", ".png")
    plt.savefig(fname)
    # plt.colorbar()
    # plt.show()

def difference_heatmap(running_avg_ps, running_avg_ps_baseline):

    fname = get_next_file(FIG_DIR, model_time, "heatmap", ".png")
    entropy_p = running_avg_ps[len(running_avg_ps) - 1]
    random_p = running_avg_ps_baseline[len(running_avg_ps_baseline) - 1]

    plt.figure()
    diff_map = entropy_p - random_p

    # normalize so 0 is grey.
    max_p = max(abs(np.min(diff_map)), np.max(diff_map))
    plt.imshow(diff_map, vmin=-max_p, vmax=max_p, interpolation='spline16', cmap='coolwarm')
    plt.colorbar()
    plt.title(r'$p_{\pi_{entropy}} - p_{\pi_{random}}$')
    plt.savefig(fname)
    # plt.show()

def three_d_histogram(running_avg_ps):
    plt.figure(7)
    ax_x = plt.subplot(211)
    ax_v = plt.subplot(212, projection='3d')
    for t in range(len(running_avg_ps)):
        if (t % 10) != 0:
            continue

        running_avg_p = running_avg_ps[t]
        x_distribution = np.sum(running_avg_p, axis=1)
        v_distribution = np.sum(running_avg_p, axis=0)

        # state vs. value
        states_x = np.arange(x_distribution.shape[0])
        states_v = np.arange(v_distribution.shape[0])
        hist_states = np.zeros(shape=(len(states_x), len(states_v)))
        
        for idx in range(len(states_x)):
            for jdx in range(len(states_v)):
                x = states_x[idx]
                v = states_v[jdx]

                for tick in range(int(np.floor(running_avg_p[x][v]*1000))):
                    hist_states[x][v] += 1

        alphas_x = x_distribution.flatten()
        ax_x.plot(states_x, alphas_x)

        # n = len(states_x)                       
        # mean = sum(states_x*alphas_x)/n                  
        # sigma = sum(alphas_x*(states_x-mean)**2)/n    
        # print(sigma)
        # print(mean)

        # def gaus(x,a,x0,sigma):
        #     print(a*np.exp(-(x-x0)**2/(2*sigma**2)))
        #     return a*np.exp(-(x-x0)**2/(2*sigma**2))

        # popt,pcov = curve_fit(gaus, states_x, alphas_x, p0=[1, mean, sigma])
        # # ax_x.plot(states_x,gaus(states_x,*popt),'ro:',label='fit')
        # trial_x = np.linspace(0, n, 100)
        # ax_x.plot(trial_x, gaus(trial_x, *popt), 'r', label='fit')


        hist, xedges, yedges = np.histogram2d(hist_states[:,0], hist_states[:,1])
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
        xpos = xpos.flatten('F')
        ypos = ypos.flatten('F')
        zpos = np.zeros_like(xpos)

        # Construct arrays with the dimensions for the 16 bars.
        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = hist.flatten()

        ax_v.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

    fname = get_next_file(FIG_DIR, model_time, "_running_avg_plot3d", ".png")
    plt.savefig(fname)




