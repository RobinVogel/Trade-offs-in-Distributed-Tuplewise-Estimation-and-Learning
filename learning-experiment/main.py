"""
    Code for:
    Trade-offs in Large Scale Distributed Tuplewise Estimation and Learning
    Author: Robin Vogel
"""
import os
import argparse
import logging
import json
import matplotlib.pyplot as plt
import numpy as np

import make_exps as me

DEFAULT_BASE_DIR = "exps"

def make_runs(start_run=0, end_run=25,
              base_dir=DEFAULT_BASE_DIR):
    """Starts runs indexed between start_run and end_run on the database."""
    for i, reshuffle_mod in enumerate([1, 5, 25, 125, 10000]):
        for j in range(start_run, end_run):
            # Remove all handlers associated with the root logger object.
            # Allows to write the log in another folder.
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            path_exp = "{}/exp_{}/run_{:02d}".format(base_dir, i, j)
            me.make_exps(reshuffle_mod, path_exp)
            # If one wants to plot again already made results.
            # me.load_results_and_plot(path_exp)
        out_folder_list = ["{}/exp_{}/run_{:02d}".format(base_dir, i, j)
                           for j in range(start_run, end_run)]
        me.load_all_results_and_plot(out_folder_list, type_plot="average")
        me.load_all_results_and_plot(out_folder_list, type_plot="quantile")

def make_final_graph(base_dir=DEFAULT_BASE_DIR,
                     start_run=0, end_run=100):
    """Makes the figure 4 of the publication."""
    plt.style.use('default')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif') # sans-
    plt.rcParams.update({'font.size': 16,
                         'font.serif' : ['Computer Modern Roman']})
    plt.figure(1, figsize=(8, 7))
    pos = {4: 221, 2: 222, 1: 223, 0:224}
    for i, _ in [(4, 10000), (2, 25), (1, 5), (0, 1)]:
        out_folder_list = ["{}/exp_{}/run_{:02d}".format(base_dir, i, j)
                           for j in range(start_run, end_run)]
        res_dict = dict()

        for out_folder in out_folder_list:
            p_learn = json.load(open(
                "{}/dynamics.json".format(out_folder), "rt"))

            # Convert to array to make everything plottable.
            for k in p_learn:
                if k.endswith("AUC"):
                    p_learn[k] = np.array(p_learn[k])
                if k in res_dict:
                    res_dict[k].append(p_learn[k])
                else:
                    res_dict[k] = [p_learn[k]]

        out_folder_plot = "/".join(out_folder_list[0].split("/")[:-1])
        plt.subplot(pos[i])
        me.plot_quantiles(res_dict, out_folder_plot, "quantile",
                          pos=pos[i]%10, saveit=False)
    plt.savefig("cumul_shuttle_exp.pdf")

def make_final_legend():
    """Makes the legend of figure 4 of the publication."""
    fig = plt.figure(figsize=(10, 1))
    me.get_final_graph_legend(fig)
    fig.savefig("cumul_shuttle_leg.pdf")

def main():
    os.environ['MKL_NUM_THREADS'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("start_run",
                        help="Beggining of runs indexes.", type=int)
    parser.add_argument("end_run",
                        help="End of run indexes.", type=int)
    parser.add_argument("base_dir",
                        help="Base directory for the experiments.", type=str)
    args = parser.parse_args()
    make_runs(args.start_run, args.end_run, args.base_dir)

if __name__ == "__main__":
    # Interactive mode, change the parameters.
    # main()

    # Results presented in the paper:
    me.convert_data_to_pickle()
    make_runs(0, 100, "exps")
    make_final_graph()
    make_final_legend()
