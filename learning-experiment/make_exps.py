"""
    Code for:
    Trade-offs in Large Scale Distributed Tuplewise Estimation and Learning
    Author: Robin Vogel
"""
import os
import shutil
import json
import logging
import pickle
from scipy.io import loadmat
import numpy as np
# Avoid trouble when generating pdf's on a distant server.
# import matplotlib
# matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import compute_stats as cs

SEED_SHUFFLE = 42

# Bounds for the axis in the plots
LB_CONV_COST = 0.01
LB_AUC = 0.005
UB_CONV_COST = 0.1
UB_AUC = 0.02
QUANTILE_PLOT_FILTER = 750
# Limit on the number of iterations represented by the plot

TYPE_TRAIN_MONITOR = "FIXED_PAIRS" # "SAME_AS_BATCH"
# FIXED_PAIRS: monitors the training loss on a fixed set of instances.
# SAME_AS_BATCH: monitors the training loss on training batches.

# Parameters of the monitoring pairs if fixed pairs.
SEED_TRAIN_MONITOR = 54
SIZE_TRAIN_MONITOR = 450000

# Others
PROP_TEST = 0.2
DEFAULT_ITE_NUMBER = 5000 # 5000

# ------------------------ Fundations functions -------------------------

def convert_data_to_pickle():
    """Loads the data from mat format and exports it to the pickle format."""
    data = loadmat("shuttle.mat")
    data["y"] = 2*data["y"].ravel().astype(int) - 1
    pickle.dump(data, open("shuttle.pickle", "wb"))

def load_preprocess_data():
    """Loads and preprocess the data."""
    data = pickle.load(open("shuttle.pickle", "rb"))
    X = data["X"]
    y = data["y"]
    Z_tot = X[y == +1] # Minority class is the anomaly class, i.e. y = +1
    X_tot = X[y == -1]

    np.random.seed(SEED_SHUFFLE)
    ind_X_test = np.random.choice(X_tot.shape[0],
                                  size=int(PROP_TEST*X_tot.shape[0]),
                                  replace=False)
    ind_Z_test = np.random.choice(Z_tot.shape[0],
                                  size=int(PROP_TEST*Z_tot.shape[0]),
                                  replace=False)

    np.random.seed()
    Z_train, X_train = Z_tot[~ind_Z_test], X_tot[~ind_X_test]
    Z_test, X_test = Z_tot[ind_Z_test], X_tot[ind_X_test]
    train_tot = np.vstack([X_train, Z_train])

    # Normalize the instances
    train_mean = train_tot.mean(axis=0)
    train_std = train_tot.std(axis=0)

    if np.min(train_std) == 0:
        raise ValueError("One of the columns in the data has constant var.")

    X_train = (X_train - train_mean)/train_std
    Z_train = (Z_train - train_mean)/train_std
    X_test = (X_test - train_mean)/train_std
    Z_test = (Z_test - train_mean)/train_std

    # Add a constant
    def add_constant(a):
        return np.hstack([a, np.ones((a.shape[0], 1))])

    X_train = add_constant(X_train)
    Z_train = add_constant(Z_train)
    X_test = add_constant(X_test)
    Z_test = add_constant(Z_test)

    return Z_train, X_train, Z_test, X_test


def learning_process(X, Z, p_learn, optim_type="momentum"):
    """Learning process for our experiments."""
    n_X, n_Z = X.shape[0], Z.shape[0]
    N = p_learn["N"]
    B = p_learn["B"]
    learning_rate = p_learn["learning_rate"]
    margin = p_learn["margin"]
    w = p_learn["w_init"]

    to_log = ["{} : {}".format(k, v) for k, v in p_learn.items()
              if not k.startswith(("train_", "test_", "w"))]

    i = 0
    while i < len(to_log)/4:
        logging.info("%s", " / ".join(to_log[(i*4):(i*4+4)]))
        i += 1

    logging.info("#X: %d / #Z: %d ", n_X, n_Z)
    logging.info("#X/N: %d / #Z/N: %d ", n_X/N, n_Z/N)
    logging.info("pairs_per_clust: %d ", (n_X/N)*(n_Z/N))
    logging.info("#eval_pairs_before_reshuffle: %d ",
                 B*p_learn["reshuffle_mod"])

    X_s, Z_s = cs.SWR_divide(X, Z, N)

    delta_w = 0
    for i in range(0, p_learn["n_it"]):
        if i % p_learn["reshuffle_mod"] == 0:
            # logging.info("it %5d: reshuffling", i)
            X_s, Z_s = cs.SWR_divide(X, Z, N)

        if i % p_learn["eval_mod"] == 0:
            evaluation_step(i, X_s, Z_s, w, p_learn)

        gradient = (cs.UN_split(X_s, Z_s, cs.grad_inc_block(w, B, margin))
                    + p_learn["reg"]*w)

        assert optim_type in ["SGD", "momentum"]

        if optim_type == "SGD":
            delta_w = learning_rate*gradient
        if optim_type == "momentum":
            momentum = 0.9
            delta_w = momentum*delta_w + learning_rate*gradient

        w = w - delta_w

def evaluation_step(i, X_s, Z_s, w, p_learn):
    """
        Modify the value of p_learn to add to the evaluation.
        Monitored values, added in p_learn:
        * br_AUC: block real AUC, on the training data,
        * bc_AUC: block convexified AUC, on the training data,
        * tr_AUC: real AUC, on the testing data,
        * tc_AUC: convexified AUC, on the testing data,
    """
    margin = p_learn["margin"]
    logging.debug("Step %d: Begin evaluation", i)
    if TYPE_TRAIN_MONITOR == "SAME_AS_BATCH":
        sc_X = [x.dot(w) for x in X_s]
        sc_Z = [z.dot(w) for z in Z_s]
        bc_AUC = (cs.UN_split(sc_X, sc_Z, cs.conv_AUC(margin))
                  + p_learn["reg"]*(np.linalg.norm(w)**2)/2)
        br_AUC = cs.UN_split(sc_X, sc_Z,
                              lambda x, z: cs.Un(x, z, kernel="AUC"))

    elif TYPE_TRAIN_MONITOR == "FIXED_PAIRS":
        sc_X, sc_Z = p_learn["train_X"].dot(w), p_learn["train_Z"].dot(w)
        bc_AUC = (cs.conv_AUC_deter_pairs(margin)(
            sc_X, sc_Z, p_learn["train_mon_pairs"])
                  + p_learn["reg"]*(np.linalg.norm(w)**2)/2)
        br_AUC = cs.UB_pairs(
            sc_X, sc_Z, p_learn["train_mon_pairs"], kernel="AUC")

    sc_X_test = p_learn["test_X"].dot(w)
    sc_Z_test = p_learn["test_Z"].dot(w)
    tc_AUC = (cs.conv_AUC(margin)(sc_X_test, sc_Z_test)
              + p_learn["reg"]*(np.linalg.norm(w)**2)/2)
    tr_AUC = cs.Un(sc_X_test, sc_Z_test, kernel="AUC")

    s_log = ("it %5d: bc_AUC = %.4f | br_AUC = %.4f "
             + "| tc_AUC = %5.4f | tr_AUC = %5.4f")
    logging.info(s_log, i, bc_AUC, br_AUC, tc_AUC, tr_AUC)

    # Elements to add to the dictionary:
    elems = [("iter", i), ("norm_w", np.linalg.norm(w)),
             ("bc_AUC", bc_AUC), ("br_AUC", br_AUC),
             ("tr_AUC", tr_AUC), ("tc_AUC", tc_AUC)]
    for k, v in elems:
        if k in p_learn:
            p_learn[k].append(v)
        else:
            p_learn[k] = [v]

    logging.debug("Step %d: End evaluation", i)

def make_exps(reshuffle_mod, out_folder="exps/test", p_learn=None):
    """Make the experiments for the desired parameters."""
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        shutil.copy(os.path.basename(__file__),
                    "{}/executed_script.py".format(out_folder))

    Z_train, X_train, Z_test, X_test = load_preprocess_data()
    n_feats = Z_train.shape[1]

    # Set parameters:
    logging.basicConfig(filename='{}/learning_process.log'.format(out_folder),
                        format='%(asctime)s - %(message)s',
                        # - %(levelname)s
                        level=logging.INFO, datefmt='%m/%d/%y %I:%M:%S %p',
                        filemode="w")

    if p_learn is None:
        p_learn = {"n_it": DEFAULT_ITE_NUMBER, "margin": 1, "N": 100,
                   "B": 100, "reshuffle_mod": reshuffle_mod, "reg": 0.05,
                   "learning_rate": 0.01, "eval_mod": 25,
                   "w_init": np.random.normal(0, 1, (n_feats, 1)),
                   "test_X": X_test, "test_Z": Z_test}

    if TYPE_TRAIN_MONITOR == "FIXED_PAIRS":
        np.random.seed(SEED_TRAIN_MONITOR)
        p_learn["train_mon_pairs"] = zip(list(
            np.random.randint(0, X_train.shape[0], SIZE_TRAIN_MONITOR)), list(
                np.random.randint(0, Z_train.shape[0], SIZE_TRAIN_MONITOR)))
        p_learn["train_mon_pairs"] = list(p_learn["train_mon_pairs"])
        p_learn["train_X"] = X_train
        p_learn["train_Z"] = Z_train
        np.random.seed()

    print("Started optimization")
    learning_process(X_train, Z_train, p_learn)
    print("Finished optimization")

    # We get rid of the testing numpy arrays, as well as the init.
    keys_to_delete = list(filter(
        lambda x: x.startswith(("train_", "test_", "w_")), p_learn.keys()))

    for x in keys_to_delete:
        p_learn.pop(x)
    json.dump(p_learn, open("{}/dynamics.json".format(out_folder), "wt"))

    # Convert to array to make everything plottable.
    for k in p_learn:
        if k.endswith("AUC"):
            p_learn[k] = np.array(p_learn[k])

    plot_results(p_learn, out_folder)

def plot_results(p_learn, out_folder):
    """Plots the results."""
    # n_it = p_learn["n_it"]
    # N = p_learn["N"]
    # B = p_learn["B"]
    # learning_rate = p_learn["learning_rate"]
    # w_init = p_learn["w_init"]
    # margin = p_learn["margin"]
    reshuffle_mod = p_learn["reshuffle_mod"]
    # eval_mod = p_learn["eval_mod"]

    # Plot the result
    plt.figure()
    plt.plot(p_learn["iter"], p_learn["tc_AUC"],
             label="test convAUC", color="red")
    plt.plot(p_learn["iter"], p_learn["bc_AUC"],
             label="train convAUC", color="red", linestyle="--")
    plt.grid()
    plt.legend(loc="lower left")
    plt.ylabel("Convex cost", color="red")
    plt.ylim([LB_CONV_COST, UB_CONV_COST])
    plt.twinx()
    plt.plot(p_learn["iter"], 1 - p_learn["tr_AUC"],
             label="test 1-AUC", color="blue")
    plt.plot(p_learn["iter"], 1 - p_learn["br_AUC"],
             label="train 1-AUC", color="blue", linestyle="--")
    plt.ylabel("1-AUC", color="blue")
    plt.ylim([LB_AUC, UB_AUC])
    plt.legend(loc="upper right")
    plt.title("n_r = {}".format(reshuffle_mod))
    plt.tight_layout()
    plt.savefig("{}/dynamics.pdf".format(out_folder), format="pdf")
    plt.close()


def load_all_results_and_plot(out_folder_list, type_plot="average"):
    """Loads the results for lots of runs and plots them."""
    res_dict = dict()

    for out_folder in out_folder_list:
        p_learn = json.load(open("{}/dynamics.json".format(out_folder), "rt"))

        # Convert to array to make everything plottable.
        for k in p_learn:
            if k.endswith("AUC"):
                p_learn[k] = np.array(p_learn[k])
            if k in res_dict:
                res_dict[k].append(p_learn[k])
            else:
                res_dict[k] = [p_learn[k]]

    assert type_plot in ["average", "quantile"]

    out_folder_plot = "/".join(out_folder_list[0].split("/")[:-1])
    if type_plot == "average":
        for k in res_dict:
            res_dict[k] = np.mean(res_dict[k], axis=0)

        p_learn["reshuffle_mod"] = np.mean(p_learn["reshuffle_mod"])
        plot_results(p_learn, out_folder_plot)

    elif type_plot == "quantile":
        plot_quantiles(res_dict, out_folder_plot, type_plot)

def load_results_and_plot(out_folder):
    """Loads the results and plots them."""
    p_learn = json.load(open("{}/dynamics.json".format(out_folder), "rt"))

    # Convert to array to make everything plottable.
    for k in p_learn:
        if k.endswith("AUC"):
            p_learn[k] = np.array(p_learn[k])

    plot_results(p_learn, out_folder)


def plot_quantiles(p_learn, out_folder, out_name, pos=1, saveit=True):
    """Plots the results."""
    reshuffle_mod = p_learn["reshuffle_mod"]

    # The entries of the dictionary contain a matrix n_runs, n_values.
    alphas = [0.05]
    def quantile(X, q, axis=0):
        """np.quantile only exists on numpy 1.15 and higher."""
        assert axis == 0
        X = np.array(X)
        return np.sort(X, axis=0)[int(X.shape[0]*q), :]

    filt = np.array(p_learn["iter"][0]) <= QUANTILE_PLOT_FILTER
    def filt_elem(a):
        return np.array(a)[filt]
    # Beginning of plotting operations:
    if saveit:
        plt.figure(figsize=(3, 4))
    for alpha in alphas:
        plt.fill_between(
            filt_elem(p_learn["iter"][0]),
            filt_elem(quantile(p_learn["tc_AUC"], (1-alpha/2), axis=0)),
            filt_elem(quantile(p_learn["tc_AUC"], alpha/2, axis=0)),
            color="red", label="95% CI", alpha=0.25)

    plt.plot(filt_elem(p_learn["iter"][0]),
             filt_elem(np.median(p_learn["tc_AUC"], axis=0)),
             label="test", color="red")

    plt.plot(filt_elem(p_learn["iter"][0]),
             filt_elem(np.median(p_learn["bc_AUC"], axis=0)),
             label="train", color="red", linestyle="--")
    plt.grid()

    if not saveit:
        if pos%2 == 0: # Checks whether we need a label for y axis.
            # plt.gca().set_yticks([])
            plt.gca().yaxis.set_major_formatter(NullFormatter())
        else:
            plt.ylabel("Loss", color="red")
            plt.ticklabel_format(style='sci', axis="y", scilimits=(0, 0))

        if pos <= 2: # Checks whether we need a label for x axis.
            # plt.gca().set_xticks([])
            plt.gca().xaxis.set_major_formatter(NullFormatter())
        if pos > 2:
            plt.xlabel("iter")

    plt.ylim([LB_CONV_COST, UB_CONV_COST])
    plt.twinx()
    for alpha in alphas:
        plt.fill_between(filt_elem(p_learn["iter"][0]),
                         filt_elem(quantile(1 - np.array(p_learn["tr_AUC"]),
                                            (1-alpha/2), axis=0)),
                         filt_elem(quantile(1 - np.array(p_learn["tr_AUC"]),
                                            alpha/2, axis=0)),
                         color="blue", label="95% CI", alpha=0.25)
    plt.plot(filt_elem(p_learn["iter"][0]),
             (1 - filt_elem(np.median(p_learn["tr_AUC"], axis=0))),
             label="test", color="blue")
    plt.plot(filt_elem(p_learn["iter"][0]),
             (1 - filt_elem(np.median(p_learn["br_AUC"], axis=0))),
             label="train", color="blue", linestyle="--")

    if not saveit:
        if pos%2 == 0: # Checks whether we need a label for y axis.
            plt.ylabel("1-AUC", color="blue")
            plt.ticklabel_format(style='sci', axis="y", scilimits=(0, 0))
        else:
            # plt.gca().set_yticks([])
            plt.gca().yaxis.set_major_formatter(NullFormatter())
        plt.ylim([LB_AUC, UB_AUC])

    if saveit:
        plt.legend(loc="upper right")
    if int(np.mean(reshuffle_mod)) == 10000:
        plt.title("$n_r = 10,000$")
    else:
        plt.title("$n_r = {}$".format(int(np.mean(reshuffle_mod))))

    plt.tight_layout()
    if saveit:
        plt.savefig("{}/{}.pdf".format(out_folder, out_name), format="pdf")
        plt.close()

def get_final_graph_legend(fig):
    """Builds the legend of figure 4 of the publication."""
    plt.style.use('default')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif') # sans-
    plt.rcParams.update({'font.size': 16,
                         'font.serif' : ['Computer Modern Roman']})
    legend_elements = list()
    # Filler if the elements are ill-disposed:
    # matplotlib.patches.Rectangle((0,0), 1, 1, fill=False,
    # edgecolor='none', visible=False)
    legend_elements.append(plt.fill_between([0], [0], [0], color="red",
                                            label=r"95\% CI", alpha=0.25))
    legend_elements.append(plt.Line2D(
        [0], [0], label="test", color="red"))
    legend_elements.append(plt.Line2D(
        [0], [0], label="train", color="red", linestyle="--"))
    fig.legend(handles=legend_elements, ncol=4,
               loc="center left", title="Loss")

    legend_elements = list()
    legend_elements.append(plt.fill_between([0], [0], [0], color="blue",
                                            label=r"95\% CI", alpha=0.25))

    legend_elements.append(plt.Line2D(
        [0], [0], label="test", color="blue"))
    legend_elements.append(plt.Line2D(
        [0], [0], label="train", color="blue", linestyle="--"))
    fig.legend(handles=legend_elements, ncol=4,
               loc="center right", title="1-AUC")
    plt.xticks([])
    plt.yticks([])
    plt.gca().axis('off')

if __name__ == "__main__":
    pass
