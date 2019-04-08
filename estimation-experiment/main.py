"""
    Code for:
    Trade-offs in Large Scale Distributed Tuplewise Estimation and Learning
    Author: Robin Vogel
"""
import time
import numpy as np
import matplotlib.pyplot as plt

def p(e):
    return e

def q(e):
    return 1-e

def sigma_1(e):
    return (p(e)**2)*q(e)*(1-q(e))

def sigma_2(e):
    return ((1-q(e))**2)*p(e)*(1-p(e))

def sigma_0(e):
    return p(e)*q(e)*(1-p(e))*(1-q(e))


def Mean_Un(e):
    return q(e) + (1-q(e))*(1-p(e))

def Un(X, Z):
    """ Computes Un, full two-sample U-statistic."""
    return (X.reshape((-1, 1)) > Z.reshape((1, -1))).mean()

def UN(X, Z, N, f_block, sampling_type="SWOR"):
    """
        Computes complete or incomplete (depending on f_block)
        two-sample U-statistic on each worker and averages them.
        Cuts the dataset X,Z in N splits.
        sampling_type can be SWOR, prop-SWOR or prop-SWR
    """
    vals = list()
    X_rem = X
    Z_rem = Z
    np.random.shuffle(X_rem)
    np.random.shuffle(Z_rem)
    n_X = X_rem.shape[0]
    n_Z = Z_rem.shape[0]
    tau = int((n_X + n_Z) / N)
    for _ in range(N):
        if sampling_type != "prop-SWR":  # If sampling without replacement
            if sampling_type.startswith("prop"):
                k = int(n_X/N)  # for prop-SWOR
            else:  # if it is SWOR or SWOR-nobias
                n_X = X_rem.shape[0]
                n_Z = Z_rem.shape[0]
                k = np.random.binomial(tau, n_X / (n_X + n_Z))

            if k in (0, tau):
                if sampling_type == "SWOR":  # if biais
                    vals.append(0)
            else:
                vals.append(f_block(X_rem[:k], Z_rem[:(tau - k)]))

            X_rem = X_rem[k:]
            Z_rem = Z_rem[(tau - k):]

        elif sampling_type == "prop-SWR":
            vals.append(f_block(X_rem[np.random.randint(0, n_X, int(n_X/N))],
                                Z_rem[np.random.randint(0, n_Z, int(n_Z/N))]))
    return np.mean(vals)


def UnN(X, Z, N, sampling_type):
    """Computes block-wise complete U-statistic."""
    return UN(X, Z, N, Un, sampling_type=sampling_type)

def UnNT(X, Z, N, T, sampling_type):
    """Computes reshuffled block-wise complete U-statistic."""
    return np.mean([UnN(X, Z, N, sampling_type=sampling_type)
                    for _ in range(T)])


def main():
    plt.style.use('default')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 16,
                         'font.serif' : ['Computer Modern Roman']})

    n = 5000
    m = 50
    N = 10
    T = 4
    n_tries = 5000
    sampling_type = "prop-SWOR"
    epsilons = [0.00016, 0.0008, 0.004, 0.02, 0.1, 0.5]

    def gen_X(epsilon):
        return 2*np.random.binomial(1, 1-epsilon, n)

    def gen_Z(epsilon):
        return 2*np.random.binomial(1, epsilon, m)-1

    def Var_Un(e):
        return sigma_1(e)/n + sigma_2(e)/m + sigma_0(e)/(n*m)

    def get_values_epsilon(epsilon):
        val_Un = [Un(gen_X(epsilon), gen_Z(epsilon)) for _ in range(n_tries)]
        val_UnN = [UnN(gen_X(epsilon), gen_Z(epsilon), N, sampling_type)
                   for _ in range(n_tries)]
        val_UnNT = [UnNT(gen_X(epsilon), gen_Z(epsilon), N, T, sampling_type)
                    for _ in range(n_tries)]
        return val_Un, val_UnN, val_UnNT

    old_time = time.time()
    epsilon_vals = [get_values_epsilon(epsilon) for epsilon in epsilons]
    print(time.time()-old_time)

    plt.figure(figsize=(10, 2.5))
    renorm = True
    y_min = 100000
    y_max = -100000

    for epsilon, vals in zip(epsilons, epsilon_vals):
        val_Un, val_UnN, val_UnNT = vals
        var_Un = np.var(val_Un)
        var_UnN = np.var(val_UnN)
        var_UnNT = np.var(val_UnNT)
        if renorm:
            var_Un = var_Un/Var_Un(epsilon)
            var_UnN = var_UnN/Var_Un(epsilon)
            var_UnNT = var_UnNT/Var_Un(epsilon)
        y_min = min(var_Un, y_min)
        y_max = max(var_UnN, y_max)
        plt.scatter(epsilon, var_Un, color="black", label="Un")
        plt.scatter(epsilon, var_UnN, color="red", label="UnN")
        plt.scatter(epsilon, var_UnNT, color="blue",
                    label="$U_{\\mathbf{n},N,T}$")

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(True, which="both", ls="--")


    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Rel. var.")
    plt.xlim([np.min(epsilons)*(0.80), np.max(epsilons)*1.2])
    plt.ylim([y_min-0.1, y_max+0.2])

    plt.xscale("log")
    # plt.yscale("log")
    # Following line to add space in legend if required:
    # matplotlib.patches.Rectangle((0,0), 1, 1, fill=False,
    #                              edgecolor='none', visible=False)]
    leg_elems = [
        plt.scatter([0], [0], color="black", label="$U_{\\mathbf{n}}$"),
        plt.scatter([0], [0], color="red", label="$U_{\\mathbf{n},N}$"),
        plt.scatter([0], [0], color="blue", label="$U_{\\mathbf{n},N,T}$")]
    plt.legend(handles=leg_elems, ncol=1)

    plt.tight_layout()
    plt.title("")
    # plt.show()
    plt.savefig("est_exp.pdf", dpi=400, format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()
