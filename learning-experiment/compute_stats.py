"""
    Code for:
    Trade-offs in Large Scale Distributed Tuplewise Estimation and Learning
    Author: Robin Vogel
"""
import numpy as np

# ---------- Definitions of the estimators --------

def Un(X, Z, kernel="prod"):
    """ Computes Un, full two-sample U-statistic."""
    X_col = X.reshape((-1, 1))
    Z_row = Z.reshape((1, -1))
    assert kernel in ["prod", "gini", "AUC"]
    if kernel == "prod":
        return (X_col.dot(Z_row)).mean()
    if kernel == "gini":
        return np.mean(np.abs(X_col - Z_row))
    return np.mean((X_col - Z_row > 0).astype(int))


def UB_indices(X, Z, ind_X, ind_Z, kernel):
    X_select = X[ind_X]
    Z_select = Z[ind_Z]
    assert kernel in ["prod", "gini", "AUC"]
    if kernel == "prod":
        return np.mean(X_select*Z_select)
    if kernel == "gini":
        return np.mean(np.abs(X_select-Z_select))
    return np.mean((X_select-Z_select > 0).astype(int))

def UB_pairs(X, Z, indices, kernel):
    """Computes incomplete two-sample U-statistic for given pairs."""
    return UB_indices(X, Z, np.array([ind[0] for ind in indices]),
                      np.array([ind[1] for ind in indices]), kernel)

def UB(X, Z, B, kernel="prod"):
    """Computes incomplete two-sample U-statistic."""
    n_X = X.shape[0]
    n_Z = Z.shape[0]
    return UB_indices(X, Z, np.random.randint(0, n_X, B),
                      np.random.randint(0, n_Z, B), kernel)

def UN_split(X_s, Z_s, f_block):
    """Computes UN for all of the blocks in X_s, Z_s."""
    return np.mean([f_block(X, Z) for X, Z in zip(X_s, Z_s)], axis=0)

def SWR_divide(X, Z, N):
    """Divides the sample with sampling with replacement."""
    n_X = X.shape[0]
    n_Z = Z.shape[0]
    X_s = [X[np.random.randint(0, n_X, int(n_X/N))] for _ in range(N)]
    Z_s = [Z[np.random.randint(0, n_Z, int(n_Z/N))] for _ in range(N)]
    return X_s, Z_s

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
                assert sampling_type == "SWOR"
                vals.append(0)
            else:
                vals.append(f_block(X_rem[:k], Z_rem[:(tau - k)]))

            X_rem = X_rem[k:]
            Z_rem = Z_rem[(tau - k):]

        elif sampling_type == "prop-SWR":
            vals.append(f_block(X_rem[np.random.randint(0, n_X, int(n_X/N))],
                                Z_rem[np.random.randint(0, n_Z, int(n_Z/N))]))
    return np.mean(vals)


def UnN(X, Z, N, sampling_type, kernel="prod"):
    """Computes block-wise complete U-statistic."""

    def fun_block(x, z):
        return Un(x, z, kernel=kernel)

    return UN(X, Z, N, fun_block, sampling_type=sampling_type)


def UnNB(X, Z, N, B, sampling_type, kernel="prod"):
    """Computes block-wise incomplete U-statistic."""

    def fun_block(x, z):
        return UB(x, z, B, kernel=kernel)

    return UN(X, Z, N, fun_block, sampling_type=sampling_type)


def UnNT(X, Z, N, T, sampling_type, kernel="prod"):
    """Computes reshuffled block-wise complete U-statistic."""
    return np.mean([UnN(X, Z, N, sampling_type=sampling_type, kernel=kernel)
                    for _ in range(T)])


def UnNBT(X, Z, N, B, T, sampling_type, kernel="prod"):
    """Computes reshuffled block-wise incomplete U-statistic."""
    return np.mean([UnNB(X, Z, N, B, sampling_type=sampling_type,
                         kernel=kernel)
                    for _ in range(T)])

# -------- End of the definition of the estimators -------

# ---------- Gradient descent functions ----------

def conv_AUC(margin):
    def res_function(X, Z):
        """Computes the convexification of the 1-AUC that we minimize."""
        X_col = X.reshape((-1, 1))
        Z_row = Z.reshape((1, -1))
        return np.maximum(Z_row - X_col + margin, 0).mean()
    return res_function

def conv_AUC_deter_pairs(margin):
    """Returns function that computes the convex loss on incomplete U-stat."""
    def res(X, Z, indices):
        """Computes the convexification of the 1-AUC that we minimize."""
        X_select = X[np.array([ind[0] for ind in indices])]
        Z_select = Z[np.array([ind[1] for ind in indices])]
        return np.maximum(Z_select - X_select + margin, 0).mean()
    return res

def grad_inc_block(w, B, margin):
    """Returns a function that computes the gradient on incomplete U-stat."""
    def res(X, Z):
        """
            Returns:
            1/B sum_{i,j in D_B} I{w^T(Z_j - X_i + margin > 0)}(Z_j - X_i)
        """
        n_X = X.shape[0]
        n_Z = Z.shape[0]
        X_sel = X[np.random.randint(0, n_X, B)]
        Z_sel = Z[np.random.randint(0, n_Z, B)]
        diff = Z_sel - X_sel
        S_diff = diff.dot(w) + margin
        filt = (S_diff > 0).ravel()
        return (diff[filt].sum(axis=0)/B).reshape([-1, 1])
    # res returns a vector
    return res

# ---------- End gradient descent functions ----------
