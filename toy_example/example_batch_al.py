# %%

import os
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

from BatchActiveLearning.active_learning import ActiveLearner
from BatchActiveLearning.sampling_strategies import (
    QueryRandom,
    QueryUncertainty,
    QueryDiversity,
    QueryGreedyDiversity,
    QueryRankedBatchMode,
    QueryDiverseMiniBatch,
    QueryClusterMargin,
)

import matplotlib.pyplot as plt

results_path = 'outputs/batch_al_results'
os.makedirs(results_path, exist_ok=True)

# %%


def norm(x, x0, sigma):
    return np.exp(-0.5 * (x - x0) ** 2 / sigma**2)


def sigmoid(x, x0, alpha):
    return 1.0 / (1.0 + np.exp(-(x - x0) / alpha))

# non-uniform input space
X_full = np.sort(
    np.concatenate(
        (
            np.random.uniform(0, 3, 6000),
            np.random.normal(1, 0.1, 3000),
            np.random.normal(2.75, 0.5, 3000),
            np.random.normal(0.0, 0.1, 3000),
        )
    )
)
X_full = X_full[X_full >= 0]
X_full = X_full[X_full <= 3]

# non-linear output space
def y_func(X):
    y = (
        0.1 * np.sin(norm(X, 0.2, 0.05))
        + 0.25 * norm(X, 0.6, 0.05)
        + 0.3 * norm(X, 0.5, 0.08)
        + 1 * norm(X, 1.5, 0.03)
        + 0.6 * norm(X, 1.6, 0.07)
        + 0.8 * norm(X, 2.5, 0.3)
        + np.sqrt(norm(X, 0.8, 0.06))
        + np.sqrt(norm(X, 0.5, 0.6))
        + 0.1 * (1 - sigmoid(X, 0.45, 0.15))
        + 0.2 * np.sin(5 * X) * np.cos(10 * X)
        + 0.15 * np.sin(20 * X) * np.cos(30 * X)
        + 0.05 * np.sin(50 * X) * np.cos(70 * X)
    )

    splice = (np.argmin(np.abs(X-1.75)), np.argmin(np.abs(X-2.25)))
    seg = X[splice[0] : splice[1]]
    y[splice[0] : splice[1]] += 0.4 * np.sin(10 * seg) * np.cos(120 * seg)

    return y

X_full = np.reshape(X_full, (-1, 1))
y_full = np.reshape(y_func(X_full), (-1, 1))

inds = sorted(np.random.choice(X_full.shape[0], size=1000, replace=False))
X = X_full[inds]
y = y_full[inds]

print(X.shape)
print(y.shape)

plt.subplot(211)
plt.plot(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.subplot(212)
plt.hist(X, bins=50)
plt.ylabel("X frequency")

# %%

# training a GP model on the whole dataset

X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaler.transform(X), y_scaler.transform(y), test_size=0.33
)

kernel = Matern()
model = GPR(kernel=kernel, n_restarts_optimizer=10)
model = model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_full_pred = model.predict(X_scaler.transform(X_full))

plt.scatter(y_train, y_train_pred, label="Train")
plt.scatter(y_test, y_test_pred, label="Test")
plt.xlabel("True")
plt.ylabel("Pred")
plt.legend()
plt.show()

print(f"RMSE:\t {np.sqrt(np.mean((y_test - y_test_pred) ** 2)):0.3f}")

plt.figure(figsize=(8, 4))
plt.plot(X_full, y_full, "k-", label="True", zorder=0)
plt.plot(X_full, y_scaler.inverse_transform(y_full_pred.reshape(-1, 1)), "b-", label="Pred", zorder=1)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# %%

# create pool, train, test for AL

n_init = 10

pool_inds, test_inds = train_test_split(
    np.arange(len(X)), test_size=0.2, random_state=42
)
init_inds = np.random.choice(pool_inds, size=n_init, replace=False)
pool_inds = np.setdiff1d(pool_inds, init_inds, assume_unique=True)

print('Pool size:', pool_inds.shape)
print('Train size:', init_inds.shape)
print('Test size:', test_inds.shape)

X_pool = X[pool_inds]
y_pool = y[pool_inds]

X_init = X[init_inds]
y_init = y[init_inds]

X_test = X[test_inds]
y_test = y[test_inds]

# %%

############### create active learners and run ###############

batch_size = 25
n_iter = 10

# define the probe function, f(X) --> y
def probe_X(X):
    X = np.reshape(X, (-1, X.shape[-1]))

    inds = np.zeros(len(X), dtype=int)
    for i, arr in enumerate(X):
        inds[i] = np.where(np.all(X_pool == arr, axis=1))[0][0]

    return y_pool[inds]


# define the query strategy
query_strategies = [
    QueryRandom(),
    QueryUncertainty(),
    QueryDiversity(),
    QueryGreedyDiversity(),
    QueryRankedBatchMode(expected_max_samples=batch_size * n_iter),
    QueryDiverseMiniBatch(),
    QueryClusterMargin(),
]

for query_strategy in query_strategies:
    # define the model
    kernel = Matern()
    model = GPR(kernel=kernel, n_restarts_optimizer=10, random_state=42)

    # define the active learner
    active_learner = ActiveLearner(
        Xy_pool=(X_pool, None),
        Xy_init=(X_init, y_init),
        Xy_test=(X_test, y_test),
        model=model,
        model_kwargs={"return_std": True},
        query_strategy=query_strategy,
        probe_function=probe_X,
        scaler=StandardScaler(),
        batch_size=batch_size,
        random_state=42,
    )

    al_results_path = f'{results_path}/{query_strategy.NAME}'
    os.makedirs(al_results_path, exist_ok=True)

    # run the active learner
    active_learner.run(
        n_iter=n_iter,
        save_percent=0.2,
        save_path=f"{al_results_path}/model-{n_iter}-{batch_size}.pkl",
    )

# %%

plt.title(f"Test Set Performance\nbatch size: {batch_size}, iterations: {n_iter}")

for query_strategy in query_strategies:
    with open(
        f"{results_path}/{query_strategy.NAME}/model-{n_iter}-{batch_size}.pkl",
        "rb",
    ) as f:
        results = pickle.load(f)
        plt.plot(results["n_samples"], results["err_test"], label=query_strategy.NAME)

plt.xlabel("Number of Samples")
plt.ylabel("RMSE")
plt.legend()
plt.savefig(f"{results_path}/_al_error_trajecs-{n_iter}-{batch_size}.png")

# %%
