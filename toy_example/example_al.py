# %%

import os
import sys
sys.path.append('..')

import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

from active_learning import ActiveLearner
from sampling_strategies import QueryUncertainty

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

results_path = "outputs/al_results"
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

    splice = (np.argmin(np.abs(X - 1.75)), np.argmin(np.abs(X - 2.25)))
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
plt.hist(X, bins=50, density=True)
plt.ylabel("X density")

# %%

# train GP model on whole dataset, just to see what happens

X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(y)

X = X_scaler.transform(X)
y = y_scaler.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

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
plt.plot(
    X_full,
    y_scaler.inverse_transform(y_full_pred.reshape(-1, 1)),
    "b-",
    label="Pred",
    zorder=1,
)
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# %%

# create pool, train, test
n_init = 10

pool_inds, test_inds = train_test_split(
    np.arange(len(X)), test_size=0.2, random_state=42
)
init_inds = np.random.choice(pool_inds, size=n_init, replace=False)
pool_inds = np.setdiff1d(pool_inds, init_inds, assume_unique=True)

print("Pool size:", pool_inds.shape)
print("Train size:", init_inds.shape)
print("Test size:", test_inds.shape)

X_pool = X[pool_inds]
y_pool = y[pool_inds]

X_init = X[init_inds]
y_init = y[init_inds]

X_test = X[test_inds]
y_test = y[test_inds]

# %%

############### create active learners and run ###############

n_iter = 250


# define the probe function, f(X) --> y
def probe_X(X):
    X = np.reshape(X, (-1, X.shape[-1]))

    inds = np.zeros(len(X), dtype=int)
    for i, arr in enumerate(X):
        inds[i] = np.where(np.all(X_pool == arr, axis=1))[0][0]

    return y_pool[inds]


# define the query strategy
query_strategy = (
    QueryUncertainty()
)  # can play around with what sampling strategy you want to use

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
    batch_size=1,  # setting batch size to 1 to show traditional AL sampling
    random_state=42,
)

# run the active learner
active_learner.run(
    n_iter=n_iter,
    save_path=f"{results_path}/model-{query_strategy.NAME}.pkl",
)

# %%

with open(
    f"{results_path}/model-{query_strategy.NAME}.pkl",
    "rb",
) as f:
    results = pickle.load(f)

# %%

# plot error vs iterations

plt.title(f"Test Set Performance\niterations: {n_iter}")
plt.plot(results["n_samples"], results["err_test"], label=query_strategy.NAME)
plt.xlabel("Number of Samples")
plt.ylabel("RMSE")
plt.legend()
plt.savefig(f"{results_path}/results_{query_strategy.NAME}.png")

# %%

# plot sampled points

inds_sampled = results["inds_queried"]

plt.plot(X, y, zorder=0)
plt.scatter(
    X_pool[inds_sampled],
    y_pool[inds_sampled],
    s=5,
    c=np.arange(len(inds_sampled)),
    cmap="hot",
    zorder=5,
)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Sampled Points")
plt.colorbar(label="Sampling Order")

# %%

# gif of function prediction vs iterations

print(results.keys())
print(len(results["models"]))

model_full_predictions = []
points_sampled_full = results["inds_queried"]

for model in results["models"]:
    model_full_predictions.append(model.predict(X_scaler.transform(X_full)))

model_full_predictions = np.array(model_full_predictions)
print(model_full_predictions.shape)

# %%

def animate(frame):
    y_full_pred = y_scaler.inverse_transform(
        model_full_predictions[frame].reshape(-1, 1)
    )

    plt.clf()
    plt.plot(X_full, y_full, "k-", label="True", zorder=0)
    plt.plot(X_full, y_full_pred, "b-", label="Pred", zorder=1)
    plt.scatter(
        X_scaler.inverse_transform(
            X_pool[points_sampled_full[: frame + 1]].reshape(-1, 1)
        ),
        [-0.2] * (frame + 1),
        c='tab:orange',
        alpha=0.2,
        edgecolor="none",
        label="Sampled X",
    )
    plt.scatter(
        X_scaler.inverse_transform(X_init.reshape(-1, 1)),
        [-0.2] * len(X_init),
        c='tab:blue',
        edgecolor="none",
        label="Initial X",
    )
    plt.xlabel("X")
    plt.ylabel("y")
    plt.xlim(0, 3)
    plt.ylim(-0.3, 2.1)
    plt.legend()

    plt.text(0.1, -0.1, f'{frame + 1} points sampled', fontsize=10)

    return plt

# Create the animation
anim = FuncAnimation(
    plt.figure(), animate, frames=len(model_full_predictions), interval=1, repeat=False
)
anim.save(f"{results_path}/results_{query_strategy.NAME}.gif")
plt.show()

# %%
