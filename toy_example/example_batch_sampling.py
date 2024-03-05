# %%

import os
import numpy as np
import matplotlib.pyplot as plt

from BatchActiveLearning.sampling_strategies import (
    QueryStrategy,
    QueryRandom,
    QueryUncertainty,
    QueryDiversity,
    QueryGreedyDiversity,
    QueryRankedBatchMode,
    QueryDiverseMiniBatch,
    QueryClusterMargin,
)

results_path = 'outputs/sampling_examples'
os.makedirs(results_path, exist_ok=True)

# %%

data = np.concatenate(
    (
        np.random.normal(loc=(-1, 2), scale=(1, 1), size=(100, 2)),
        np.random.normal(loc=(-2, -0.5), scale=(3, 1), size=(100, 2)),
        np.random.normal(loc=(3, 4), scale=(0.5, 2), size=(100, 2)),
        np.random.normal(loc=(-5, 6), scale=(0.5, 0.5), size=(100, 2)),
    )
)

inds_pool = np.arange(len(data))

inds_train = np.random.choice(inds_pool, size=15, replace=False)
inds_pool = np.setdiff1d(inds_pool, inds_train, assume_unique=True)

training_set = data[inds_train]
candidate_pool = data[inds_pool]

uncertainties_pool = candidate_pool[:, 1]
diversities_pool = QueryStrategy.get_diversities(
    candidate_pool, initial_design=training_set
)

# %%

plt.title("Candidate Pool Uncertainty")
plt.scatter(candidate_pool[:, 0], candidate_pool[:, 1], c=uncertainties_pool, cmap='viridis')
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.colorbar(label='"Uncertainty"')
plt.savefig(f'{results_path}/_pool_uncertainty.png')
plt.close()

# %%

plt.title("Candidate Pool Diversity")
plt.scatter(candidate_pool[:, 0], candidate_pool[:, 1], c=diversities_pool, cmap='viridis')
plt.scatter(training_set[:, 0], training_set[:, 1], c='k')
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.colorbar(label='Diversity from Training Set')
plt.savefig(f'{results_path}/_pool_uncertainty.png')
plt.close()

# %%

# define the query strategies

query_strategies = [
    QueryRandom(),
    QueryUncertainty(),
    QueryDiversity(),
    QueryGreedyDiversity(),
    QueryRankedBatchMode(),
    QueryDiverseMiniBatch(),
    QueryClusterMargin(),
]

# %%

# get their queries

query_strategy_queries = []

for query_strategy in query_strategies:
    query_strategy_queries.append(
        query_strategy.query(
            inds_pool,
            batch_size=10,
            candidate_pool=candidate_pool,
            initial_design=training_set,
            diversities_pool=diversities_pool,
            uncertainties_pool=uncertainties_pool,
        )
    )

# %%

# plot each strategy individually
    
for strategy, query in zip(query_strategies, query_strategy_queries):

    plt.title(f"Single Query from QueryStrategy: {strategy.NAME}")
    plt.scatter(candidate_pool[:, 0], candidate_pool[:, 1], label="pool")
    plt.scatter(training_set[:, 0], training_set[:, 1], label="train")
    plt.scatter(
        data[query][:, 0],
        data[query][:, 1],
        label="query",
    )
    plt.legend()
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")

    plt.savefig(f'{results_path}/{strategy.NAME}.png')
    plt.close()

# %%
    
# plot all strategies on one plot. It's a mess, but oh well.

plt.title("Query Strategy Comparisons: First Query")
plt.scatter(candidate_pool[:, 0], candidate_pool[:, 1], label="pool", zorder=0)
plt.scatter(training_set[:, 0], training_set[:, 1], label="train", zorder=15)

for strategy, query in zip(query_strategies, query_strategy_queries):
    plt.scatter(
        data[query][:, 0],
        data[query][:, 1],
        label=strategy.NAME,
        zorder=10
    )

plt.legend()
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")

plt.savefig(f'{results_path}/_all_strategies.png')
plt.close()

# %%
