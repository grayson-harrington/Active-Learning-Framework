

import random
import numpy as np

from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering


class QueryStrategy:
    """QueryStrategy is the parent class to custom sampling strategies"""

    def __init__(self):
        """super class constructor and default class variables"""
        self.NAME = None

        self.requires_uncertainties = False
        self.requires_diversities = False
        self.requires_candidate_pool = False
        self.requires_initial_design = False

    def check_common_input(self, inds_pool, batch_size):
        """Here we make sure that inds_pool and batch_size are given correctly

        Args:
            inds_pool (iterable): list of indices for remaining pool samples
            batch_size (int): the number of samples to query
        """

        assert inds_pool is not None, "inds_pool must be provided"
        assert batch_size is not None, "batch_size must be provided"

        try:
            iter(inds_pool)
        except TypeError:
            raise Exception(f"inds_pool must be an iterable, got {type(inds_pool)}")

        assert (
            isinstance(batch_size, int) and batch_size >= 1
        ), "batch_size must be a positive integer, i.e., >= 1"

    def check_requirements(self, **kwargs):
        """Here we check to make sure that uncertainties_pool and diversities_pool are provided if they are required
        through the use of self.requires_uncertainties and self.requires_diversities
        """

        if self.requires_uncertainties:
            assert (
                "uncertainties_pool" in kwargs
            ), f"This QueryStrategy, {type(self)}, requires that uncertainties be provided via the uncertainties_pool keyword"

        if self.requires_diversities:
            assert (
                "diversities_pool" in kwargs
            ), f"This QueryStrategy, {type(self)}, requires that diversities be provided via the diversities_pool keyword"

        if self.requires_candidate_pool:
            assert (
                "candidate_pool" in kwargs
            ), f"This QueryStrategy, {type(self)}, requires that a candidate pool be provided via the candidate_pool keyword"

        if self.requires_initial_design:
            assert (
                "initial_design" in kwargs
            ), f"This QueryStrategy, {type(self)}, requires that an initial design be provided via the initial_design keyword"

    def query(self, inds_pool, batch_size, **kwargs):
        """default query function. If not implemented in a subclass, this throws an error. The class QueryStrategy is not meant to be used itself.

        Args:
            inds_pool (numpy.ndarray): indices that index into the pool. Normally is np.arange(len(pool)), but can be different depending on user requirements.
            batch_size (int): Number of points to query
        """

        raise NotImplementedError(
            "Specific functionality must be implemented in derived classes"
        )

    @staticmethod
    def get_diversities(candidate_pool, initial_design=None):
        """Computes the diversity (euclidean distance) of each point in the candidate pool with respect to the initial design

        Args:
            candidate_pool (numpy.ndarray): The candidate pool of X points
            initial_design (numpy.ndarray, optional): The initial design of X points. Defaults to None.

        Returns:
            numpy.ndarray: Euclidean distance of each point in the candidate pool to the initial design
        """

        if initial_design is None:
            initial_design = np.random.choice(candidate_pool)

        kd = KDTree(initial_design)
        distances, _ = kd.query(candidate_pool)

        return distances

    @staticmethod
    def get_greedy_diverse_inds(candidate_pool, n_samples=None, initial_design=None):
        """Return the indices of candidate pool which are choosen based on maximum diversity scores in a greedy fashion. 
        See QueryStrategy.greedy_thining

        Args:
            candidate_pool (numpy.ndarray): The candidate pool of X point
            n_samples (int, optional): Number of smaples to select. Defaults to None.
            initial_design (numpy.ndarray, optional): The initial design of X points. Defaults to None.

        Returns:
            numpy.ndarray: indices of the selected samples
        """

        if n_samples is None:
            n_samples = len(candidate_pool)

        return QueryStrategy.greedy_thinning(
            candidate_pool, number_kept=n_samples, initial_design=initial_design
        )

    @staticmethod  # greedy_thinning code written by Andreas Robertson
    def greedy_thinning(candidate_pool, number_kept, initial_design=None):
        """Greedy thinning of the candidate pool given an initial design. 
        The most diverse (see QueryStrategy.get_diversities) point from the candidate pool is selected and added to the initial design. 
        This process is repeated until number_kept samples have been selected.

        Args:
            candidate_pool (numpy.ndarray): The candidate pool of X point
            number_kept (int, optional): Number of samples to select. Defaults to None.
            initial_design (numpy.ndarray, optional): The initial design of X points. Defaults to None.

        Returns:
            numpy.ndarray: indices of the selected samples
        """
        assert number_kept <= len(
            candidate_pool
        ), "The design must be smaller than the initial candidate pool."
        if initial_design is not None:
            assert type(initial_design) is np.ndarray, "Must pass an ndarray"
            assert (
                candidate_pool.shape[1:] == initial_design.shape[1:]
            ), "The initial design must have the same shape as the candidate pool."

        candidate_pool = candidate_pool.reshape(len(candidate_pool), -1)

        candidate_indexes = list(range(len(candidate_pool)))
        final_pool_indexes = []

        # initial choice:
        if initial_design is None:
            # pick a random point to start the design
            temp_index = random.choice(list(range(len(candidate_indexes))))
            final_pool_indexes.append(candidate_indexes[temp_index])
            del candidate_indexes[temp_index]

        # debug(final_pool_indexes, candidate_indexes, candidate_pool)

        # produce the remaining structure
        while len(final_pool_indexes) < number_kept:
            if initial_design is None:
                current_design = candidate_pool[final_pool_indexes]
            else:
                # add initial design to current design
                current_design = np.concatenate(
                    [
                        initial_design,
                        candidate_pool[final_pool_indexes],
                    ],
                    axis=0,
                )
            remainder = candidate_pool[candidate_indexes]

            # build the KD tree:
            kd = KDTree(current_design)
            distances, _ = kd.query(remainder)
            max_distance_indx = distances.argmax()

            # add the point to the pool
            final_pool_indexes.append(candidate_indexes[max_distance_indx])
            del candidate_indexes[max_distance_indx]

            # debug(final_pool_indexes, candidate_indexes, candidate_pool)

        return final_pool_indexes


class QueryRandom(QueryStrategy):
    """Random sampling query strategy"""

    def __init__(self):
        super().__init__()
        self.NAME = "random"

    def query(self, inds_pool, batch_size, **kwargs):
        super().check_common_input(inds_pool, batch_size)
        super().check_requirements(**kwargs)

        return np.random.choice(inds_pool, size=batch_size, replace=False)


class QueryUncertainty(QueryStrategy):
    """Max variance/uncertainty query strategy"""

    def __init__(self):
        super().__init__()
        self.NAME = "uncertainty"
        self.requires_uncertainties = True

    def query(self, inds_pool, batch_size, **kwargs):
        super().check_common_input(inds_pool, batch_size)
        super().check_requirements(**kwargs)

        inds = np.argpartition(kwargs["uncertainties_pool"], -batch_size)[-batch_size:]
        return inds_pool[inds]


class QueryDiversity(QueryStrategy):
    """Max diversity query strategy. See QueryStrategy.get_diversities"""

    def __init__(self):
        super().__init__()
        self.NAME = "diversity"
        self.requires_diversities = True

    def query(self, inds_pool, batch_size, **kwargs):
        super().check_common_input(inds_pool, batch_size)
        super().check_requirements(**kwargs)

        inds = np.argpartition(kwargs["diversities_pool"], -batch_size)[-batch_size:]
        return inds_pool[inds]


class QueryGreedyDiversity(QueryStrategy):
    """Greedy diversity query strategy. See QueryStrategy.get_greedy_diverse_inds"""

    def __init__(self):
        super().__init__()
        self.NAME = "greedy_diversity"
        self.requires_candidate_pool = True
        self.requires_initial_design = True

    def query(self, inds_pool, batch_size, **kwargs):
        super().check_common_input(inds_pool, batch_size)
        super().check_requirements(**kwargs)

        inds = self.get_greedy_diverse_inds(
            kwargs["candidate_pool"],
            n_samples=batch_size,
            initial_design=kwargs["initial_design"],
        )

        return inds_pool[inds]


class QueryRankedBatchMode(QueryStrategy):
    """Ranked Batch-mode sampling
        method developed by Cordoso et al. 
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

        Here we make a slight adaptation so that the scaling factor, alpha, 
        which transitions the sampling from pure greedy diversity to pure max variance, is related to the value of expected_max_samples.
        If expected_max_samples is not given, then it defaults to the size of the candidate pool.
    """

    def __init__(self, expected_max_samples=None):
        super().__init__()
        self.NAME = "ranked_batch_mode"
        self.requires_uncertainties = True
        self.requires_candidate_pool = True
        self.requires_initial_design = True

        self.expected_max_samples = expected_max_samples

    def query(self, inds_pool, batch_size, **kwargs):
        super().check_common_input(inds_pool, batch_size)
        super().check_requirements(**kwargs)

        if self.expected_max_samples is None:
            self.expected_max_samples = len(inds_pool)

        candidate_pool = kwargs["candidate_pool"]
        initial_design = kwargs["initial_design"]
        uncertainties_pool = kwargs["uncertainties_pool"]

        # need to normalize uncertainties to be from 0-1
        # 0 being least uncertain, 1 being most uncertain
        # this is an adaptation based on the original rank batch mode
        uncertainties_pool = (
            MinMaxScaler().fit_transform(uncertainties_pool.reshape(-1, 1)).squeeze()
        )

        queried_inds = []

        n_instances = (
            batch_size if batch_size < len(candidate_pool) else len(candidate_pool)
        )
        for _ in range(n_instances):
            diversities_pool = QueryStrategy.get_diversities(
                candidate_pool,
                initial_design=np.concatenate(
                    (initial_design, candidate_pool[queried_inds])
                ),
            )

            # need to normalize diverities to be from 0-1
            # 0 being least diverse, 1 being most diverse
            # this is the same as in the original rank batch mode
            diversities_pool = (
                MinMaxScaler().fit_transform(diversities_pool.reshape(-1, 1)).squeeze()
            )

            # calculate alpha for doing transtion from diversity to uncertainty
            if self.expected_max_samples is not None:
                l_mag = len(initial_design)
                u_mag = self.expected_max_samples - l_mag
                alpha = u_mag / (u_mag + l_mag)
            else:
                l_mag = len(initial_design)
                u_mag = len(candidate_pool)
                alpha = u_mag / (u_mag + l_mag)
            alpha = 0 if alpha < 0 else (1 if alpha > 1 else alpha)
            similarity_pool = 1 / (1 + diversities_pool)
            scores_pool = (
                alpha * (1 - similarity_pool) + (1 - alpha) * uncertainties_pool
            )

            # ensure that no indices selected previously are selected again
            scores_pool[queried_inds] = 0

            best_ind = np.argmax(scores_pool)

            queried_inds.append(best_ind)

        return inds_pool[queried_inds]


class QueryDiverseMiniBatch(QueryStrategy):
    """Diverse Mini-batch sampling
        method developed by Fedor Zhdanov
        https://arxiv.org/abs/1901.05954
    """

    def __init__(self, beta=10):
        # beta = 10 is found to work well independent of dataset size. Large beta values can be used to promote diversification
        super().__init__()
        self.NAME = "diverse_mini_batch"
        self.requires_uncertainties = True
        self.requires_candidate_pool = True

        self.beta = beta

    def query(self, inds_pool, batch_size, **kwargs):
        super().check_common_input(inds_pool, batch_size)
        super().check_requirements(**kwargs)

        # get top beta*batch_size uncertainty samples
        inds_chosen = np.argsort(kwargs["uncertainties_pool"])[
            -1 : -self.beta * batch_size : -1
        ]

        # do kmeans on selected data using batch_size clusters
        data_chosen = kwargs["candidate_pool"][inds_chosen]
        kmeans = KMeans(n_clusters=batch_size).fit(data_chosen)

        # identify the closest point to each cluster.
        # Should just be one O(|data|) loop
        inds_closests = np.ones(batch_size, dtype=int) * -1
        dists_closests = np.ones(batch_size) * np.inf
        for i, point in enumerate(data_chosen):
            label = kmeans.labels_[i]
            centroid = kmeans.cluster_centers_[label]

            dist = np.linalg.norm(centroid - point)
            if dist < dists_closests[label]:
                dists_closests[label] = dist
                inds_closests[label] = i

        return inds_pool[inds_chosen[inds_closests]]


class QueryClusterMargin(QueryStrategy):
    """Cluster Margin sampling
        method developed by Citovsky et al.
        https://proceedings.neurips.cc/paper/2021/file/64254db8396e404d9223914a0bd355d2-Paper.pdf
    """

    def __init__(
        self,
        frac_hac=1.0,
        epsilon=1,
        beta=10,
    ):
        super().__init__()
        self.NAME = "cluster_margin"
        self.requires_uncertainties = True
        self.requires_candidate_pool = True

        self.frac_hac = frac_hac
        self.beta = beta
        self.epsilon = epsilon

        self.hac = None  # will store slearn AgglomerativeClustering object
        self.hac_centroids = []
        self.pool_labels = {}  # will hold label for all points in initial pool

    def label_point(self, point):
        if tuple(point) in self.pool_labels:
            return self.pool_labels[tuple(point)]

        min_dist = np.inf
        min_dist_label = 0
        for i, centroid in enumerate(self.hac_centroids):
            d = np.linalg.norm(point - centroid)
            if d < min_dist:
                min_dist = d
                min_dist_label = i

        return min_dist_label

    def label_points(self, points, return_clustered_inds=False):
        labels = []
        clustered_inds = [[] for _ in range(self.hac.n_clusters_)]

        for i, point in enumerate(points):
            label = self.label_point(point)
            labels.append(label)
            if return_clustered_inds:
                clustered_inds[label].append(i)

        if return_clustered_inds:
            return labels, clustered_inds
        else:
            return labels

    def preprocessing(self, candidate_pool):
        n_pool = len(candidate_pool)

        inds_hac = np.random.choice(
            n_pool,
            size=int(n_pool * self.frac_hac),
            replace=False,
        )
        inds_others = np.setdiff1d(np.arange(n_pool), inds_hac, assume_unique=True)

        data_hac = candidate_pool[inds_hac]
        data_others = candidate_pool[inds_others]

        # hac trained on inds_hac
        self.hac = AgglomerativeClustering(
            n_clusters=None, linkage="average", distance_threshold=self.epsilon
        ).fit(data_hac)
        labels_hac = self.hac.labels_

        # find centroid for each cluster
        for i in range(self.hac.n_clusters_):
            inds_i = np.where(labels_hac == i)[0]
            centroid = np.mean(data_hac[inds_i], axis=0)
            self.hac_centroids.append(centroid)

        # get cluster labels for the inds that weren't used to create hac
        labels_others = self.label_points(data_others)

        # create class dictionary to hold all pool points and labels.
        for point, label in zip(data_hac, labels_hac):
            self.pool_labels[tuple(point)] = label

        for point, label in zip(data_others, labels_others):
            self.pool_labels[tuple(point)] = label

    def query(self, inds_pool, batch_size, **kwargs):
        super().check_common_input(inds_pool, batch_size)
        super().check_requirements(**kwargs)

        # do cluster-margin sampling

        # get arrays
        candidate_pool = kwargs["candidate_pool"]
        uncertainties_pool = kwargs["uncertainties_pool"]

        # do initial HAC if it hasn't been done already
        if self.hac is None:
            self.preprocessing(kwargs["candidate_pool"])

        # get the most uncertain inds and samples (beta*batch_size = km)
        inds_km = np.argsort(uncertainties_pool)[-1 : -batch_size * self.beta : -1]

        points_km = candidate_pool[inds_km]
        labels_km, clustered_inds_km = self.label_points(
            points_km, return_clustered_inds=True
        )

        # get the clusters associated with the km points
        # sort them ascendingly by number of points in cluster
        clusters = sorted(
            np.unique(labels_km),
            key=lambda label: len(clustered_inds_km[label]),
        )

        inds_km_selected = []
        ci = 0  # to keep track of cluster position in clusters
        for _ in range(batch_size):
            # jump into cluster at clusters[ci]
            cluster = clusters[ci]

            # pull random sample from cluster using clustered_inds_km[cluster]
            # remove selected point from clustered_inds_km[cluster]
            ind_selected = np.random.choice(clustered_inds_km[cluster])
            clustered_inds_km[cluster].remove(ind_selected)

            inds_km_selected.append(ind_selected)

            # if clusters[ci] is full, remove clusters[ci]
            if len(clustered_inds_km[cluster]) == 0:
                clusters.pop(ci)
            else:
                # increment ci (only need to do if ci wasn't popped)
                ci += 1

            # make sure to loop back to 0 if you go past the end of the array
            ci = 0 if ci == len(clusters) else ci

        return inds_pool[inds_km[inds_km_selected]]
