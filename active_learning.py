import os
import pickle
from copy import copy, deepcopy

from time import time as now

import numpy as np
from .error_metrics import get_error

from .sampling_strategies import QueryStrategy


class ActiveLearner:
    """ActiveLearner is meant to be a class for performing general purpose active learning.
    It is written so that it can either be used in a very basic way with an initialization and the run() function,
    or it can be used in a much more specific way by the user by calling the individual functions inside the class however needed.

    Pool-based sampling is done in one of two ways. The provided query strategy is used to identify indices in the
    X_pool for which labels are needed. Those labels are either retrieved directly from y_pool if provided, or they are retrieved from the
    probe_function, which takes X as input and returns the associated y values
    """

    def __init__(
        self,
        Xy_pool,
        Xy_init,
        Xy_test,
        model,
        query_strategy,
        probe_function,
        batch_size,
        random_state=None,
        model_kwargs={},
    ):
        """Initialization of the ActiveLearner class

        Args:
            Xy_pool (tuple): The pool X and y numpy arrays provided as (X, y) in tuple form. Only X is required.
            Xy_init (tuple): The initial X and y training points provided as numpy arrays as (X, y) in tuple form. Both X and y are required.
            Xy_test (tuple): The test X and y numpy arrays provided as (X, y) in tuple form. Not required, but both X and y are required if it is given.
            model: The model used for active learning. Must have fit() and predict() functions. The predict function must return (y_pred, y_var)
            query_strategy (QueryStrategy): A QueryStrategy subclass used for querying new points in the pool.
            probe_function (function): This function takes X as input and outputs associated y values. Required if y_pool is not given in Xy_pool.
            batch_size (int): The number of points to sample in each active learning iteration
            random_state (int, optional): Random state. Defaults to None.
            model_kwargs (dict, optional): additional kwargs that are used in model.predict(). For example, the sklearn GPR class would have model_kwargs={"return_std": True}. Defaults to {}.
        """

        np.random.seed(random_state)

        self.model = model
        self.model_kwargs = model_kwargs
        self.query_strategy = query_strategy
        self.probe_function = probe_function

        self.batch_size = batch_size

        self.X_pool = Xy_pool[0]
        self.y_pool = Xy_pool[1]

        self.X_train = np.copy(Xy_init[0])
        self.y_train = np.copy(Xy_init[1])

        self.do_test = False
        self.X_test = None
        self.y_test = None
        if Xy_test is not None:
            self.do_test = True
            self.X_test = np.copy(Xy_test[0])
            self.y_test = np.copy(Xy_test[1])

        self.inds_pool = np.arange(self.X_pool.shape[0])
        self.inds_selected = []
        self.inds_queried = []

        self.save_states = {
            "n_iter": 0,
            "models": [],
            "err_train": [],
            "err_test": [],
            "n_selected": [],
            "n_samples": [],
            "inds_selected": [],
            "inds_queried": [],
            "train_time": [],
            "query_time": [],
            "update_time": [],
        }

    def train_model(self):
        """train model with the model's fit() function"""

        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(
        self,
        X,
        y=None,
        error_metric="MAE",
        error_axis=None,
        return_pred_uncert=False,
    ):
        """evaluation of model accuracy given X and y. If y is not provided, then the error is None.

        Args:
            X (numpy.ndarray): X values
            y (numpy.ndarray, optional): True y values. If provided, error is calculated using the following parameters. Defaults to None.
            error_metric (str, optional): Acronym for type of error calculation that should be done. Common options are "MSE", "RMSE", "MAE". Defaults to "MAE".
            error_axis (int|tuple, optional): The axis on which the error calculation should be performed. Works just like the axis parameter in most numpy functions. Defaults to None.
            return_pred_uncert (bool, optional): If True, returns both the model predictions and their uncertainties (var, std, etc... it depends on the model being used). Defaults to False.

        Returns:
            (tuple): Return value is either (y_pred, y_uncertainty, error) or error. The former is true if return_pred_uncert is true. Otherwise the later is true.
        """

        y_pred, y_uncert = self.model.predict(X, **self.model_kwargs)

        error = (
            get_error(y, y_pred, metric=error_metric, axis=error_axis)
            if y is not None
            else None
        )

        if return_pred_uncert:
            return y_pred, y_uncert, error
        else:
            return error

    def run(
        self,
        n_iter=10,
        err_threshold=None,
        err_set="test",
        batch_size=None,
        save_percent=0.1,
        save_path=None,
    ):
        """The main active learning loop. Once the loop as started, in each iteration the following take place:
        1) training datasets are updated,
        2) the model is trained,
        3) the model is evaluated on train, test, and pool,
        4) the performance on pool is used to new points to add to the training datasets.
        This is all done inside the single_update function

        Args:
            n_iter (int, optional): Number of active learning iterations. Defaults to 10.
            err_threshold (float, optional): If the model error is <= err_threshold it stops iterating. Defaults to None.
            err_set (str, optional): Can be "train" or "test". The error each iteration will be calculated using one of
                                    those sets for determining if the threshold has been met. Defaults to "test".
            batch_size (int, optional): Number of points to sample in each iteration. If None, self.batch_size is used.
                                    This allows the user to change the batch size throughout an active learning run. Defaults to None.
            save_percent (float, optional): Value between 0.0 and 1.0. Dictates how often the model and all important results should be saved to file. Defaults to 0.1.
            save_path (str, optional): If provided, the model and all important results are saved to this path. Defaults to None.
        """

        if err_threshold is not None:
            assert err_set in [
                "test",
                "train",
            ], f"err_set must be either 'test' or 'train', got {err_set}"

        if len(save_path.split("/")) > 1:
            print(save_path.split("/")[:-1])
            print(os.path.join(*save_path.split("/")[:-1]))
            os.makedirs(os.path.join(*save_path.split("/")[:-1]), exist_ok=True)

        if batch_size is not None:
            self.batch_size = batch_size

        save_mod = int(n_iter * save_percent)

        print("Starting Active Learning Loop")
        print(f"Query Strategy: {self.query_strategy.NAME}")
        print("Parameters:")
        print("\titerations:\t", n_iter)
        print("\tbatch size:\t", self.batch_size)
        print("\terror thresh:\t", err_threshold)
        print()
        print("Iter\tTrain Error\tTest Error")
        print("-" * 30)

        err = np.inf
        for i in range(1, n_iter + 1):
            self.single_update()

            # if err_threshold, get err (either err_train or err_test)
            if err_threshold is not None:
                err = self.save_states[f"err_{err_set}"]

                if err <= err_threshold:
                    break

            # make sure the number of samples queried in the next iteration does not surpass the size of the pool
            if self.batch_size * 2 > len(self.inds_pool):
                self.batch_size = len(self.inds_pool) - self.batch_size

                if self.batch_size == 0:
                    break

            # save al state every so often, like 10% of iterations
            saved = False
            if (save_path is not None) and (i % save_mod == 0):
                self.save_state(save_path)
                saved = True

            print(
                f'{i}\t{self.save_states["err_train"][-1]:.3e}\t{self.save_states["err_test"][-1]:.3e}'
                + ("\tresults saved" if saved else "")
            )

        # save final results
        if save_path is not None:
            self.save_state(save_path)
            print(f"final results saved to {save_path}")

        print(
            f'Training Error: {self.save_states["err_train"][-1]:.3e}\tTesting Error: {self.save_states["err_test"][-1]:.3e}'
        )
        print("Done.")

    def single_update(self):
        """This function is called in the active learning loop and does the following:
        1) training datasets are updated,
        2) the model is trained,
        3) the model is evaluated on train, test, and pool,
        4) the performance on pool is used to new points to add to the training datasets.
        """

        t_u = now()

        # update pool and training sets with queried points
        self.update_datasets(self.inds_queried)

        # train
        t_t = now()
        self.train_model()
        t_t = now() - t_t

        # evaluate model on train
        err_train = self.evaluate_model(self.X_train, self.y_train, error_metric="RMSE")

        # evaluate model on test
        err_test = None
        if self.do_test:
            err_test = self.evaluate_model(
                self.X_test, self.y_test, error_metric="RMSE"
            )

        # evaluate model on pool (this must be done on what remains of the pool)
        _, y_uncert, _ = self.evaluate_model(
            self.X_pool[self.inds_pool], return_pred_uncert=True, error_metric="RMSE"
        )

        # query
        t_q = now()
        self.inds_queried = self.query(self.batch_size, uncertainties_pool=y_uncert)
        t_q = now() - t_q

        t_u = now() - t_u

        # save all things that should be saved
        self.save_states["n_iter"] = self.save_states["n_iter"] + 1
        self.save_states["model"].append(
            deepcopy(self.model)
        )  # this could be memory intensive for large models with many sampling iterations
        self.save_states["err_train"].append(err_train)
        self.save_states["err_test"].append(err_test)
        self.save_states["n_selected"].append(len(self.inds_selected))
        self.save_states["n_samples"].append(len(self.X_train))
        self.save_states["inds_selected"].append(copy(self.inds_selected))
        self.save_states["inds_queried"].append(copy(self.inds_queried))
        self.save_states["update_time"].append(t_u)
        self.save_states["query_time"].append(t_q)
        self.save_states["train_time"].append(t_t)

    def query(self, batch_size, **kwargs):
        """Given a batch size and kwargs associated with the specific QueryStrategy, pool indices are sampled to be labeled.

        Args:
            batch_size (int): The number of points to be labeled.

        Returns:
            list | numpy.ndarray: list of pool indices which need labelling
        """

        # get diversities here if needed
        diversities_pool = None
        if self.query_strategy.requires_diversities:
            diversities_pool = QueryStrategy.get_diversities(
                self.X_pool[self.inds_pool], initial_design=self.X_train
            )

        return self.query_strategy.query(
            self.inds_pool,
            batch_size,
            # below are used for specific query strategies, not all
            diversities_pool=diversities_pool,
            candidate_pool=self.X_pool[self.inds_pool],
            initial_design=self.X_train,
            **kwargs,
        )

    def probe(self, f, X):
        """This function calls f(X), where f is the premade probe_function for aquiring labels for the samples in X

        Args:
            f (function): probe function. Takes X as input and outputs y
            X (numpy.ndarray): X values to label. X is of shape (samples, features)

        Raises:
            Exception: If f is not callable, an exception is raised

        Returns:
            numpy.ndarray: array of y values associated with the samples in X
        """

        if not callable(f):
            raise Exception(
                "function", f"f must be a function that takes X as input. Got {type(f)}"
            )

        return f(X)

    def update_datasets(self, inds=None):
        """Here the training sets are updated using the queried points. The arrays used to keep track of queried, sampled, and pool indices aare udpated internally.

        Args:
            inds (list | numpy.ndarray, optional): If provided, inds is used rather than the most recent queried indices. Defaults to None.
        """

        if inds is not None:
            self.inds_queried = inds
        if len(self.inds_queried) == 0:
            return

        # get new X, y values
        new_X = self.X_pool[self.inds_queried]
        if self.y_pool is None:
            new_y = self.probe(self.probe_function, new_X)
        else:
            new_y = self.y_pool[self.inds_queried]

        # update train arrays
        self.X_train = np.concatenate((self.X_train, new_X), axis=0)
        self.y_train = np.concatenate((self.y_train, new_y), axis=0)

        # update inds arrays
        self.inds_selected.extend(self.inds_queried)
        self.inds_pool = np.setdiff1d(
            self.inds_pool, self.inds_selected, assume_unique=True
        )
        self.inds_queried = []

    def save_state(self, path):
        """Saves the current ActiveLearner state to file

        Args:
            path (str): file path for saving
        """

        with open(path, "wb") as f:
            pickle.dump(self.save_states, f)
