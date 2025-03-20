from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from driftguard.driftlens import driftlens
from tqdm import tqdm
import numpy as np
import json
import os


class ThresholdClass:
    """ Threshold Class: it contains all the attributes and methods of the threshold. """
    def __init__(self):
        self.label_list = None  # List of labels used to train the model
        self.batch_n_pc = None  # Number of principal components to reduce the embedding for the entire batch drift
        self.per_label_n_pc = None  # number of principal components to reduce embedding for the per-label drift
        self.window_size = None  # Window size that will be used in the online phase

        self.mean_distances_dict = {}
        self.std_distances_dict = {}

        self.mean_distances_dict["per-label"] = {}
        self.std_distances_dict["per-label"] = {}

        self.mean_distances_dict["batch"] = None
        self.std_distances_dict["batch"] = None

        self.distribution_distances_list = None

        self.description = ""
        self.threshold_method_name = ""
        return

    def fit(self, batch_mean, batch_std, per_label_mean_dict, per_label_std_dict, label_list, batch_n_pc, per_label_n_pc, window_size, distribution_distances_list, threshold_method_name=""):
        """ Fits the threshold attributes.
        Args:
            batch_mean: Mean distance for the batch.
        Returns:
            None
        """
        self.mean_distances_dict["batch"] = batch_mean
        self.std_distances_dict["batch"] = batch_std

        self.mean_distances_dict["per-label"] = per_label_mean_dict
        self.std_distances_dict["per-label"] = per_label_std_dict

        self.distribution_distances_list = distribution_distances_list

        self.label_list = label_list
        self.batch_n_pc = batch_n_pc
        self.per_label_n_pc = per_label_n_pc
        self.window_size = window_size

        self.threshold_method_name = threshold_method_name
        return

    def _fit_from_dict(self, threshold_dict):
        """ Fits the threshold class attributes from a dictionary. """
        self.__dict__.update(threshold_dict)
        return

    def save(self, folderpath, threshold_name):
        """ Saves persistently on disk the threshold.
        Args:
            folderpath (str): Folder path where save the threshold.
            threshold_name (str): Filename of the threshold file.
        Returns:
            (str): Threshold filepath.
        """
        experiment_json_str = json.dumps(self, default=lambda x: getattr(x, '__dict__', str(x)))
        experiment_json_dict = json.loads(experiment_json_str)

        with open(os.path.join(folderpath, "{}.json".format(threshold_name)), 'w') as fp:
            json.dump(experiment_json_dict, fp)
        return

    def load(self, folderpath, threshold_name):
        """ Loads the threshold from folder.
        Args:
            folderpath: (str) folderpath containing the threshold json.
            threshold_name: (str) name of the threshold json file.
        """
        if threshold_name.endswith(".json"):
            threshold_filepath = os.path.join(folderpath, threshold_name)
        else:
            threshold_filepath = '{}.json'.format(os.path.join(folderpath, threshold_name))

        with open(threshold_filepath) as json_file:
            json_dict = json.load(json_file)

        if json_dict is None:
            raise Exception(f'Error: impossible to parse threshold json.')
        else:
            try:
                self._fit_from_dict(json_dict)
            except Exception as e:
                raise Exception(f'Error in deserializing the threshold json: {e}')
        return

    def set_description(self, description):
        """ Sets the 'description' attribute of the threshold. """
        self.description = description
        return

    def get_description(self):
        """ Gets the 'description' attribute of the threshold. """
        return self.description

    def get_label_list(self):
        """ Gets the 'label_list' attribute of the threshold. """
        return self.label_list

    def get_batch_mean_distance(self):
        """ Gets the mean distance for the batch. """
        return self.mean_distances_dict["batch"]

    def get_batch_std_distance(self):
        """ Gets the standard deviation of the distance for the batch. """
        return self.std_distances_dict["batch"]

    def get_mean_distance_by_label(self, label):
        """ Gets the per-label mean distance for a given label. """
        return self.mean_distances_dict["per-label"][str(label)]

    def get_std_distance_by_label(self, label):
        """ Gets the per-label standard deviation of the distance for a given label. """
        return self.std_distances_dict["per-label"][str(label)]

    def get_per_label_mean_distances(self):
        """ Gets the dictionary of per-label mean distances. """
        return self.mean_distances_dict["per-label"]

    def get_per_label_std_distances(self):
        """ Gets the dictionary of per-label standard deviation distances. """
        return self.std_distances_dict["per-label"]


class ThresholdEstimatorMethod(ABC):
    """ Abstract Baseline Estimator Method class. """
    def __init__(self, label_list, threshold_method_name):
        self.label_list = label_list
        self.threshold_method_name = threshold_method_name
        return

    @abstractmethod
    def estimate_threshold(self, *args) -> ThresholdClass:
        """ Abstract method: Estimates the Threshold and returns a ThresholdClass object. """
        pass


class KFoldThresholdEstimator(ThresholdEstimatorMethod):
    """ KFold Threshold Estimator Class: Implementation of the ThresholdEstimatorMethod Abstract Class. """
    def __init__(self, label_list):
        ThresholdEstimatorMethod.__init__(self, label_list, threshold_method_name="KFoldThresholdEstimator")
        return

    def estimate_threshold(self, E, Y, batch_n_pc, per_label_n_pc, window_size, flag_shuffle=True):
        """ Estimates the Threshold using K-Fold cross-validation. """
        if window_size > E.shape[0] * 2:
            raise ValueError("Window size is too large compared to the number of samples.")

        K = round(E.shape[0] / window_size)
        E_selected = E[:window_size * K]
        Y_selected = Y[:window_size * K]

        distribution_distances_list = self._kfold_threshold_estimation(E_selected, Y_selected, K, batch_n_pc, per_label_n_pc, flag_shuffle, start_window_id=0)

        batch_mean = np.mean(distribution_distances_list["batch"])
        batch_std = np.std(distribution_distances_list["batch"])
        per_label_mean = {}
        per_label_std = {}

        for label in self.label_list:
            per_label_mean[str(label)] = np.mean(distribution_distances_list["per-label"][str(label)])
            per_label_std[str(label)] = np.std(distribution_distances_list["per-label"][str(label)])

        threshold = ThresholdClass()
        threshold.fit(batch_mean, batch_std, per_label_mean, per_label_std, self.label_list, batch_n_pc, per_label_n_pc,
                      window_size, distribution_distances_list, self.threshold_method_name)
        return threshold

    def _kfold_threshold_estimation(self, E, Y, K, batch_n_pc, per_label_n_pc, flag_shuffle, start_window_id=0):
        window_id = start_window_id
        distribution_distances_list = {"batch": [], "per-label": {str(l): [] for l in self.label_list}}

        folds = StratifiedKFold(n_splits=K, shuffle=flag_shuffle, random_state=0)

        for approximated_baseline_idxs, simulated_new_window_idx in folds.split(E, Y):
            E_b, E_w = E[approximated_baseline_idxs], E[simulated_new_window_idx]
            Y_b, Y_w = Y[approximated_baseline_idxs], Y[simulated_new_window_idx]

            approximated_baseline = self._baseline.StandardBaselineEstimator(self.label_list, batch_n_pc,
                                                                            per_label_n_pc).estimate_baseline(E_b, Y_b)

            window_distribution_distances_dict = driftlens.DriftLens()._compute_frechet_distribution_distances(
                self.label_list, approximated_baseline, E_w, Y_w, window_id)

            distribution_distances_list["batch"].append(window_distribution_distances_dict["batch"])
            for l in self.label_list:
                distribution_distances_list["per-label"][str(l)].append(window_distribution_distances_dict["per-label"][str(l)])
            window_id += 1

        return distribution_distances_list


class RepeatedKFoldThresholdEstimator(KFoldThresholdEstimator):
    """ Repeated KFold Threshold Estimator Class: Extends KFoldThresholdEstimator with repetitions. """
    def __init__(self, label_list):
        ThresholdEstimatorMethod.__init__(self, label_list, threshold_method_name="RepeatedKFoldThresholdEstimator")
        return

    def estimate_threshold(self, E, Y, batch_n_pc, per_label_n_pc, window_size, repetitions=2, flag_shuffle=True):
        """ Estimates the Threshold with repeated K-Fold cross-validation. """
        if window_size > E.shape[0] * 2:
            raise ValueError("Window size is too large compared to the number of samples.")

        K = round(E.shape[0] / window_size)
        E_selected = E[:window_size * K]
        Y_selected = Y[:window_size * K]

        distribution_distances_list = {"batch": [], "per-label": {str(l): [] for l in self.label_list}}

        for r in tqdm(range(repetitions)):
            partial_distribution_distances_list = self._kfold_threshold_estimation(E_selected, Y_selected, K, batch_n_pc, per_label_n_pc, flag_shuffle, start_window_id=r)
            distribution_distances_list["batch"] += partial_distribution_distances_list["batch"]
            for label in self.label_list:
                distribution_distances_list["per-label"][str(label)] += partial_distribution_distances_list["per-label"][str(label)]

        batch_mean = np.mean(distribution_distances_list["batch"])
        batch_std = np.std(distribution_distances_list["batch"])
        per_label_mean = {}
        per_label_std = {}

        for label in self.label_list:
            per_label_mean[str(label)] = np.mean(distribution_distances_list["per-label"][str(label)])
            per_label_std[str(label)] = np.std(distribution_distances_list["per-label"][str(label)])

        threshold = ThresholdClass()
        threshold.fit(batch_mean, batch_std, per_label_mean, per_label_std, self.label_list, batch_n_pc, per_label_n_pc,
                      window_size, distribution_distances_list, self.threshold_method_name)
        return threshold


class StandardThresholdEstimator(ThresholdEstimatorMethod):
    """ Standard Threshold Estimator Class: Estimates threshold using fixed windows. """
    def __init__(self, label_list):
        ThresholdEstimatorMethod.__init__(self, label_list, threshold_method_name="StandardThresholdEstimator")
        return

    def estimate_threshold(self, E, Y, baseline, window_size, flag_shuffle=True):
        """ Estimates the Threshold by dividing the dataset into fixed windows. """
        batch_n_pc = baseline.batch_n_pc
        per_label_n_pc = baseline.per_label_n_pc

        if E.shape[0] % window_size != 0:
            print(f"Warning: Number of samples is not an exact multiple of window size. Discarding {E.shape[0] % window_size} samples.")

        if flag_shuffle:
            p = np.random.permutation(len(E))
            E = E[p]
            Y = Y[p]

        n_windows = E.shape[0] // window_size
        distribution_distances_list = {"batch": [], "per-label": {str(l): [] for l in self.label_list}}

        for i in range(n_windows):
            window_id = i
            E_w = E[i * window_size:(i * window_size) + window_size]
            Y_w = Y[i * window_size:(i * window_size) + window_size]

            window_distribution_distances_dict = driftlens.DriftLens()._compute_frechet_distribution_distances(
                self.label_list, baseline, E_w, Y_w, window_id)

            distribution_distances_list["batch"].append(window_distribution_distances_dict["batch"])
            for l in self.label_list:
                distribution_distances_list["per-label"][str(l)].append(window_distribution_distances_dict["per-label"][str(l)])

        batch_mean = np.mean(distribution_distances_list["batch"])
        batch_std = np.std(distribution_distances_list["batch"])
        per_label_mean = {}
        per_label_std = {}

        for label in self.label_list:
            per_label_mean[str(label)] = np.mean(distribution_distances_list["per-label"][str(label)])
            per_label_std[str(label)] = np.std(distribution_distances_list["per-label"][str(label)])

        threshold = ThresholdClass()
        threshold.fit(batch_mean, batch_std, per_label_mean, per_label_std, self.label_list, batch_n_pc, per_label_n_pc,
                      window_size, distribution_distances_list, self.threshold_method_name)
        return threshold


class RandomSamplingThresholdEstimator(ThresholdEstimatorMethod):
    """ Random Sampling Threshold Estimator Class: Estimates threshold via random sampling. """
    def __init__(self, label_list):
        ThresholdEstimatorMethod.__init__(self, label_list, threshold_method_name="RandomSamplingThresholdEstimator")
        return

    def estimate_threshold(self, E, Y, baseline, window_size, n_samples, flag_replacement=True, flag_shuffle=True, proportional_flag=False, proportions_dict=None):
        """ Estimates the Threshold using random sampling. """
        batch_n_pc = baseline.batch_n_pc
        per_label_n_pc = baseline.per_label_n_pc

        per_batch_distances = []
        per_label_distances = {label: [] for label in self.label_list}

        print("Threshold Estimation")
        for i in tqdm(range(n_samples)):
            if proportional_flag:
                if proportions_dict is None:
                    raise ValueError("proportions_dict must be provided when proportional_flag is True")
                E_windows, Y_predicted_windows, Y_original_windows = self._proportional_sampling(
                    self.label_list, E, Y, Y, window_size, 1, flag_replacement, proportions_dict)
            else:
                E_windows, Y_predicted_windows, Y_original_windows = self._balanced_sampling(
                    self.label_list, E, Y, Y, window_size, 1, flag_replacement)

            if flag_shuffle:
                E_windows[0], Y_predicted_windows[0], Y_original_windows[0] = self._shuffle_dataset(
                    E_windows[0], Y_predicted_windows[0], Y_original_windows[0])

            dl_th = driftlens.DriftLens(self.label_list)
            dl_th.set_baseline(baseline)
            distribution_distances = dl_th.compute_window_list_distribution_distances(E_windows, Y_predicted_windows)

            per_batch_distances.append(distribution_distances[0][0]["per-batch"])
            for l in self.label_list:
                per_label_distances[l].append(distribution_distances[0][0]["per-label"][str(l)])

        per_batch_distances_arr = np.array(per_batch_distances)
        indices = (-per_batch_distances_arr).argsort()
        per_batch_distances_sorted = per_batch_distances_arr[indices]

        for l in self.label_list:
            per_label_distances[l] = sorted(per_label_distances[l], reverse=True)

        # Compute statistics for ThresholdClass
        batch_mean = np.mean(per_batch_distances_sorted)
        batch_std = np.std(per_batch_distances_sorted)
        per_label_mean = {str(l): np.mean(per_label_distances[l]) for l in self.label_list}
        per_label_std = {str(l): np.std(per_label_distances[l]) for l in self.label_list}

        distribution_distances_list = {"batch": per_batch_distances_sorted.tolist(), "per-label": {str(l): per_label_distances[l] for l in self.label_list}}

        threshold = ThresholdClass()
        threshold.fit(batch_mean, batch_std, per_label_mean, per_label_std, self.label_list, batch_n_pc, per_label_n_pc,
                      window_size, distribution_distances_list, self.threshold_method_name)
        return threshold

    @staticmethod
    def _proportional_sampling(label_list, E, Y_predicted, Y_original, window_size, n_windows, flag_replacement, proportions_dict):
        per_label_E = {str(l): E[Y_original == l].copy() for l in label_list}
        per_label_Y_predicted = {str(l): Y_predicted[Y_original == l].copy() for l in label_list}
        per_label_Y_original = {str(l): Y_original[Y_original == l].copy() for l in label_list}

        n_samples_per_label = {str(l): int(proportions_dict[str(l)] * window_size) for l in label_list}
        total_samples = sum(n_samples_per_label.values())
        n_residual_samples = window_size - total_samples

        E_windows, Y_predicted_windows, Y_original_windows = [], [], []

        for _ in range(n_windows):
            E_window_list, Y_predicted_window_list, Y_original_window_list = [], [], []

            for l in label_list:
                m_l = len(per_label_E[str(l)])
                n_samples = n_samples_per_label[str(l)]
                if m_l == 0:
                    print(f"Warning: No samples available for label {l}. Skipping.")
                    continue
                if m_l < n_samples and not flag_replacement:
                    print(f"Warning: Insufficient samples ({m_l}) for label {l} with n_samples={n_samples} and no replacement. Using all available.")
                    n_samples = m_l

                try:
                    l_idxs = np.random.choice(m_l, n_samples, replace=flag_replacement)
                except ValueError as e:
                    print(f"Error sampling label {l}: {e}. Using all available samples.")
                    l_idxs = np.arange(m_l)

                E_l_window = per_label_E[str(l)][l_idxs]
                Y_predicted_l_window = per_label_Y_predicted[str(l)][l_idxs]
                Y_original_l_window = per_label_Y_original[str(l)][l_idxs]

                E_window_list.extend(E_l_window.tolist())
                Y_predicted_window_list.extend(Y_predicted_l_window.tolist())
                Y_original_window_list.extend(Y_original_l_window.tolist())

                if not flag_replacement:
                    per_label_E[str(l)] = np.delete(per_label_E[str(l)], l_idxs, 0)
                    per_label_Y_predicted[str(l)] = np.delete(per_label_Y_predicted[str(l)], l_idxs, 0)
                    per_label_Y_original[str(l)] = np.delete(per_label_Y_original[str(l)], l_idxs, 0)

            if n_residual_samples > 0:
                labels, counts = np.unique(Y_original, return_counts=True)
                label_distribution = counts / counts.sum()
                additional_labels = np.random.choice(labels, n_residual_samples, replace=True, p=label_distribution)

                for l in additional_labels:
                    m_l = len(per_label_E[str(l)])
                    if m_l > 0:
                        idx = np.random.choice(m_l, 1, replace=False)
                        E_l_window = per_label_E[str(l)][idx]
                        Y_predicted_l_window = per_label_Y_predicted[str(l)][idx]
                        Y_original_l_window = per_label_Y_original[str(l)][idx]

                        E_window_list.extend(E_l_window.tolist())
                        Y_predicted_window_list.extend(Y_predicted_l_window.tolist())
                        Y_original_window_list.extend(Y_original_l_window.tolist())

                        if not flag_replacement:
                            per_label_E[str(l)] = np.delete(per_label_E[str(l)], idx, 0)
                            per_label_Y_predicted[str(l)] = np.delete(per_label_Y_predicted[str(l)], idx, 0)
                            per_label_Y_original[str(l)] = np.delete(per_label_Y_original[str(l)], idx, 0)

            E_windows.append(np.array(E_window_list))
            Y_predicted_windows.append(np.array(Y_predicted_window_list))
            Y_original_windows.append(np.array(Y_original_window_list))

        return E_windows, Y_predicted_windows, Y_original_windows

    @staticmethod
    def _balanced_sampling(label_list, E, Y_predicted, Y_original, window_size, n_windows, flag_replacement):
        per_label_E = {str(l): E[Y_original == l].copy() for l in label_list}
        per_label_Y_predicted = {str(l): Y_predicted[Y_original == l].copy() for l in label_list}
        per_label_Y_original = {str(l): Y_original[Y_original == l].copy() for l in label_list}

        n_samples_per_label = window_size // len(label_list)
        n_residual_samples = window_size % len(label_list)

        E_windows, Y_predicted_windows, Y_original_windows = [], [], []

        for _ in range(n_windows):
            E_window_list, Y_predicted_window_list, Y_original_window_list = [], [], []

            for l in label_list:
                m_l = len(per_label_E[str(l)])
                if m_l == 0:
                    print(f"Warning: No samples available for label {l}. Skipping.")
                    continue
                n_samples = n_samples_per_label
                if m_l < n_samples and not flag_replacement:
                    print(f"Warning: Insufficient samples ({m_l}) for label {l} with n_samples={n_samples} and no replacement. Using all available.")
                    n_samples = m_l

                try:
                    l_idxs = np.random.choice(m_l, n_samples, replace=flag_replacement)
                except ValueError as e:
                    print(f"Error sampling label {l}: {e}. Using all available samples.")
                    l_idxs = np.arange(m_l)

                E_l_window = per_label_E[str(l)][l_idxs]
                Y_predicted_l_window = per_label_Y_predicted[str(l)][l_idxs]
                Y_original_l_window = per_label_Y_original[str(l)][l_idxs]

                E_window_list.extend(E_l_window.tolist())
                Y_predicted_window_list.extend(Y_predicted_l_window.tolist())
                Y_original_window_list.extend(Y_original_l_window.tolist())

                if not flag_replacement:
                    per_label_E[str(l)] = np.delete(per_label_E[str(l)], l_idxs, 0)
                    per_label_Y_predicted[str(l)] = np.delete(per_label_Y_predicted[str(l)], l_idxs, 0)
                    per_label_Y_original[str(l)] = np.delete(per_label_Y_original[str(l)], l_idxs, 0)

            if n_residual_samples != 0:
                count_residual = 0
                while count_residual < n_residual_samples:
                    random_idx_l = np.random.choice(len(label_list), 1, replace=True)[0]
                    random_l = label_list[random_idx_l]
                    m_l = len(per_label_E[str(random_l)])
                    if m_l == 0:
                        continue
                    idx = np.random.choice(m_l, 1, replace=False)

                    E_l_window = per_label_E[str(random_l)][idx]
                    Y_predicted_l_window = per_label_Y_predicted[str(random_l)][idx]
                    Y_original_l_window = per_label_Y_original[str(random_l)][idx]

                    E_window_list.extend(E_l_window.tolist())
                    Y_predicted_window_list.extend(Y_predicted_l_window.tolist())
                    Y_original_window_list.extend(Y_original_l_window.tolist())

                    count_residual += 1
                    if not flag_replacement:
                        per_label_E[str(random_l)] = np.delete(per_label_E[str(random_l)], idx, 0)
                        per_label_Y_predicted[str(random_l)] = np.delete(per_label_Y_predicted[str(random_l)], idx, 0)
                        per_label_Y_original[str(random_l)] = np.delete(per_label_Y_original[str(random_l)], idx, 0)

            E_windows.append(np.array(E_window_list))
            Y_predicted_windows.append(np.array(Y_predicted_window_list))
            Y_original_windows.append(np.array(Y_original_window_list))

        return E_windows, Y_predicted_windows, Y_original_windows

    @staticmethod
    def _shuffle_dataset(E, Y_predicted, Y_original):
        p = np.random.permutation(len(E))
        return E[p], Y_predicted[p], Y_original[p]
