import numpy as np


class SuggestorBase:

    def __init__(self, rescale_functions, param_names, param_log):

        # Setting rescale function information
        self._rescale_functions = rescale_functions
        self.n_param = len(rescale_functions)
        self.param_names = param_names

        # Starting log
        self.param_log = param_log

    def suggest_parameters(self):
        return self.calculate_suggestion()

    def calculate_suggestion(self):
        raise NotImplementedError("This class is a baseclass,"
                                  " this function should be implemented in inheriting classes")


class ParamLog:

    def __init__(self, n_params, actual=None, unscaled=None, score=None, param_descriptions=None):

        self.n_params = n_params

        # Initializing logs
        check = [actual is not None, unscaled is not None, score is not None]
        if all(check):
            if not len(actual) == len(unscaled) == len(score):
                raise Exception("Parameter actual, unscaled and score must have the same amount of entries")

            self._actual_param_log = actual
            self._unscaled_param_log = unscaled
            self._score = score
        else:
            if any(check):
                raise Warning("Parameters actual, unscaled and score were not all provided. Initializing"
                              "empty log")

            self._actual_param_log = None
            self._unscaled_param_log = None
            self._score = None

        # Param descriptions
        if param_descriptions is not None:
            self.param_descriptions = param_descriptions

    def log_param(self, actual_param, unscaled_param, score):

        if self._actual_param_log is None:
            self._actual_param_log = actual_param.reshape((-1, self.n_params))
            self._unscaled_param_log = unscaled_param.reshape((-1, self.n_params))
            self._score = score.reshape((-1, 1))

            return True
        else:

            if self.find_param_log_idx(actual_param, list(range(0, self.n_params))) is None:
                self._actual_param_log = np.append(self._actual_param_log, actual_param.reshape((-1, self.n_params)), 0)
                self._unscaled_param_log = np.append(self._unscaled_param_log,
                                                     unscaled_param.reshape((-1, self.n_params)), 0)
                self._score = np.append(self._score, score.reshape((-1, 1)), 0)

                return True

        return False

    def log_score(self, score, idx=None):

        if idx is None:
            idx, _ = self._score.shape
        self._score[idx-1] = score

    def find_param_log_idx(self, real_values, column_idx):

        # Finding the columns that contains the values
        columns = self._actual_param_log[:, column_idx]
        real_values = real_values.reshape((-1, len(column_idx)))

        # Finding wether or not the param logs contain the values in the columns
        if (columns == [real_values]).all(1).any():

            # Find index of the values and return the corresponding params
            idx = np.argmax(columns == real_values, axis=1)
            return idx

        return None

    def get_actual_params(self):
        return self._actual_param_log

    def get_unscaled_params(self):
        return self._unscaled_param_log

    def get_score(self):
        return self._score
