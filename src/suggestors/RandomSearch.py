import numpy as np
from .SuggestorBase import SuggestorBase


class RandomSearch(SuggestorBase):

    def calculate_suggestion(self):
        unique = False
        dict_param = None
        real = None

        while not unique:
            dict_param, real, unscaled = self._random_param_sample()

            unique = self.param_log.log_param(real, unscaled, np.array([0]))

        return dict_param, real

    def _random_param_sample(self):
        real_parameters = []
        dict_param = {}
        unscaled_parameters = np.random.random_sample(self.n_param)

        for i, func in enumerate(self._rescale_functions):
            param_value = func(unscaled_parameters[i])
            real_parameters.append(param_value)
            dict_param[self.param_names[i]] = param_value

        return dict_param, np.asarray(real_parameters), unscaled_parameters
