import numpy as np
from .SuggestorBase import SuggestorBase


class ZoomRandomSearch(SuggestorBase):

    def __init__(self, trials_per_zoom, rescale_functions, param_names, param_log, n_eval_trials=None):
        # setting super init
        super(ZoomRandomSearch, self).__init__(rescale_functions, param_names=param_names, param_log=param_log)

        # Starting zoom
        self.trials_per_zoom = trials_per_zoom
        self.current_trial_in_zoom = 0
        self.upper_bounds = np.ones((self.n_param, ))
        self.lower_bounds = np.zeros((self.n_param, ))
        self.difference = self.upper_bounds-self.lower_bounds

        # Setting amount of top suggestions to look at when zooming
        if n_eval_trials is None:
            self.n_eval_trials = int(trials_per_zoom*0.1)
        else:
            self.n_eval_trials = n_eval_trials

    def calc_zoom_bounds(self):
        score = self.param_log.get_score()
        unscaled_params = self.param_log.get_unscaled_params()

        # Finding idx of top performing suggestions
        idx = np.argpartition(score, -self.n_eval_trials, axis=None)[-self.n_eval_trials:]

        # Getting values of top performing suggestions
        best_suggestions = unscaled_params[idx]

        # Finding new bounds
        self.upper_bounds = np.amax(best_suggestions, axis=0)
        self.lower_bounds = np.amin(best_suggestions, axis=0)
        self.difference = self.upper_bounds - self.lower_bounds

    def calculate_suggestion(self):
        # If done with this trial range, zoom onto new range
        if self.current_trial_in_zoom == self.trials_per_zoom:
            self.calc_zoom_bounds()
            self.current_trial_in_zoom = 0

        dict_param = None
        real = None

        unique = False
        while not unique:
            dict_param, real, unscaled = self._random_param_sample()

            unique = self.param_log.log_param(real, unscaled, np.array([0]))

        self.current_trial_in_zoom += 1
        return dict_param, real

    def _random_param_sample(self):
        real_parameters = []
        dict_param = {}
        # random are multiplied by range and lower bounds is added to get a value in the new range
        unscaled_parameters = np.multiply(np.random.random_sample(self.n_param), self.difference)+self.lower_bounds

        for i, func in enumerate(self._rescale_functions):
            param_value = func(unscaled_parameters[i])
            real_parameters.append(param_value)
            dict_param[self.param_names[i]] = param_value

        return dict_param, np.asarray(real_parameters), unscaled_parameters
