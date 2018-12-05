import pytest
import numpy as np
from src.parameter_config.ParamConfig import ParamConfig, SingleParam
from src.suggestors import RandomSearch, ZoomRandomSearch
from src.suggestors.SuggestorBase import ParamLog


def test_random_search():

    # Initializing param configuration
    param_config = (
        SingleParam("learning_rate", output_type="double", value_range=(0.0001, 0.01), scaling="log"),
        SingleParam("dropout_l", output_type="double", value_range=(0, 0.7), scaling="incremental", increment=0.05),
        SingleParam("hidden_size_l", "integer", (100, 1000), "incremental", 50),
        SingleParam("batch_size", output_type="discrete", value_range=[64, 128, 256, 512, 1024])
    )

    p_configurer = ParamConfig()

    # Make rescaler functions
    functions, names = p_configurer.make_rescale_dict(param_config)

    # Make param log
    param_log = ParamLog(len(functions), param_descriptions=names)

    # Initialize RandomSearch with rescaler functions
    rand_search = RandomSearch(functions, names, param_log=param_log)

    # Making parameter suggestions
    parameter_suggestions = []
    for i in range(0, 1000):
        print("Suggesting parameter: {}".format(i))
        parameter_suggestions.append(rand_search.suggest_parameters()[1])

    parameter_suggestions = np.vstack(parameter_suggestions)

    assert not np.any(parameter_suggestions[:, 0] < 0.0001)
    assert not np.any(parameter_suggestions[:, 0] > 0.01)

    assert not np.any(parameter_suggestions[:, 1] < 0)
    assert not np.any(parameter_suggestions[:, 1] > 0.7)

    assert not np.any(parameter_suggestions[:, 2] < 100)
    assert not np.any(parameter_suggestions[:, 2] > 1000)

    assert np.all(np.isin(parameter_suggestions[:, 3], [64, 128, 256, 512, 1024]))


def test_zoom_random_search():
    # Initializing param configuration
    param_config = (
        SingleParam("learning_rate", output_type="double", value_range=(0.0001, 0.01), scaling="log"),
        SingleParam("dropout_l", output_type="double", value_range=(0, 0.7), scaling="incremental", increment=0.05),
        SingleParam("hidden_size_l", "integer", (100, 1000), "incremental", 50),
        SingleParam("batch_size", output_type="discrete", value_range=[64, 128, 256, 512, 1024])
    )

    p_configurer = ParamConfig()

    # Make rescaler functions
    functions, names = p_configurer.make_rescale_dict(param_config)

    # Make param log
    param_log = ParamLog(len(functions), param_descriptions=names)

    # Initialize RandomSearch with rescaler functions
    rand_search = ZoomRandomSearch(trials_per_zoom=500, rescale_functions=functions, param_names=names,
                                   param_log=param_log, n_eval_trials=5)

    # Making parameter suggestions
    parameter_suggestions = []
    for i in range(0, 10000):
        print("Suggesting parameter: {}".format(i))
        params = rand_search.suggest_parameters()[1]
        param_log.log_score(score=score_param(params))
        if i > 9499:
            parameter_suggestions.append(params)

    parameter_suggestions = np.vstack(parameter_suggestions)

    assert not np.any(parameter_suggestions[:, 0] < 0.004)
    assert not np.any(parameter_suggestions[:, 0] > 0.008)

    assert not np.any(parameter_suggestions[:, 1] < 0.5)
    assert not np.any(parameter_suggestions[:, 1] > 0.6)

    assert not np.any(parameter_suggestions[:, 2] < 200)
    assert not np.any(parameter_suggestions[:, 2] > 400)

    assert np.all(np.isin(parameter_suggestions[:, 3], [512, 1024]))


def score_param(parameter_suggestion):
    score = 0

    if (not parameter_suggestion[0] < 0.004) and (not parameter_suggestion[0] > 0.008):
        score += 1

    if (not parameter_suggestion[1] < 0.5) and (not parameter_suggestion[1] > 0.6):
        score += 1

    if (not parameter_suggestion[2] < 200) and (not parameter_suggestion[2] > 400):
        score += 1

    if np.isin(parameter_suggestion[3], [512, 1024]):
        score += 1

    return score
