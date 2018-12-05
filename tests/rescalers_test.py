import pytest
import numpy as np
import src.parameter_config.rescalers as res


def test_create_min_max_rescaler():

    og_min = 40
    og_max = 120
    t_min = 0
    t_max = 1

    rescaler = res.create_min_max_rescaler((og_max, og_min), (t_min, t_max))

    assert rescaler(0) == 40
    assert rescaler(1) == 120


def test_incremental_rescaler():
    increment = 5
    min_test = 0
    max_test = 100

    rescaler = res.incremental_rescaler(incremental=increment, min_max_range=(min_test, max_test))

    assert rescaler(0.09) == 10


def test_log_rescaler():

    min_test = 0.01
    max_test = 10

    rescaler = res.log_rescaler((min_test, max_test))

    assert rescaler(0) == 0.01
    assert rescaler(1) == 10

