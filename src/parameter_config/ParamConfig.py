from src.parameter_config.rescalers import *


class ParamConfig:

    def __init__(self):
        self._func_types = {
            "double": self._double,
            "integer": self._integer,
            "discrete": self._discrete
        }

    def make_rescale_dict(self, params):

        rescaler_functions = []
        names = []
        for single_param in params:
            rescaler_functions.append(self._get_rescale_function(single_param))
            names.append(single_param.name)

        return rescaler_functions, names

    def _get_rescale_function(self, single_param):
        func_type = single_param.output_type

        if func_type in self._func_types.keys():
            args = [single_param.value_range]

            # Adding scaling if existing
            if single_param.scaling is not None:
                args.append(single_param.scaling)

            # Adding increment if existing
            if single_param.increment is not None:
                args.append(single_param.increment)

            rescaler = self._func_types[func_type](*args)
        else:
            raise ValueError("The given func type \"{}\" is not supported".format(func_type))

        return rescaler

    def _integer(self, min_max_range, scaling=None, incremental=None):
        min_range = min(min_max_range)
        max_range = max(min_max_range)

        if scaling is None:
            scaling = "incremental"

        if scaling == "incremental":
            if incremental is None:
                incremental = 1

            return incremental_rescaler(incremental, (min_range, max_range))

        elif scaling == "log":
            return log_rescaler((min_range, max_range), int_log=True)

        else:
            raise ValueError("The given scaling \"{}\" is not supported".format(scaling))

    def _double(self, min_max_range, scaling=None, incremental=None):
        min_range = min(min_max_range)
        max_range = max(min_max_range)

        if scaling is None:
            scaling = "incremental"

        if scaling == "incremental":
            if incremental is None:
                incremental = (max_range - min_range) / 100

            return incremental_rescaler(incremental, (min_range, max_range))

        elif scaling == "log":
            return log_rescaler((min_range, max_range))

        else:
            raise ValueError("The given scaling \"{}\" is not supported".format(scaling))

    def _discrete(self, discrete_values):
        min_range = 0
        max_range = len(discrete_values)-1

        internal_rescaler = incremental_rescaler(1, (min_range, max_range))

        def rescaler(value):
            idx = internal_rescaler(value)
            return discrete_values[int(round(idx))]

        return rescaler


class SingleParam:

    def __init__(self, name, output_type, value_range, scaling=None, increment=None):

        self.name = name
        self.value_range = value_range
        self.output_type = output_type
        self.scaling = scaling

        if scaling == "increment" and increment is None:
            self.increment = 1
        else:
            self.increment = increment


if __name__ == "__main__":

    param_config = {
        "learning_rate": ("double", (0.0001, 0.01), "log"),
        "dropout_1": ("double", (0, 0.7), "incremental", 0.05),
        "hidden_size_1": ("integer", (100, 1000), "incremental", 50),
        "batch_size": ("discrete", [64, 128, 256, 512, 1024])
    }

    p_config = ParamConfig()
    rescaler_functions = p_config.make_rescale_dict(param_config)
    print(rescaler_functions.values())
    print("Learning rate rescaler input 0.5")
    print(rescaler_functions["batch_size"](0.55))
