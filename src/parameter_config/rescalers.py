import numpy as np


def create_min_max_rescaler(original_min_max, target_min_max):
    """
    Min max rescaler creator. Creates a rescaler that rescales from target range into original range.
    :param original_min_max: tuple of floats, the range of values that is rescaled to
    :param target_min_max: tuple of floats, the range of value used to input to the rescaler
    :return: function, the created min_max_rescaler
    """
    o_min = min(original_min_max)
    o_max = max(original_min_max)
    t_min = min(target_min_max)
    t_max = max(target_min_max)

    def min_max_rescaler(x):
        if (t_min > x) or (x > t_max):
            raise ValueError("x is not within the min and max range of the rescaler. "
                             "\n min/max: {}-{}"
                             "\n x: {}".format(t_min, t_max, x))

        return ((o_max - o_min) * (x - t_min)) / (t_max - t_min) + o_min

    return min_max_rescaler


def incremental_rescaler(incremental, min_max_range):
    """
    Rescaler creator that creates a rescaler that rescales into a range with increments.
    :param incremental: float, the rescaled value is rounded to nearest multiple of incremental
    :param min_max_range: tuple of floats, min and max of range to rescale into
    :return: function, the created rescaler
    """
    min_range = min(min_max_range)
    max_range = max(min_max_range)

    # create min_max_scaler
    min_max_scaler = create_min_max_rescaler((min_range, max_range), (0, 1))

    def rescaler(value):
        # rescale value
        rescaled_value = min_max_scaler(value)
        # Round to neares incremental value and return it
        result = rescaled_value + incremental / 2
        result -= result % incremental

        # Floating point inprecision makes it necessary to create checks for small misalingments
        # To avoid breaking the unit testing, the difference has to be less than 0.1 % of increment
        if result > max_range:
            diff = result - max_range
            if diff < (0.001 * incremental):
                result = max_range

        if result < min_range:
            diff = min_range - result
            if diff < (0.001 * incremental):
                result = min_range

        return result

    return rescaler


def log_rescaler(min_max_range, int_log=False):
    """
    Rescaler creator that creates a resclaer which rescales into a range with logarithmic increment.
    :param min_max_range: tuple of float, min and max range to rescale into
    :return: function, the created rescaler
    """
    min_range = min(min_max_range)
    max_range = max(min_max_range)

    # calculating limits
    a = np.log10(min_range)
    b = np.log10(max_range)

    # create the rescaler for log
    min_max_log_scale = create_min_max_rescaler((a, b), (0, 1))

    if int_log:
        if min_range < 1 or max_range < 1:
            raise ValueError("With int_log min_range and max_range can not be less than 1\n"
                             "min_range: {}, max_range: {]".format(min_range, max_range))

        def rescaler(value):
            # calculate log rescale value
            r = min_max_log_scale(value)
            log_rescale_value = 10 ** r

            return int(round(log_rescale_value))
    else:
        def rescaler(value):
            # calculate log rescale value
            r = min_max_log_scale(value)
            log_rescale_value = 10 ** r

            return log_rescale_value

    return rescaler
