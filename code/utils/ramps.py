import math


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = max(0.0, min(float(current), rampup_length))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))

