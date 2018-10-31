import numpy as np
from utils_cm_toolbox import utils_signal

def test_generate_gbn():
    low = np.array([1.0])
    upp = np.array([5.0])
    tspan = np.linspace(0.0, 10, 11)
    prob_change = np.array([0.95])
    num_samples_min = 2
    args = (tspan, low, upp, prob_change, num_samples_min)
    utils_signal.generate_gbn(*args)
    assert(True)