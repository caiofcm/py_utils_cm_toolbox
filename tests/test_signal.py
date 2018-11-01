import numpy as np
from utils_cm_toolbox import utils_signal
from matplotlib import pyplot as plt

PLOT = True

def test_generate_gbn():
    low = np.array([1.0, -10.0])
    upp = np.array([5.0, 2.0])
    tspan = np.linspace(0.0, 10, 101)
    prob_change = np.array([0.95, 0.95])
    num_samples_min = [2, 10]
    args = (tspan, low, upp, prob_change, num_samples_min)
    out_signal = utils_signal.generate_gbn(*args)
    if PLOT:
        plt.figure()
        plt.plot(tspan, out_signal, '.-k', lw=2, label='signal')
        plt.legend()
        plt.show()
    assert(out_signal[0, 0] == low[0])
    assert(out_signal[0, 1] == low[1])