import numpy as np

def generate_gbn(tspan: np.ndarray, 
        lower_vals: np.ndarray,
        upper_vals: np.ndarray,
        prob_change: np.ndarray,
        number_of_samples_min: (list, np.ndarray)) -> np.ndarray:
    previous_signal = lower_vals.copy()
    n = len(previous_signal)
    last_t_change = np.zeros(n)
    nt = len(tspan)
    dt = tspan[1] - tspan[0]
    signal = np.zeros((nt, n))
    np.random.seed()
    for k in range(nt):
        actual_time = tspan[k]
        signal_aux = np.zeros(n)
        R = np.random.uniform(size=n)
        for i in range(n):
            if R[i] < prob_change[i]:
                if int((actual_time - last_t_change[i])/dt) < number_of_samples_min[i]:
                    if previous_signal[i] == upper_vals[i]:
                        signal_aux[i] = upper_vals[i]
                    else:
                        signal_aux[i] = lower_vals[i]
                else:
                    if previous_signal[i] == upper_vals[i]:
                        signal_aux[i] = lower_vals[i]
                    else:
                        signal_aux[i] = upper_vals[i]
                    last_t_change[i] = actual_time
            else:
                if previous_signal[i] == upper_vals[i]:
                    signal_aux[i] = upper_vals[i]
                else:
                    signal_aux[i] = lower_vals[i]
        signal[k, :] = signal_aux
        previous_signal = signal_aux.copy()
    
    return signal