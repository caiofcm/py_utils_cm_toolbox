import numpy as np
from collections.abc import Iterable
from utils_cm_toolbox.misc import if_decorator
try:
    import numba
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

############################################
###########################################
#
#	 REGULARIZATION FUNCTION ARCTAN
###########################################
############################################
@if_decorator(numba.njit, USE_NUMBA)
def regularization_func(x, A0, A1, xEvent, psi):
    """Perform the regularization of a step using tanh

    Example:
        plt.figure()
        xspan = linspace(0, 5, 501)
        y = regularization_func(xspan, 2, 10, 2.0, 1e1)
        plt.plot(xspan, y)
        plt.title('simple reg function')
    """
    reg = A0 + (A1-A0)/2.0*(np.tanh((x - xEvent)*psi) + 1.0)
    return reg

def regularization_mult_func(x, A0, Aspan, xEspan, psi):
    """Perform the regularization of step train using tanh function.
    - It can be compiled with numba - Use: jit_regularization_funcs

    Inputs:
        - x: array
        - A0: scalar of initial value
        - xEspan: array of events times
        - psi: array of smoothing factor
    Example:
        plt.figure()
        xspan = linspace(0, 50, 501)
        Ai = [10, 5, 20, -40]
        xEi = [5, 20, 30, 40]
        y = regularization_mult_func(xspan, 2, Ai, xEi, 1e1)
        plt.plot(xspan, y)
        plt.title('Multiple step reg function')
        plt.show()
    """
    # reg= A0
    # Apast = A0
    # for A, xE, ps in zip(Aspan, xEspan, psi):
    #     reg = reg + regularization_func(x, 0, A-Apast, xE, ps)
    #     Apast = A

    reg= A0
    Apast = A0
    reg = np.ones_like(x) * A0
    for i, v in enumerate(xEspan):
        reg = reg + regularization_func(x, 0, Aspan[i]-Apast, v, psi[i])
        Apast = Aspan[i]
    return reg

# def jit_regularization_funcs():
#     # global regularization_func, regularization_mult_func
#     regularization_func = numba.njit(regularization_func)
#     regularization_mult_func = numba.njit(regularization_mult_func)
#     return


############################################
###########################################
#
#	 RUNGE KUTTA 3ORDER LOW MEMORY
###########################################
############################################
class rk33lm:
    A2 = -5.0/9.0
    A3= -153.0/128.0
    B1 = 1./3.
    B2 = 15./16.
    B3 = 8./15.0
def step_rk3lmMOD(t0, h, y, fun, rhs, arg = None):
    """Uses Runge Kutta solver Low Memory 3 order for the dissolution case
    where fun is rhs=fun(t, y, arg) """

    # y = y0[i0:iF].copy()
    t = t0
    rhs[:] = fun(t0, y, arg)
    k = h * rhs
    y[:] = y + rk33lm.B1*k

    tnew = t + rk33lm.B1
    rhs[:] = fun(tnew, y, arg)
    k = rk33lm.A2*k + h*rhs
    y[:] = y + rk33lm.B2*k

    # tnew = t + rk33lm.B2*(rk33lm.A2*h) #CHECK THIS HERE CHANGE FOR MOC CLASS
    tnew = t + rk33lm.B2*(h)
    rhs[:] = fun(tnew, y, arg)
    k = rk33lm.A3*k + h*rhs
    y[:] = y + rk33lm.B3*k

    return y

import matplotlib.pyplot as plt
def eval_regularization():
    jit_regularization_funcs()
    plt.figure()
    xspan = np.linspace(0, 200, 501)
    Ai = [10.0, 5.0]
    xEi = [25.0, 100.0]
    psi = [0.5, 0.05]
    y = regularization_mult_func(xspan, 0.0, Ai, xEi, psi)
    plt.plot(xspan, y)
    plt.title('Multiple step reg function for Aggregation Kernel')
    plt.show()    

if __name__ == "__main__":
    eval_regularization()