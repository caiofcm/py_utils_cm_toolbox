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
@if_decorator(USE_NUMBA, numba.njit)
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

@if_decorator(USE_NUMBA, numba.njit)
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

import matplotlib.pyplot as plt
def eval_regularization():
    plt.figure()
    xspan = np.linspace(0, 200, 501)
    Ai = [10.0, 5.0]
    xEi = [25.0, 100.0]
    psi = [0.5, 0.05]
    y = regularization_mult_func(xspan, 0.0, Ai, xEi, psi)
    plt.plot(xspan, y)
    plt.title('Multiple step reg function for Aggregation Kernel')
    plt.show()    
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


############################################
###########################################
#
#	 B-SPLINE IMPLEMENTATION
#Ref: http://www.lce.hut.fi/teaching/S-114.1100/lect_6.pdf p 29
###########################################
############################################

@numba.njit()
def b_spline_coeff(t: np.ndarray, y: np.ndarray):
    """Calculates b-spline coefficients
    
    Parameters
    ----------
    t : np.ndarray [n+1]
        independent variable points
    y : np.ndarray [n+1]
        dependent variable points
    
    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        [0]: Coefficients array [n+2]
        [1]: Adjusted interval array [n+2]
    """
    n = len(t) - 1 #n: number of intervals
    h = b_spline_intervals(t)

    # Determine A0:
    delta = -1.0
    gamma = 2.0 * y[0]
    p = delta*gamma
    q = 2.0
    # r = h[2:]/h[]

    for i in np.arange(1, n + 1):
        r = h[i+1]/h[i]
        delta = -r*delta
        gamma = -r*gamma + (r+1.0)*y[i]
        p = p + gamma*delta
        q = q + delta*delta
    
    a = np.empty(n + 2)
    a[0]=-p/q
    # Determine other coefficients Ai:
    for i in np.arange(1, n + 2):
        a[i] = ((h[i-1]+h[i])*y[i-1]-h[i]*a[i-1])/h[i-1]
    
    return a, h

@numba.njit()
def b_spline_intervals(t):
    n = len(t) - 1 #n: number of intervals
    h = np.empty(n+2)
    h[1:-1] = t[1:] - t[0:-1]
    h[0] = h[1]
    h[-1] = h[-2]
    return h

@numba.njit()
def b_spline_eval(t: np.ndarray, a: np.ndarray, h: np.ndarray, x: float):
    """Evaluate the b-spline
    
    Parameters
    ----------
    t : np.ndarray
        independent points 
    a : np.ndarray
        b-spline coefficients
    h : np.ndarray
        modified interval array
    x : float
        defined point to interpolate
    
    Returns
    -------
    float, nd.array
        Interpolated value (values)
    """

    # Check in which interval x lies
    n = len(t) - 1
    for i in np.arange(n - 1, 0 - 1, -1):
        if (x - t[i] >= 0.):
            break

    # /* Evaluate S(x) */
    # h = t[1:] - t[0:-1]

    i = i+1
    d = (a[i+1]*(x-t[i-1])+a[i]*(t[i]-x+h[i+1]))/(h[i]+h[i+1])
    e = (a[i]*(x-t[i-1]+h[i-1])+a[i-1]*(t[i-1]-x+h[i]))/(h[i-1]+h[i])
    result = (d*(x-t[i-1])+e*(t[i]-x))/h[i]

    return result



if __name__ == "__main__":
    eval_regularization()