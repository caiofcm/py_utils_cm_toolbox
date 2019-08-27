import numpy as np
from collections.abc import Iterable
from utils_cm_toolbox.misc import if_decorator
try:
    import numba
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

############################################
############################################
#	 REGULARIZATION FUNCTION ARCTAN
############################################
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
############################################
#	 RUNGE KUTTA 3ORDER LOW MEMORY
############################################
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
############################################
#	 B-SPLINE IMPLEMENTATION
#Ref: http://www.lce.hut.fi/teaching/S-114.1100/lect_6.pdf p 29
############################################
############################################

@if_decorator(USE_NUMBA, numba.njit)
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

@if_decorator(USE_NUMBA, numba.njit)
def b_spline_intervals(t):
    n = len(t) - 1 #n: number of intervals
    h = np.empty(n+2)
    h[1:-1] = t[1:] - t[0:-1]
    h[0] = h[1]
    h[-1] = h[-2]
    return h

@if_decorator(USE_NUMBA, numba.njit)
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

############################################
############################################
#	 FSOLVE WITH JACOBIAN
############################################
############################################

@numba.njit(cache=True)
def root_finding_newton(fun, J, x, eps, max_iter, args):
    """
    Solve nonlinear system fun(x)=0 by Newton's method.
    J is the Jacobian of fun(x). Both fun(x) and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < eps.
    """
    F_value = fun(x, *args)
    F_value_ = F_value.reshape((-1,1))
    F_norm = np.linalg.norm(F_value, 2)  # l2 norm of vector
    iteration_counter = 0
    while abs(F_norm) > eps and iteration_counter < max_iter:
        delta = np.linalg.solve(J(x, args), -F_value_)

        for i in range(x.size): #wtf numba!?!?!
            x[i] += delta[i,0]

        F_value = fun(x, *args)
        F_value_ = F_value.reshape((-1,1))
        F_norm = np.linalg.norm(F_value, 2)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
        raise ValueError('Maximum iteration reached in newton root finding!')
    return x, iteration_counter

@numba.njit(cache=True)
def numeric_jacobian(fun, x, diff_eps, args):
    J = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += diff_eps
        x2[i] -= diff_eps
        f1 = fun(x1, *args)
        f2 = fun(x2, *args)
        J[:, i] = (f1 - f2) / (2 * diff_eps)

    return J

def create_jacobian(fun):

    @numba.njit()
    def numba_J(x, args):
        return numeric_jacobian(fun, x, 1e-8, args)
    return numba_J


@numba.njit(cache=True)
def root_finding_newton_previously(fun, J, x, eps, max_iter, args):
    """
    Solve nonlinear system fun(x)=0 by Newton's method.
    J is the Jacobian of fun(x). Both fun(x) and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < eps.
    """
    F_value = fun(x, args)
    F_value_ = F_value.reshape((-1,1))
    F_norm = np.linalg.norm(F_value, 2)  # l2 norm of vector
    iteration_counter = 0
    while abs(F_norm) > eps and iteration_counter < max_iter:
        delta = np.linalg.solve(J(x, args), -F_value)
        x = x + delta
        F_value = fun(x, args)
        F_value_ = F_value.reshape((-1,1))
        F_norm = np.linalg.norm(F_value, 2)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
        raise ValueError('Maximum iteration reached in newton root finding!')
    return x, iteration_counter

@numba.njit #(cache=True)
def brents(f, x0, x1, max_iter=50, tolerance=1e-5):
    """Brents Method for onedimensional fsolve

    Ref: https://nickcdryan.com/2017/09/13/root-finding-algorithms-in-python-line-search-bisection-secant-newton-raphson-boydens-inverse-quadratic-interpolation-brents/
    
    Parameters
    ----------
    f : callable
        Function
    x0 : float
        Lower Bound
    x1 : float
        Upper Bound
    max_iter : int, optional
        Maximum number of iteration, by default 50
    tolerance : float, optional
        by default 1e-5
    
    Returns
    -------
    Tuple
        (x solution, number of steps)
    """
 
    fx0 = f(x0)
    fx1 = f(x1)
 
    assert (fx0 * fx1) <= 0, "Root not bracketed" 
 
    if abs(fx0) < abs(fx1):
        x0, x1 = x1, x0
        fx0, fx1 = fx1, fx0
 
    x2, fx2 = x0, fx0
 
    d = np.nan
    mflag = True
    steps_taken = 0
 
    while steps_taken < max_iter and abs(x1-x0) > tolerance:
        fx0 = f(x0)
        fx1 = f(x1)
        fx2 = f(x2)
 
        if fx0 != fx2 and fx1 != fx2:
            L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
            L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
            L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
            new = L0 + L1 + L2
 
        else:
            new = x1 - ( (fx1 * (x1 - x0)) / (fx1 - fx0) )
 
        tt1 = (new < ((3 * x0 + x1) / 4) or new > x1)
        tt2 = (mflag == True and (abs(new - x1)) >= (abs(x1 - x2) / 2))
        tt3 = (mflag == False and (abs(new - x1)) >= (abs(x2 - d) / 2))
        tt4 = (mflag == True and (abs(x1 - x2)) < tolerance)
        tt5 = (mflag == False and (abs(x2 - d)) < tolerance)
        if (tt1 or
            tt2 or
            tt3 or
            tt4 or
            tt5):
            new = (x0 + x1) / 2
            mflag = True
 
        else:
            mflag = False
 
        fnew = f(new)
        d, x2 = x2, x1
 
        if (fx0 * fnew) < 0:
            x1 = new
        else:
            x0 = new
 
        if abs(fx0) < abs(fx1):
            x0, x1 = x1, x0
 
        steps_taken += 1
 
    return x1, steps_taken

############################################
############################################
#	 INTEGRATION
############################################
############################################

@if_decorator(USE_NUMBA, numba.njit)
def integr_simpson(fun, xa, xb, args):
    fa = fun(xa, args)
    fb = fun(xb, args)
    fm = fun(0.5*(xa+xb), args)
    I = (xb-xa)/2.0*(fa + 4.0*fm + fb)
    return I


if __name__ == "__main__":
    eval_regularization()