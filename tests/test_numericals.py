import pytest
import numpy as np
import numba
from utils_cm_toolbox import numericals
import matplotlib.pyplot as plt

NUM_POINTS_REF = 10
T_REF = np.linspace(0., 10., NUM_POINTS_REF + 1)
Y_REF = np.array([10., 9., 11., 10., 5., 4., 7., 8., 5., 9., 11.])

# def test_b_spline_coeff():
#     coefs, h = numericals.b_spline_coeff(T_REF, Y_REF)

#     assert coefs[0] == 5.0

# def test_b_spline_eval():
#     coeffs_test = np.ones(NUM_POINTS_REF + 1)
#     coeffs_test[0::2] = -1
#     x = 4.5
#     r = numericals.b_spline_eval(T_REF, coeffs_test, x)

    # assert r == 5.0

def test_b_spline():
    coefs, h = numericals.b_spline_coeff(T_REF, Y_REF)
    x = 4.5
    r = numericals.b_spline_eval(T_REF, coefs, h, x)
    assert r < 5.0 and r > 4.0


def show_plot_b_spline():
    coefs, h = numericals.b_spline_coeff(T_REF, Y_REF)
    nt = 101
    x = np.linspace(0.0, T_REF[-1], nt)
    r = np.empty_like(x)
    for i in np.arange(0, nt):
        r[i] = numericals.b_spline_eval(T_REF, coefs, h, x[i])
    plt.figure()
    plt.plot(x, r, 'o-')
    plt.plot(T_REF, Y_REF, 'dr')
    print(coefs)
    plt.show()

def case_testing_b_spline():
    coefs, h = numericals.b_spline_coeff(T_REF, Y_REF)
    coefs = np.random.random(11) * len(coefs)
    nt = 101
    x = np.linspace(0.0, T_REF[-1], nt)
    r = np.empty_like(x)
    for i in np.arange(0, nt):
        r[i] = numericals.b_spline_eval(T_REF, coefs, h, x[i])
    plt.figure()
    plt.plot(x, r, 'o-')
    plt.plot(T_REF, Y_REF, 'dr')
    print(coefs)
    plt.show()    


######
# Brents
######
@numba.njit
def f(x):
    return x**2 - 20

def test_brents():

    root, steps = numericals.brents(f, 2, 5, tolerance=10e-5)
    print("root is:", root)
    print("steps taken:", steps)
    assert(np.isclose(root, np.sqrt(20.0)))
    assert(steps < 50)

if __name__ == "__main__":
    # case_testing_b_spline()
    show_plot_b_spline()
