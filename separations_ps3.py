import fire
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


# Global Variables for q6_8
x = np.array([0.033, 0.072, 0.117, 0.171])
partial_p = np.array([30.00, 62.80, 85.4, 103.0]) #torr
P = 760 # torr
y = partial_p / P
Y = y / (1-y)
X = x / (1-x)
# Equilibrium Curve
X = np.insert(X, 0, 0)
Y = np.insert(Y, 0, 0)
print(Y)
print(X)
X_smooth = np.linspace(X.min(), X.max(), 1000)
spline = make_interp_spline(X, Y)
Y_smooth = spline(X_smooth)
Y1 = 0.0088


def q6_8_a():
    plt.plot(X_smooth, Y_smooth)
    # Operating Line
    spline_derivative = spline.derivative()
    tangent_x = 0.065
    slope = spline_derivative(tangent_x)
    x0, y0 = X[0], Y1
    tangent_y = spline(tangent_x)
    tangent_line_x = np.linspace(X.min(), X.max(), 300)
    tangent_line_y = slope * (tangent_line_x - x0) + y0
    plt.plot(tangent_line_x, tangent_line_y, '--', color='red',
             label=f'Tangent Line (slope = {slope:.2f})')
    plt.scatter(tangent_x, tangent_y, color='green', zorder=5)
    plt.text(tangent_x, tangent_y, f'Tangent @ ({tangent_x:.2f}, {tangent_y:.2f})')
    plt.ylabel("Y")
    plt.legend()
    plt.show()
    return


def q6_8_b():
    plt.plot(X_smooth, Y_smooth)
    # Operating Line
    min_slope = 1.07
    fixed_slope = 1.25 * min_slope
    x0 = X[0]
    linear_line_x = np.linspace(X.min(), X.max(), 300)
    linear_line_y = fixed_slope * (linear_line_x - x0) + Y1
    plt.plot(linear_line_x, linear_line_y, '--',
             label=f'Fixed Line (slope = {fixed_slope:.2f})', color='red')
    plt.xlabel("X")
    plt.legend()
    plt.ylabel("Y")
    plt.show()
    return


def q_6_10():
    # Operating Line
    X1, Y0 = (0.0217, 0)
    XN1, YN = (0.087, 0.0309)
    m = (YN - Y0) / (XN1 - X1)
    b = Y0 - m * X1
    x_line = np.linspace(X1, XN1, 300)
    y_line = m * x_line + b
#    plt.plot(x_line, y_line, label=f"Line through ({X1}, {Y0}) and ({XN1}, {YN}), slope = {m:.4f}")

    # Operating Line 2
    X1_2, Y0_2 = (0.013, 0)
    XN1_2, YN_2 = (0.087, 0.0309)
    b_2 = Y0_2 - m * X1_2
    x_line_2 = np.linspace(X1_2, XN1_2, 300)
    y_line_2 = m * x_line_2 + b_2
    plt.plot(x_line_2, y_line_2, label=f"Line through ({X1_2}, {Y0_2}) and ({XN1_2}, {YN_2}), slope = {m:.4f}")

    # Equilibrium Curve
    x_values = np.linspace(0, 0.10, 400)
    y_values = x_values / (x_values + 2)
    plt.plot(x_values, y_values, label=r"$y = \frac{x}{x+2}$", color="orange")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
    return

def main():
#    q6_8_a()
#    q6_8_b()
    q_6_10()
    return

if __name__ == "__main__":
    fire.Fire(main())
