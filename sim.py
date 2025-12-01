import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from typing import Literal


pi = np.pi

def sim_func(X, U, mu, R = None):
    r = X[0]
    r_dot = X[1]
    theta = X[2]
    theta_dot = X[3]

    X_dot = np.zeros(4)
    
    u1, u2 = U(X, R)
   

    X_dot[0] = r_dot
    X_dot[1] = r*theta_dot**2 - mu/(r**2) + u1
    X_dot[2] = theta_dot
    X_dot[3] = -2*theta_dot*r_dot/r + u2/r

    return X_dot

def sim_func_linear(X, U, mu, r0, R = None):
    X_r = X.reshape((4,1))

    A = np.array([[           0, 1, 0,                0],
                  [3*mu/(r0**3), 0, 0, 2*np.sqrt(mu/r0)],
                  [ 0, 0, 0, 1],
                  [0, -2*np.sqrt(mu/r0**5), 0, 0 ]])
    
    B = np.array([[0,0],
                  [1,0],
                  [0,0],
                  [0,1/r0]])
    u = U(X)
    u = u.reshape((2,1))

    X_dot = A@X_r + B@u
    return X_dot.reshape(X.shape)


def contr_func(X, R = None):
    return np.array([0, 0])


def prop_orbit(X0, tspan, U_func, mu, num_eval = 200):

    int_func = lambda t, X: sim_func(X, U_func, mu)
    teval = np.linspace(tspan[0], tspan[1], num_eval)
    sol = sci.integrate.solve_ivp(int_func, tspan, X0, t_eval= teval)

    return [sol.y, sol.t]

def prop_orbit_linear(X0, tspan, U_func, mu, r0, num_eval = 200):

    int_func = lambda t, X: sim_func_linear(X, U_func, mu, r0)
    teval = np.linspace(tspan[0], tspan[1], num_eval)
    sol = sci.integrate.solve_ivp(int_func, tspan, X0, t_eval= teval)

    return [sol.y, sol.t]


def plot_XY(X_sol):
    r_sol = X_sol[0,:]
    theta_sol = X_sol[2, :]

    x_pos_sol = r_sol * np.cos(theta_sol)
    y_pos_sol = r_sol * np.sin(theta_sol)

    fig, ax = plt.subplots()

    # Trajectory line
    ax.plot(x_pos_sol, y_pos_sol, linewidth=2)

    # Start (green) and end (red)
    ax.scatter(x_pos_sol[0],  y_pos_sol[0],  color='green', s=50, label='Start')
    ax.scatter(x_pos_sol[-1], y_pos_sol[-1], color='red',   s=50, label='End')
    ax.scatter(0,0, c="k")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectory')
    ax.axis('equal')
    ax.legend()

    return fig


def plot_state(X_sol, t_sol, diff=False, X_ref=None, t_ref=None):

    # ---------------------------------
    # State over time
    # ---------------------------------
    fig, axs = plt.subplots(4, 1, figsize=(9, 7), sharex=True)

    labels = ['r', 'r_dot', 'theta', 'theta_dot']

    for i in range(4):
        axs[i].plot(t_sol, X_sol[i, :], label='state')

        if diff:
            if X_ref is None or t_ref is None:
                raise ValueError("diff=True requires X_ref and t_ref.")
            if X_ref.shape != X_sol.shape:
                raise ValueError("X_sol and X_ref must have same shape for diff plots.")

            axs[i].plot(t_sol, X_sol[i, :] - X_ref[i, :], '--', label='difference')

        axs[i].set_ylabel(labels[i])

    axs[-1].set_xlabel("time")

    if diff:
        fig.suptitle("State Variables and Differences Over Time")
        axs[0].legend()
    else:
        fig.suptitle("State Variables Over Time")

    plt.tight_layout()
    return fig


def polar_to_cart_inertial(X_sol):
    r_sol = X_sol[0,:]
    theta_sol = X_sol[2, :]

    r_dot_sol = X_sol[1,:]
    theta_dot_sol = X_sol[3,:]

    x_pos_sol = r_sol * np.cos(theta_sol)
    y_pos_sol = r_sol * np.sin(theta_sol)

    x_dot_sol = r_dot_sol*np.cos(theta_sol) - r_sol*theta_dot_sol*np.sin(theta_sol)
    y_dot_sol = r_dot_sol*np.sin(theta_sol) + r_sol*theta_dot_sol*np.cos(theta_sol)

    return np.array([x_pos_sol, y_pos_sol, x_dot_sol, y_dot_sol])

def orbit_diff(X_ref, t_ref, X_perturb, t_perturb):

    if np.array_equal(t_ref, t_perturb):
        return t_ref, X_perturb - X_ref

    if (X_ref.shape[0] != X_perturb.shape[0]):
        raise ValueError

    t_start = max( min(t_ref), min(t_perturb))
    t_end = min(max(t_ref), max(t_perturb))

    dt = np.mean(np.diff(t_ref))
    t_interp = np.arange(t_start, t_end, dt)

    diff_data = np.zeros((X_ref.shape[0], len(t_interp)))
    for i in range(X_ref.shape[0]):
        e_ref = X_ref[i, :]
        e_p = X_perturb[i, :]
        e_ref_interp = np.interp(t_interp, t_ref, e_ref)
        e_p_interp = np.interp(t_interp, t_perturb, e_p)

        e_diff = e_p_interp - e_ref_interp
        diff_data[i, :] = e_diff


    return t_interp, diff_data


def RI_rot_from_XY(coord_xy):

    coord_xy_r = coord_xy.reshape(4)

    v_vec = coord_xy_r[2:]

    v_n = v_vec/np.linalg.norm(v_vec)
    rot_mat_partial = np.array([[v_n[1], v_n[0]],
                                [-v_n[0], v_n[1]]])
    
    
    rot_mat = np.zeros((4,4))
    rot_mat[:2,:2] = rot_mat_partial
    rot_mat[2:,2:] = rot_mat_partial
    
    return rot_mat





