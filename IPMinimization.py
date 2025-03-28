import casadi as ca 
import matplotlib.pyplot as plt
import numpy as np
import time as time

def plot_contours(zks):
    x1 = np.arange(-5, 5, 0.02)
    x2 = np.arange(-5, 5, 0.02)
    X1, X2 = np.meshgrid(x1, x2)
    zks = [zk.full() for zk in zks]

    x1ks = [zk[0] for zk in zks]
    x2ks = [zk[1] for zk in zks]
    

    obj = (X1 - 4)**2 + (X2 - 4)**2
    eq  = np.sin(X1) - X2**2

    fig, ax = plt.subplots(figsize=(7,7))
    for i in range(len(zks)):
        ax.plot(x1ks[i:i+2], x2ks[i:i+2], marker='o', color="black")
        ax.contour(X1, X2, obj, levels=20)
        ax.contour(X1, X2, eq, levels=[0], colors='red', linewidths=2)

        circle = plt.Circle((0,0), radius=2, fill=False, color='green', linewidth=2)
        ax.add_patch(circle)

        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.pause(0.1)


def armijo_backtracking_line_search(z, tau, p, R_z, alpha = 1, max_iter = 100, gamma = 0.01, beta = 0.8):
    gradV_dot_p = - (ca.norm_2(R_z(z, tau))**2)
    V_z = 0.5 * ca.norm_2(R_z(z, tau))**2

    for i in range(max_iter):
        z_new = z + alpha * p
        nu_h = z_new[3]
        s_h = z_new[4]
        V_znew = 0.5 * ca.norm_2(R_z(z_new, tau))**2
        armijo_condition = V_z + alpha * gamma * gradV_dot_p
        if ((V_znew <= armijo_condition) and (s_h >= 0) and (nu_h >= 0)):
            return alpha
        else:
            alpha = beta * alpha

    return alpha
        

def NLP_Solve(zk, obj, grad_obj, jac_eq, jac_ineq, hess_obj, hess_eq, hess_ineq, residual, tau=1, tolerance=1e-7, max_iter=200, verbose=False):
    zks = [zk]
    t1 = time.time()

    for i in range(max_iter):

        lambda_k = zks[-1][2]
        nu_k = zks[-1][3]
        s_k = zks[-1][4]
        n_zeros = ca.DM.zeros(2)
        evalf_gradf = grad_obj(zks[-1])
        evalf_jacg = jac_eq(zks[-1])
        evalf_jach = jac_ineq(zks[-1])

        evalf_hessf = hess_obj(zks[-1])
        evalf_hessg = hess_eq(zks[-1])
        evalf_hessh = hess_ineq(zks[-1])

        hess_Lagrangian = evalf_hessf + lambda_k*evalf_hessg + nu_k*evalf_hessh

        evalf_g = eq_const(zks[-1])
        evalf_h = ineq_const(zks[-1])

        R_z = residual(zks[-1],tau)

        if ca.norm_2(R_z) < tolerance:
            if verbose == True:
                print("=====================================================================")
                print(f'Solution found in {i+1} iterations')
                print(f'Solution time (seconds): {round(time.time()- t1,5)}')
                print(f'Optimal primal variables p*: {zks[-1].full()[0:2].T}')
                print(f'Objective evaluation at f(p*): {obj(zks[-1])}')
                print("=====================================================================")
                print(f'Optimal dual variables d* and slack variables s*: {zks[-1].full()[2:].T}')
                return zks
            else:
                return zks

        KKT = ca.vertcat(ca.horzcat(hess_Lagrangian, evalf_jacg.T, evalf_jach.T, n_zeros),
                        ca.horzcat(evalf_jacg, 0, 0, 0),
                        ca.horzcat(evalf_jach, 0, 0, 1),
                        ca.horzcat(0, 0, 0, s_k, nu_k)
                        )
        
        step_k = ca.inv(KKT) @ R_z

        alpha = armijo_backtracking_line_search(zks[-1], tau, -step_k, Residual, alpha = 1, max_iter = 100, gamma = 1e-6, beta = 0.95)

        zk_next = zks[-1] - alpha*step_k

        tau *= 0.1
        zks.append(zk_next)

    if verbose==True:
        print("=====================================================================")
        print(f'Solution found in {i+1} iterations')
        print(f'Solution time (seconds): {round(time.time()- t1,5)}')
        print(f'Optimal primal variables: {zks[-1].full()[0:2].T}')
        print(f'Objective evaluation at f(x*): {obj(zks[-1])}')
        print("=====================================================================")
        print(f'Optimal dual variables d* and slack variables s*: {zks[-1].full()[2:].T}')
        return zks
    else:
        return zks


x = ca.SX.sym('x', 2)

s = ca.SX.sym('s')
lambda_g = ca.SX.sym('lambda_g')
nu_h = ca.SX.sym('nu_h')

z = ca.vertcat(x,
               lambda_g,
               nu_h,
               s)

RHS_f = (x[0] - 4)**2 + (x[1]-4)**2

RHS_g = ca.sin(x[0]) - x[1]**2
RHS_h = x[0]**2 + x[1]**2 - 4

obj = ca.Function('obj', [z], [RHS_f])
eq_const = ca.Function('eq', [z], [RHS_g])
ineq_const = ca.Function('ineq', [z], [RHS_h])


[obj_hess, gf] = ca.hessian(RHS_f, x)
[g_hess, gg] = ca.hessian(RHS_g, x)
[h_hess, gh] = ca.hessian(RHS_h,x)


grad_obj = ca.Function('grad_obj', [z], [ca.jacobian(RHS_f, x)])
hess_obj = ca.Function('hess_obj', [z], [obj_hess])

jac_eq = ca.Function('jac_eq', [z], [ca.jacobian(RHS_g, x)])
hess_eq = ca.Function('hess_eq', [z], [g_hess])

jac_ineq = ca.Function('jac_ineq', [z], [ca.jacobian(RHS_h,x)])
hess_ineq = ca.Function('hess_ineq', [z], [h_hess])

zk = ca.DM([-2,-4,10,10,10])
#zk = ca.DM([0,0,10,10,10])

tau = ca.SX.sym('tau')


Residual_RHS = ca.vertcat(ca.jacobian(RHS_f, x).T + z[2] * ca.jacobian(RHS_g, x).T +  z[3] * ca.jacobian(RHS_h,x).T,
                          RHS_g,
                          RHS_h + z[4],
                          z[3]*z[4]-tau)

Residual = ca.Function('R_z', [z,tau], [Residual_RHS])


if __name__ == "__main__":

    zks = NLP_Solve(zk, obj, grad_obj, jac_eq, jac_ineq, hess_obj, hess_eq, hess_ineq, Residual, tau=1, tolerance=1e-6, max_iter=1000,verbose=True)

    plot_contours(zks)
