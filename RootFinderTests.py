import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time as time



def SolveExactNewton(wk, Jac, Hess, tolerance, max_iter):
    """
    params:

    wk (ca.DM)
    Jac (ca.Function)
    Hess (ca.Function)
    tolerance (float)
    max_iter (int)

    """
    wks = [wk]
    time1 = time.time()
    for i in range(max_iter):
        J_eval = Jac(wks[-1][0], wks[-1][1])
        H_eval = Hess(wks[-1][0], wks[-1][1])
        invHessian = ca.inv(H_eval)

        step = invHessian @  J_eval
        wk_next = wks[-1] -  step

        if ca.norm_2(step) < tolerance:
            time2 = time.time()
            print(f'Solution time: {time2-time1}')
            print(f'Converged in {i+1} iterations')
            wks.append((wk_next))
            break
        
        wks.append(wk_next)

    return wks[-1]

def SolveNewtonType(wk, Jac, M,tolerance, max_iter):
    """
    params:

    wk (ca.DM)
    Jac (ca.Function)
    M (ca.Function)
    tolerance (float)
    max_iter (int)

    """

    wks = [wk]
    time1 = time.time()
    for i in range(max_iter):
        J_eval = Jac(wks[-1][0], wks[-1][1])
        invHessian = ca.inv(M)

        step = invHessian @  J_eval
        wk_next = wks[-1] -  step

        if ca.norm_2(step) < tolerance:
            time2 = time.time()
            print(f'Solution time: {time2-time1}')
            print(f'Converged in {i+1} iterations')
            wks.append((wk_next))
            break
        
        wks.append(wk_next)
    return wks[-1]
    

def CasadiExactNewton(wk, Jac, Hess, tolerance, max_iter):
    wks = [wk]
    time1 = time.time()
    for i in range(max_iter):
        J_eval = Jac(wks[-1])
        H_eval = Hess(wks[-1])
        invHessian = ca.inv(H_eval)

        step = invHessian @  J_eval.T
        wk_next = wks[-1] -  step

        if ca.norm_2(step) < tolerance:
            time2 = time.time()
            print(f'Solution time: {time2-time1}')
            print(f'Converged in {i+1} iterations')
            wks.append((wk_next))
            break
        
        wks.append(wk_next)

    return wks[-1]


########## Solutionss ###############


def exact_newton_example():
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')

    jac_f = ca.vertcat(-2*(1-x)-400*x*(y-x**2), 200*(y-x**2))
    hess_f = ca.vertcat(
        ca.horzcat(2 - 400*y + 1200*x**2, -400*x),
        ca.horzcat(-400*x, 200))

    Jacobian = ca.Function('Jacobian', [x,y], [jac_f])

    Hessian = ca.Function('Hessian', [x,y], [hess_f])


    X0 = ca.DM([1,1.1])

    print(f'Optimal Solution: {SolveExactNewton(X0, Jacobian, Hessian, 1e-6, 3000)}')


def newton_type_example():
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')

    jac_f = ca.vertcat(-2*(1-x)-400*x*(y-x**2), 200*(y-x**2))
    hess_f = ca.vertcat(
        ca.horzcat(2 - 400*y + 1200*x**2, -400*x),
        ca.horzcat(-400*x, 200))

    Jacobian = ca.Function('Jacobian', [x,y], [jac_f])

    Hessian = ca.Function('Hessian', [x,y], [hess_f])

    rho = 100           #does not converge for rho = 100, however it does converge at rho=500
    M = rho*ca.DM([[1,0],[0,1]])
    X0 = ca.DM([1,1.1])

    print(f'Optimal Solution: {SolveNewtonType(X0, Jacobian, M, 1e-6, 3000)}')




def newtonCasadi_example():
    x = ca.MX.sym('x',2,1)
    RHS = (1-x[0])**2 +100*(x[1]-x[0]**2)**2
    #RHS = (x[0]+2*x[1]-7)**2 + (2*x[0]+x[1]-5)**2



    max_iter = 10
    tolerance = 1e-6
    X0= ca.DM([1,1.1])



    J_sym = ca.jacobian(RHS,x)
    [H_sym,grad_sym] = ca.hessian(RHS,x)

    Jacobian = ca.Function('Jacobian', [x], [J_sym])
    Hessian = ca.Function('Hessian', [x], [H_sym])

    
    print(f'Optimal Solution: {CasadiExactNewton(X0, Jacobian, Hessian, 1e-6, 3000)}')


exact_newton_example()
#newton_type_example()
newtonCasadi_example()