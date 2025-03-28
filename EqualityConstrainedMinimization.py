import casadi as ca
import matplotlib.pyplot as plt
import time as time


def solveKKT(w0, grad_f, jac_g, g_equalities, solver="Exact", H_approx = None, Hessian_f=None, Hessian_g=None, max_iter=100, tolerance = 1e-6):
    if solver == "Exact":
        wks = [w0]
        time1 = time.time()
        for i in range(max_iter):
            H_f = Hessian_f(wks[i])
            H_g = Hessian_g(wks[i])
            g_f = grad_f(wks[i])
            g_g = jac_g(wks[i])
            lambda_g = wks[i][2]
            g_eq = g_equalities(wks[i])
            H_Lag = H_f + lambda_g * H_g

            R_wk = ca.vertcat(g_f.T, g_eq)

            KKT = ca.vertcat(
                ca.horzcat(H_Lag, g_g.T),
                ca.horzcat(g_g, 0)                         
                            )

            step = ca.inv(KKT) @ R_wk
            wk_next = wks[i] - step

            if ca.norm_2(step) < tolerance:
                time2 = time.time()
                print(f'Solution time: {time2-time1}')
                print(f'Converged in {i+1} iterations')
                wks.append(wk_next)
                break
            
            wks.append(wk_next)

        time2 = time.time()
        print(f'Solution time: {time2-time1}')
        print(f'Converged in {i+1} iterations')
        
        return wks[-1]

    elif solver=="Approximate":
        wks = [w0]
        B = H_approx
        time1 = time.time()
        for i in range(max_iter):
            g_f = grad_f(wks[i])
            g_g = jac_g(wks[i])
            lambda_g = wks[i][2]
            g_eq = g_equalities(wks[i])
            H_Lag = B

            R_wk = ca.vertcat(g_f.T, g_eq)

            KKT = ca.vertcat(
                ca.horzcat(B, g_g.T),
                ca.horzcat(g_g, 0)                         
                            )

            step = ca.inv(KKT) @ R_wk
            wk_next = wks[i] - step

            if ca.norm_2(step) < tolerance:
                time2 = time.time()
                print(f'Solution time: {time2-time1}')
                print(f'Converged in {i+1} iterations')
                wks.append(wk_next)
                break
            
            wks.append(wk_next)

        time2 = time.time()
        print(f'Solution time: {time2-time1}')
        print(f'Converged in {i+1} iterations')
        
        return wks[-1]



def solve_exactHessian():
    X = ca.SX.sym('x', 2)
    lambda_g = ca.SX.sym('lambda_g')
    W = ca.vertcat(X,lambda_g)

    f_RHS = 0.5*(X[0]-1)**2 + 0.5*(10*(X[1]-X[0]**2))**2 + 0.5*X[0]**2
    g_RHS = X[0] + (1-X[1])**2
    obj_f = ca.Function('obj_f', [W] , [f_RHS])
    constraint_g = ca.Function('g_RHS' , [W], [g_RHS])


    grad_f = ca.Function('grad_f' , [W], [ca.jacobian(f_RHS, X)])
    grad_g = ca.Function('grad_g' , [W], [ca.jacobian(g_RHS,X)])

    [H_f, g_f] = ca.hessian(f_RHS,X)
    [H_g, g_g] = ca.hessian(g_RHS,X)
    hessian_f = ca.Function('hessian_f' , [W], [H_f])
    hessian_g = ca.Function('hessian_g' , [W], [H_g])


    w0 = ca.DM([1,-1,1])

    print('')
    print('Solving using exact Hessian information: ')
    solution = solveKKT(w0, grad_f, grad_g, constraint_g, Hessian_f=hessian_f, Hessian_g=hessian_g)
    [sol_x, sol_y, sol_lambda] = solution.full()
    print('========================================================================')
    print(f'Found solution for x= {sol_x}, y= {sol_y}, lambda_g = {sol_lambda}')


def solve_HessianApproximation():
    X = ca.SX.sym('x', 2)
    lambda_g = ca.SX.sym('lambda_g')
    W = ca.vertcat(X,lambda_g)

    f_RHS = 0.5*(X[0]-1)**2 + 0.5*(10*(X[1]-X[0]**2))**2 + 0.5*X[0]**2
    g_RHS = X[0] + (1-X[1])**2
    obj_f = ca.Function('obj_f', [W] , [f_RHS])
    constraint_g = ca.Function('g_RHS' , [W], [g_RHS])


    grad_f = ca.Function('grad_f' , [W], [ca.jacobian(f_RHS, X)])
    grad_g = ca.Function('grad_g' , [W], [ca.jacobian(g_RHS,X)])

    rho = 600
    B = rho * ca.DM_eye(2)

    w0 = ca.DM([1,-1,1])

    print('')
    print('Solving using approximate Hessian information: ')
    solution = solveKKT(w0, grad_f, grad_g, constraint_g, solver="Approximate", H_approx=B, max_iter=1000)
    [sol_x, sol_y, sol_lambda] = solution.full()
    print('========================================================================')
    print(f'Found solution for x= {sol_x}, y= {sol_y}, lambda_g = {sol_lambda}')

solve_exactHessian()

solve_HessianApproximation()

