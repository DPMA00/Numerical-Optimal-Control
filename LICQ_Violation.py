import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import time as time
def RK4_step(x0, func, Ts):

    k1 = func(x0)
    k2 = func(x0 + Ts/2*k1)
    k3 = func(x0 + Ts/2*k2)
    k4 = func(x0 + Ts*k3)
    RK4 = x0 + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return RK4

def kinematics(x):
    g = 9.81
    d = 0.1
    w = 2

    dx1 = x[2]
    dx2 = x[3]

    drag = ca.norm_2(ca.vertcat(x[2], x[3])  - ca.vertcat(w,0))*d
    dx3 = -(x[2]-w) * drag
    dx4 = -x[3] * drag - g

    return ca.vertcat(dx1, dx2, dx3, dx4)

def setup(Tf, N, opt_var, alpha):
    vy, vz = opt_var
    Tf = 0.5
    N = 100

    Ts = 0.5/100
    x0 = ca.vertcat(0,0,vy,vz)

    for i in range(N+1):
        x0 = RK4_step(x0, lambda x: kinematics(x), Ts)
        p_y = x0[0]
        p_z = x0[1]
        v_y = x0[2]
        v_z = x0[3]
        v = ca.vertcat(v_y,v_z)
        if i == 0:
            g = ca.vertcat(ca.norm_2(v))
       

    obj = -x0[0]
    grnd_cnstr = -alpha*(p_y-10) - p_z
    g = ca.vertcat(g,
                  -p_z,
                   grnd_cnstr)

    return obj, g


vy = ca.SX.sym('vy')
vz = ca.SX.sym('vz')
alpha = ca.SX.sym('alpha')
v = ca.vertcat(vy, vz)
opt_var= v.reshape((-1,1))
N = 100
Tf = 0.5
obj,g = setup(Tf,N,[vy,vz], alpha)


lbg = ca.DM.zeros(3)
ubg = ca.DM.zeros(3)

lbg[0:1] = 0
ubg[0:1] = 15

lbg[1:2] = -ca.inf
ubg[1:2] = 0

lbg[2:3] = -ca.inf
ubg[2:3] = 0


nlp_prob =  {
            'f': obj,
            'g':g,
            'x':opt_var,
            'p':alpha
            }

solver = ca.nlpsol('solver','ipopt',nlp_prob)

x0_guess = ca.DM([5, 5]) 

def plotConstraintGradient():
    p_values = np.linspace(-1, 1, 100, endpoint=True)
    origins_x = [0, 0, 0]
    origins_y = [0, 0, 0]

    
    dual_vars = []

    fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 

    for i in range(p_values.shape[0]):
        sol = solver(lbx=-ca.inf, ubx=ca.inf, lbg=lbg, ubg=ubg, x0=x0_guess, p=p_values[i])

        v_init = sol['x']
        lambda_sol = sol['lam_g'] 
        dual_vars.append(lambda_sol.full().flatten())  

        
        Jg = ca.Function('Jac_g', [v, alpha], [ca.jacobian(g, v)])
        jac_g = Jg(v_init, 1)

        velocity_cnstr = np.array(jac_g.full()[0])
        ground_cnstr = np.array(jac_g.full()[1])
        alpha_cnstr = np.array(jac_g.full()[2])

        U = [velocity_cnstr[0], ground_cnstr[0], alpha_cnstr[0]]
        V = [velocity_cnstr[1], ground_cnstr[1], alpha_cnstr[1]]

        colors = ['r', 'g', 'b']

        axs[0].cla()  
        axs[0].quiver(origins_x, origins_y, U, V, angles='xy', scale_units='xy', scale=1, color=colors)
        axs[0].set_xlim(-1, 1)
        axs[0].set_ylim(-1, 1)
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('z')
        axs[0].set_title('Constraint Gradients')
        axs[0].grid(True)

        axs[1].cla()  
        if len(dual_vars) > 1:
            dual_vars_np = np.array(dual_vars)  
            axs[1].plot(p_values[:len(dual_vars)], dual_vars_np, label=["lambda_1", "lambda_2", "lambda_3"])  
            axs[1].set_xlabel('Alpha (p)')
            axs[1].set_ylabel('Lagrange Multipliers')
            axs[1].set_title('Dual Variables vs. Alpha')
            axs[1].legend()
            axs[1].grid(True)

        plt.pause(0.1)  

    plt.show() 


    

def graph(p):
    x0_guess = ca.DM([5, 5]) 
    sol = solver(lbx=-ca.inf, ubx=ca.inf, lbg=lbg, ubg=ubg, x0=x0_guess, p=p)
    v_init = sol['x'].full().flatten()


    trajectory = []

    x_next = ca.DM([0, 0, v_init[0], v_init[1]])

    trajectory.append(x_next.full())


    for i in range(N+1):
        x_next = RK4_step(x_next, lambda x: kinematics(x), Tf/N) 
        trajectory.append(x_next.full()) 


    trajectory = np.array(trajectory) 


    y1_traj = trajectory[:, 0] 
    z1_traj = trajectory[:, 1]  


    plt.plot(y1_traj, z1_traj, label="Ball 1")
    plt.xlabel("y-position")
    plt.ylabel("z-position (height)")
    plt.legend()
    plt.title("Optimized Ballistic Trajectories")
    plt.grid(True)
    plt.show()

    print(v_init)

plotConstraintGradient()

