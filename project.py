from functools import reduce
import math
import numpy as np
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import cvxpy as cvx

class Planet:
    def __init__(self, type, mass, radius, position, trajectory_r=0, angle=0, speed=0):
        self.mass = mass
        self.radius = radius
        self.position = position
        self.type = type
        if type == 'stationary':
            self.trajectory_r = 0
            self.speed = 0
            self.angle = 0
        elif type == 'meteorite':
            self.mass = 0
            self.trajectory_r = 0
            self.speed = speed
            self.angle = angle
        elif type == 'planet':
            self.trajectory_r = trajectory_r
            self.speed = speed
            self.angle = angle

def planet_motion(planet, t):
    if planet.type == 'stationary':
        position =  jnp.array(planet.position).reshape(2,1)
    elif planet.type == 'meteorite':
        origin = jnp.array(planet.position)
        angle = planet.angle
        speed = planet.speed
        vx = speed*jnp.cos(angle)
        vy = speed*jnp.sin(angle)
        position = jnp.array([origin[0] + vx*t, origin[1] + vy*t])
    elif planet.type == 'planet':
        origin = jnp.array(planet.position)
        angle = planet.angle
        speed = planet.speed
        R = planet.trajectory_r
        position = jnp.array([origin[0] + R*jnp.cos(speed*t+angle), origin[1] + R*jnp.sin(speed*t+angle)])
    return position


def compute_all_planet(all_planet,t):
    all_mass = jnp.zeros((1,))
    planet_x = jnp.zeros((1,))
    planet_y = jnp.zeros((1,))
    for i in range(len(all_planet)):
        planet = all_planet[i]
        all_mass = jnp.append(all_mass,planet.mass)
        position = planet_motion(planet,t)
        planet_x = jnp.append(planet_x,position[0])
        planet_y = jnp.append(planet_y,position[1])
    return all_mass[1:], planet_x[1:], planet_y[1:]

def initialize_map(map_num):
    # map 1:
    if map_num == 1:
        p1 = Planet('stationary',mass=5.972*10**(12),radius=3.0,position=[80,80])
        p2 = Planet('planet',mass=6.417*10**11,radius=3.3895,position=[0,0],trajectory_r=10,angle=0,speed=0.05)
        p3 = Planet('meteorite', mass=0, radius=2, position=[0,70], angle=-np.pi/3, speed=2)
        p4 = Planet('meteorite', mass=0, radius =2.5, position=[80,100], angle=4.1, speed=1.5)
        all_planet = [p1,p2,p3,p4]
    # map 2:
    elif map_num == 2:
        # p1 = Planet('planet', mass=6.417*10**11, radius=4, position=[0,0],trajectory_r=30,angle=0.4,speed=0.006)
        p2 = Planet('planet', mass=5.972*10**(12),radius=6.371,position=[0,0],trajectory_r=60,angle=0,speed=0.03)
        p3 = Planet('planet', mass=8.682*10**13, radius=9.0, position=[0,0], trajectory_r=140,angle=np.pi/2+0.5,speed=-0.02)
        p4 = Planet('planet', mass=1.024*10**14, radius=8.0, position=[200,200], trajectory_r=80, angle=np.pi/2+0.3, speed=0.03)
        all_planet = [p2,p3,p4]
    
    return all_planet

def map_plot(all_planet,t):
    fig, ax = plt.subplots()
    for planet in all_planet:
        planet_pos = planet_motion(planet,t)
        if planet.type == 'stationary':
            plt_c1 = plt.Circle(planet_pos,planet.radius,color='black',fill=False)
            ax.add_patch(plt_c1)
        elif planet.type == 'meteorite':
            plt_c1 = plt.Circle(planet_pos,planet.radius,color='red',fill=False)
            ax.add_patch(plt_c1)
        elif planet.type == 'planet':
            plt_c1 = plt.Circle(planet_pos,planet.radius,color='blue',fill=False)
            plt_c2 = plt.Circle(planet_pos,2*planet.radius,color='blue',ls='--',fill=False)
            ax.add_patch(plt_c1)
            ax.add_patch(plt_c2)
    ax.set_aspect('equal')
    ax.set_xlim([-10,210])
    ax.set_ylim([-10,210])
    return
    

def Spacecraft_dynamics(s,u,t):
    x, y, dx, dy = s
    m,px,py = compute_all_planet(all_planet,t)
    ds = jnp.array([
        dx,
        dy,
        G*jnp.sum(m*jnp.abs(x-px)/((x-px)**2+(y-py)**2)**(3/2))+u[0],
        G*jnp.sum(m*jnp.abs(y-py)/((x-px)**2+(y-py)**2)**(3/2))+u[1]
    ])
    return ds


@jax.partial(jax.jit, static_argnums=(0,))
def linearize(f, x, u, t):
    A, B = jax.jacfwd(f,(0,1))(x,u,t)
    c = f(x,u,t) - A@x - B@u
    return A, B, c


def scp_normal(A,B,c,P,Q,R,x0,x_goal,N):
    cost_terms = []
    constraints = []
    X = {}
    U = {}
    K = np.identity(4)
    K[0,0] = 0
    K[1,1] = 0
    speed_limit = np.zeros((4,1))
    for k in range(N+1):
        X[k] = cvx.Variable((4,1))
        U[k] = cvx.Variable((2,1))
        delta = cvx.Variable()
        if (k == 0):
            constraints.append( X[0] == x0 )
        if (k > 0):
            constraints.append( A@X[k-1]+B@U[k-1]+c.reshape(4,1) == X[k] )
        if (k < N):
            cost_terms.append(cvx.quad_form(U[k],R))
            constraints.append(cvx.norm(U[k],2) <= 3)
        constraints.append(cvx.quad_form(X[k],K) <= 100+delta)
        constraints.append(delta >= 0)
        cost_terms.append(50*delta)
        cost_terms.append(cvx.quad_form(X[k]-x_goal,Q))
    cost_terms.append(cvx.quad_form(X[N]-x_goal,P))

    objective = cvx.Minimize(cvx.sum(cost_terms))
    prob = cvx.Problem(objective,constraints)
    prob.solve()
    return U[0].value,prob.status


def log_barrier(s,so,r,mu):
    so = np.append(so.reshape(2,1),np.array([0,0]).reshape(2,1),axis=0)
    L = np.diag([1,1,0,0])
    dlog = 2*mu*(L@(s-so))/(np.linalg.norm(L@(s-so))**2-r**2)
    ddlog = (dlog@(2*mu*np.transpose(L@(s-so)))-2*mu)/(np.linalg.norm(L@(s-so))**2-r**2)
    return dlog, ddlog
    

def iLQR_avoidance(f, N, s0, s_goal, obstacle, Q, R, P, t, mu):
    eps = 0.01
    max_iters = int(1e3)
    L = np.zeros((N, 2, 4))
    l = np.zeros((N, 2))
    u_bar = np.zeros((N, 2))
    s_bar = np.zeros((N + 1, 4))
    s_bar[0] = s0.reshape(4,)
    # Kv = np.diag([1,1,0,0])
    for k in range(N):
        s_bar[k+1] = f(s_bar[k], u_bar[k], k)
    u = np.copy(u_bar)
    s = np.copy(s_bar)
    s_goal = s_goal
    converged = False
    iter = 0
    obstacle_pos = [planet_motion(obstacle, i) for i in np.arange(t,t+N)]

    while True:
        print(iter)
        A, B, c = jax.vmap(linearize, in_axes=(None, 0, 0, None))(f, s_bar[:-1], u_bar, t)
        A, B, c = np.array(A), np.array(B), np.array(c)
        v = P@(s_bar[-1].reshape(4,1)-s_goal)
        V = P
        for k in range(N-1,-1,-1):
            dlog,ddlog = log_barrier(s_bar[k].reshape(4,1),obstacle_pos[k],4,mu)
            Qs = Q @ (s_bar[k].reshape(4,1)-s_goal) - dlog + np.transpose(A[k]) @ v
            Qu = R @ u_bar[k] + (np.transpose(B[k]) @ v).reshape(2,)
            Qss = Q - ddlog+ np.transpose(A[k]) @ V @ A[k]
            Quu = R + np.transpose(B[k]) @ V @ B[k]
            Qus = np.transpose(B[k]) @ V @ A[k]
            l[k] = -np.linalg.inv(Quu) @ Qu
            L[k] = -np.linalg.inv(Quu) @ Qus
            v = Qs - np.transpose(L[k]) @ Quu @ l[k].reshape(2,1)
            V = Qss - np.transpose(L[k]) @ Quu @ L[k]

        for k in range(N):
            ds = s[k]-s_bar[k]
            du = l[k] + L[k]@ds
            u[k] = u_bar[k] + du
            s[k+1] = f(s[k], u[k], k)
        
        if (np.amax(np.abs(u - u_bar)) < eps) or (iter > max_iters):
            converged = True
            break
        else:
            u_bar[:] = u
            s_bar[:] = s
            iter += 1
    if not converged:
        raise RuntimeError('iLQR did not converge!')
    obstacle_x = []
    obstacle_y = []
    for i in range(N):
        obstacle_x.append(obstacle_pos[i][0])
        obstacle_y.append(obstacle_pos[i][1])
    for i in range(N):
        fig, ax = plt.subplots()
        plt.plot(s_bar[0:i,0],s_bar[0:i,1])
        plt.plot(obstacle_x[0:i],obstacle_y[0:i],color='red',ls='--')
        if obstacle.type == 'meteorite': col = 'red'
        elif obstacle.type == 'stationary': col = 'black'
        else: col = 'blue'
        plt_p1 = plt.Circle(obstacle_pos[i],2,color = col,fill=False)
        ax.add_patch(plt_p1)
        ax.set_aspect('equal')
        plt.grid()
        plt.show()

    return s_bar, u_bar, L, l, obstacle_pos
            

def scp_slingshot_chasing(s, planet, A, B, c, P, Q, R, N, t):
    x, y, dx, dy = s
    planet_center = planet_motion(planet,t)
    planet_pos = np.copy(planet_center)
    radius = 2*planet.radius
    dist_to_center = np.sqrt((planet_center[0]-x)**2+(planet_center[1]-y)**2)
    v = np.sqrt(s[2]**2+s[3]**2)
    t_to_des = math.ceil(dist_to_center/v)
    print(t_to_des)
    t_prev = np.copy(t_to_des)
    while True:
        planet_center = planet_motion(planet,t+t_to_des)
        dist_to_center = np.sqrt((planet_center[0]-x)**2+(planet_center[1]-y)**2)
        t_to_des = math.ceil(dist_to_center/v+10)
        if t_to_des-t_prev == 0: break
        t_prev = np.copy(t_to_des)
        print(planet_center)
    angle_to_center = np.arctan((planet_center[1]-y)/(planet_center[0]-x))
    if dist_to_center <= radius: dist_to_center = radius
    tangential_angle = np.arcsin(radius/dist_to_center)
    tangential_dist = dist_to_center*np.cos(tangential_angle)
    if angle_to_center >= 0:
        new_x = x+tangential_dist*np.cos(angle_to_center-tangential_angle)
        new_y = y+tangential_dist*np.sin(angle_to_center-tangential_angle)
    elif angle_to_center <= 0:
        new_x = x+tangential_dist*np.cos(angle_to_center-tangential_angle)
        new_y = y+tangential_dist*np.sin(angle_to_center+tangential_angle)
    new_des = np.array([new_x[0],new_y[0],0,0]).reshape(4,1)
    u, status = scp_normal(A,B,c,P,Q,R,s,new_des,4)

    return u, status, planet_pos

def slingshot(planet,s,s_goal,t):
    L = np.zeros((2,4))
    L[0,0] = 1
    L[1,1] = 1
    x, y, dx, dy = s
    omega = planet.speed
    trajectory_r = planet.trajectory_r
    v = np.sqrt(dx**2+dy**2)
    new_s = np.copy(s)
    trajectory = np.copy(s)
    optimal_u = np.zeros((2,1))
    
    while True:
        print(t)
        dx_base = -trajectory_r*omega*np.sin(omega*t+planet.angle)
        dy_base = trajectory_r*omega*np.cos(omega*t+planet.angle)
        planet_pos = planet_motion(planet,t)
        theta = np.arctan2(planet_pos[1]-s[1],planet_pos[0]-s[0])
        dx = dx_base+v*np.sin(np.pi-theta)*0.2
        dy = dy_base+v*np.cos(np.pi-theta)*0.2
        
        new_s[0] = s[0] + dx
        new_s[1] = s[1] + dy
        new_s[2] = dx
        new_s[3] = dy

        planet_pos = planet_motion(planet,t)
        planet_to_des = np.linalg.norm(L@s_goal-planet_pos)
        angle_to_des = np.arctan2((s_goal[1]-planet_pos[1]),(s_goal[0]-planet_pos[0]))
        radius = 2*planet.radius
        depart_angle = np.arccos(radius/planet_to_des)
        depart_angle = angle_to_des-depart_angle
        depart_pos_x = planet_pos[0]+radius*np.cos(depart_angle)
        depart_pos_y = planet_pos[1]+radius*np.sin(depart_angle)
        depart_pos = np.array([depart_pos_x, depart_pos_y]).reshape(2,1)
        ref_ang = np.arctan2(s_goal[1]-depart_pos_y,s_goal[0]-depart_pos_x)
        self_ang = np.arctan2(s_goal[1]-s[1],s_goal[0]-s[0])
        # print(depart_angle)
        print(depart_pos)
        print(np.linalg.norm(L@new_s - depart_pos))
        fig, ax = plt.subplots()
        plt.plot(trajectory[0,:],trajectory[1,:])
        plt_p1 = plt.Circle(planet_pos,radius,color = 'blue',fill=False,ls='-')
        ax.add_patch(plt_p1)
        ax.set_aspect('equal')
        plt.show()
        t += 1
        trajectory = np.append(trajectory,new_s,axis=1)
        optimal_u = np.append(optimal_u,np.zeros((2,1)),axis=1)
        if np.linalg.norm(L@new_s - depart_pos) <= 1.5:
            new_v = v+2*omega*trajectory_r
            new_s[2] = new_v*np.cos(depart_angle)
            new_s[3] = new_v*np.sin(-depart_angle)
            break
        if planet.radius > 8 and np.abs(ref_ang-self_ang) <=0.005:
            new_v = v+2*omega*trajectory_r
            new_s[2] = new_v*np.cos(depart_angle)
            new_s[3] = new_v*np.sin(-depart_angle)
            break
        if planet.radius == 8 and np.pi/3-self_ang <= 0:
            new_v = v+2*omega*trajectory_r
            new_s[2] = new_v*np.cos(self_ang)
            new_s[3] = new_v*np.sin(self_ang)
            break
        s = np.copy(new_s)

    return new_s,trajectory,optimal_u,t

def whole_slingshot(s_initial,iter,planet,all_planet):
    u_bar = np.zeros((1,2))
    s_bar = np.zeros((1,4))
    u_optimal = np.zeros((2,1))
    trajectory = np.copy(s_initial)
    des_list = np.zeros((4,1))
    while True:
        A, B, c = jax.vmap(linearize, in_axes=(None, 0, 0, None))(f_discrete, s_bar, u_bar, iter)
        A, B, c = np.array(A), np.array(B), np.array(c)
        u, status, planet_pos = scp_slingshot_chasing(s_initial,planet, A[0], B[0], c[0], P, Q, R, 10, iter)
        u_optimal = np.append(u_optimal,u,axis=1)
        u_bar = np.copy(u).reshape(1,2)
        s_initial = A[0]@s_initial+B[0]@u+c[0].reshape(4,1)
        s_bar = np.copy(s_initial).reshape(1,4)
        trajectory = np.append(trajectory,s_initial,axis=1)
        radius = 2*planet.radius
        dist_to_center = np.sqrt((s_initial[0]-planet_pos[0])**2+(s_initial[1]-planet_pos[1])**2)
        map_plot(all_planet,iter)
        plt.plot(trajectory[0,:],trajectory[1,:])
        plt_p1 = plt.Circle(planet_pos,planet.radius*2,color = 'blue',fill=False)
        plt.show()
        iter += 1
        print(s_initial)
        print(planet_pos)
        print(dist_to_center)
        if dist_to_center-radius <= 1:
            print('achieved')
            break

    s_initial,circular_trajectory, circular_u, iter = slingshot(planet,s_initial,s_goal,iter)
    trajectory = np.append(trajectory,circular_trajectory,axis=1)
    u_optimal = np.append(u_optimal,circular_u,axis=1)

    return s_initial,trajectory,u_optimal,iter

def avoidance(f, planet, s_initial, s_goal, P, Q, R, iter, all_planet, mu=9):
    s_bar, u_bar, L, l, obstacle_pos = iLQR_avoidance(f,10,s_initial,s_goal,planet,Q,R,P,iter,mu)
    map_plot(all_planet,iter)
    plt.plot(s_bar[:,0],s_bar[:,1])
    # v = [np.sqrt(s_bar[i,2]**2+s_bar[i,3]**2) for i in range(10)]
    # plt.plot(np.arange(0,10),v)
    plt.grid()
    plt.show()
    s_initial = s_bar[-1,:]
    return s_bar, u_bar, s_initial
    
# p1 = Planet('planet',5.972*10**(12),6.371,[30,30],20,0,0.05)
# p2 = Planet('meteorite',0,2,[9,10],0,-3*np.pi/4,1.2)
# all_planet = [p1,p2]
# obstacle = np.array([5,4,0,0]).reshape(4,1)

G = 6.67430*10**(-11)
scale = 2*10**24
f = jax.jit(Spacecraft_dynamics)
f_discrete = jax.jit(lambda s, u, t, dt=0.2: s + dt*f(s, u, t))
s_bar = np.zeros((1,4))
u_bar = np.zeros((1,2))
Q = np.identity(4)*1
Q[2,2] = 0
Q[3,3] = 0
R = np.identity(2)
P = np.identity(4)*40
P[2,2] = 0
P[3,3] = 0
T = np.zeros((2,4))
T[0,0] = 1
T[1,1] = 1 
N = 10
s_goal = np.array([200,200,0,0]).reshape(4,1)
all_planet = initialize_map(2)
u_optimal = np.zeros((2,1))
iter = 0
trajectory = np.copy(s_bar).reshape(4,1)
s_initial = np.zeros((4,1))
used = []

while True:
    print(iter)
    A, B, c = jax.vmap(linearize, in_axes=(None, 0, 0, None))(f_discrete, s_bar, u_bar, iter)
    A, B, c = np.array(A), np.array(B), np.array(c)
    u, status = scp_normal(A[0],B[0],c[0],P,Q,R,s_initial,s_goal,N)
    u_optimal = np.append(u_optimal,u,axis=1)
    s_initial = A[0]@s_initial+B[0]@u+c[0].reshape(4,1)
    s_bar = np.copy(s_initial).reshape(1,4)
    u_bar = np.copy(u).reshape(1,2)
    trajectory = np.append(trajectory,s_initial,axis=1)
    map_plot(all_planet,iter)
    plt.plot(trajectory[0,:],trajectory[1,:])
    plt.show()
    iter += 1
    _, all_planet_x, all_planet_y = compute_all_planet(all_planet,iter)
    dist_x = all_planet_x - s_initial[0]
    dist_y = all_planet_y - s_initial[1]
    for i in range(len(dist_x)):
        theta = np.arctan2(dist_y[i],dist_x[i])
        dist = np.sqrt(dist_x[i]**2+dist_y[i]**2)
        if (theta <= np.pi or theta >= 0) and (dist <= 50) and (i not in used):
            print(i)
            s_initial, slingshot_trajectory, slingshot_u, iter = whole_slingshot(s_initial,iter,all_planet[i],all_planet)
            trajectory = np.append(trajectory, slingshot_trajectory,axis=1)
            u_optimal = np.append(u_optimal, slingshot_u, axis=1)
            used.append(i)
            break
    
    print(np.linalg.norm(T@(s_goal-s_initial)))
    if (np.linalg.norm(T@(s_goal-s_initial)) <= 10):
        print('Done!')
        break

mu = [8,9,8,11]
# while True:
#     print(iter)
#     A, B, c = jax.vmap(linearize, in_axes=(None, 0, 0, None))(f_discrete, s_bar, u_bar, iter)
#     A, B, c = np.array(A), np.array(B), np.array(c)
#     u, status = scp_normal(A[0],B[0],c[0],P,Q,R,s_initial,s_goal,N)
#     u_optimal = np.append(u_optimal,u,axis=1)
#     s_initial = A[0]@s_initial+B[0]@u+c[0].reshape(4,1)
#     s_bar = np.copy(s_initial).reshape(1,4)
#     u_bar = np.copy(u).reshape(1,2)
#     trajectory = np.append(trajectory,s_initial,axis=1)
#     map_plot(all_planet,iter)
#     plt.plot(trajectory[0,:],trajectory[1,:])
#     plt.show()
#     iter += 1
#     _, all_planet_x, all_planet_y = compute_all_planet(all_planet,iter)
#     dist_x = all_planet_x - s_initial[0]
#     dist_y = all_planet_y - s_initial[1]
#     for i in range(len(dist_x)):
#         dist = np.sqrt(dist_x[i]**2+dist_y[i]**2)
#         s_goal_temp = np.array([s_initial[0]+10, s_initial[1]+10, 0, 0]).reshape(4,1)
#         if dist <= 20 and i not in used:
#             avoid_trajectory, avoid_u, s_initial = avoidance(f_discrete,all_planet[i],s_initial,s_goal_temp,P,Q,R,iter,all_planet,mu[i])
#             iter = iter + N
#             trajectory = np.append(trajectory, np.transpose(avoid_trajectory), axis=1)
#             u_optimal = np.append(u_optimal, np.transpose(avoid_u), axis=1)
#             s_initial = s_initial.reshape(4,1)
#             used.append(i)

#     print(np.linalg.norm(T@(s_goal-s_initial)))
#     if (np.linalg.norm(T@(s_goal-s_initial)) <= 10):
#         print('Done!')
#         break



all_v = [np.sqrt(trajectory[2,i]**2+trajectory[3,i]**2) for i in range(trajectory.shape[1])]
print(len(all_v))
plt.plot(np.arange(len(all_v)),all_v)
plt.title('Spacecraft Velocity vs Time')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.show()

# print(u_optimal)
# fig, ax = plt.subplots()
# plt_p1 = plt.Circle(p1.position,p1.radius*2,color='blue',fill=False)
# ax.add_patch(plt_p1)
# ax.set_aspect('equal')
# plt.plot(trajectory[0,:],trajectory[1,:])
# plt.plot(des_list[0,1:],des_list[1,1:],color='red')
# plt.plot()
# plt.show()

