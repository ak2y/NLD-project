import scipy as sci
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# Universal constant
G = 6.67408e-11

# Refrence quantities
m_nd = 1.989e+30  # kg #mass of sun
r_nd = 5.326e+12  # m #distance between stars in Alpha Centauri
v_nd = 3e4  # earth around sun
t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of aplha centauri

# net constant
K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
K2 = v_nd * t_nd / r_nd

# Define masses
m1 = 1.1  # Star A
m2 = .907  # Star B
m3 = 1  # Star C

'''i,j,k = 1,4,7

r1=[ ]
r2=[ ]
r3=[ ]
while i <=3:
    x = float(input(f"x{i} = "))
    r1.append(x)
    i=i+1
while j <=6:
    x = float(input(f"x{j} = "))
    r2.append(x)
    j=j+1
while k <=9:
    x = float(input(f"x{k} = "))
    r3.append(x)
    k=k+1
print(r1)
print(r2)
print(r3)'''

delta_r = [0.0001, 0, 0]
# some inital condition
r1 = [-0.5, 0, 0]
r2 = [0.5, 0, 0]
r3 = [0, 1, 0]

# convert postion to vector
r1 = np.array(r1, dtype="float64")
r2 = np.array(r2, dtype="float64")
r3 = np.array(r3, dtype="float64")
delta_r = np.array(r3, dtype="float64")

r1_new = r1 + delta_r

# Centre of the mass
r_com = (m1 * r1 + m2 * r2 + m3 * r3) / (m1 + m2 + m3)
r_com_new = (m1 * r1_new + m2 * r2 + m3 * r3) / (m1 + m2 + m3)

# initial velocities
v1 = [0.01, 0.01, 0]
v2 = [-0.05, 0, -0.1]
v3 = [0.01, -0.01, 0]

# convert velocity to vector
v1 = np.array(v1, dtype="float64")
v2 = np.array(v2, dtype="float64")
v3 = np.array(v3, dtype="float64")

# velocity of COM
v_com = (m1 * v1 + m2 * v2 + m3 * v3) / (m1 + m2 + m3)


def ThreeBodyEquations(w, t, G, m1, m2, m3):
    r1 = w[:3]
    r2 = w[3:6]
    r3 = w[6:9]
    v1 = w[9:12]
    v2 = w[12:15]
    v3 = w[15:18]
    r12 = sci.linalg.norm(r2 - r1)
    r13 = sci.linalg.norm(r3 - r1)
    r23 = sci.linalg.norm(r3 - r2)

    dv1bydt = K1 * m2 * (r2 - r1) / r12 ** 3 + K1 * m3 * (r3 - r1) / r13 ** 3
    dv2bydt = K1 * m1 * (r1 - r2) / r12 ** 3 + K1 * m3 * (r3 - r2) / r23 ** 3
    dv3bydt = K1 * m1 * (r1 - r3) / r13 ** 3 + K1 * m2 * (r2 - r3) / r23 ** 3
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2
    dr3bydt = K2 * v3
    r12_derivs = np.concatenate((dr1bydt, dr2bydt))
    r_derivs = np.concatenate((r12_derivs, dr3bydt))
    v12_derivs = np.concatenate((dv1bydt, dv2bydt))
    v_derivs = np.concatenate((v12_derivs, dv3bydt))
    derivs = np.concatenate((r_derivs, v_derivs))
    return derivs


# Package initial parameters
init_params = np.array([r1, r2, r3, v1, v2, v3])  # Initial parameters
init_params = init_params.flatten()  # Flatten to make 1D array
time_span = np.linspace(0, 150, 800)  # 20 orbital periods and 500 points
# Run the ODE solver
import scipy.integrate

three_body_sol = scipy.integrate.odeint(ThreeBodyEquations, init_params, time_span, args=(G, m1, m2, m3))
r1_sol = three_body_sol[:, :3]
r2_sol = three_body_sol[:, 3:6]
r3_sol = three_body_sol[:, 6:9]
fig = plt.figure(figsize=(15, 15))
# create 3D axis
ax = fig.add_subplot(111, projection="3d")

# Plot the orbits
ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], color="darkblue")
ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="tab:red")
ax.plot(r3_sol[:, 0], r3_sol[:, 1], r3_sol[:, 2], color="tab:green")

ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2], color="darkblue", marker="o", s=100, label="Star A")
ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2], color="tab:red", marker="o", s=100, label="Star B")
ax.scatter(r3_sol[-1, 0], r3_sol[-1, 1], r3_sol[-1, 2], color="tab:green", marker="o", s=100, label="Star C")

# Add a few more bells and whistles
ax.set_xlabel("x-coordinate", fontsize=14)
ax.set_ylabel("y-coordinate", fontsize=14)
ax.set_zlabel("z-coordinate", fontsize=14)
ax.set_title("Visualization of orbits of stars in a two-body system\n", fontsize=14)
ax.legend(loc="upper left", fontsize=14)
plt.style.use('dark_background')

# Animate the orbits of the three bodies


# Make the figure
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection="3d")

# Create new arrays for animation, this gives you the flexibility
# to reduce the number of points in the animation if it becomes slow
# Currently set to select every 4th point
r1_sol_anim = r1_sol[::1, :].copy()
r2_sol_anim = r2_sol[::1, :].copy()
r3_sol_anim = r3_sol[::1, :].copy()

# Set initial marker for planets, that is, blue,red and green circles at the initial positions
head1 = [ax.scatter(r1_sol_anim[0, 0], r1_sol_anim[0, 1], r1_sol_anim[0, 2], color="darkblue", marker="o", s=80,
                    label="Star 1")]
head2 = [ax.scatter(r2_sol_anim[0, 0], r2_sol_anim[0, 1], r2_sol_anim[0, 2], color="darkred", marker="o", s=80,
                    label="Star 2")]
head3 = [ax.scatter(r3_sol_anim[0, 0], r3_sol_anim[0, 1], r3_sol_anim[0, 2], color="goldenrod", marker="o", s=80,
                    label="Star 3")]


# Create a function Animate that changes plots every frame (here "i" is the frame number)
def Animate(i, head1, head2, head3):
    # Remove old markers
    head1[0].remove()
    head2[0].remove()
    head3[0].remove()



    # Plot the orbits (every iteration we plot from initial position to the current position)
    trace1 = ax.plot(r1_sol_anim[:i, 0], r1_sol_anim[:i, 1], r1_sol_anim[:i, 2], color="mediumblue")
    trace2 = ax.plot(r2_sol_anim[:i, 0], r2_sol_anim[:i, 1], r2_sol_anim[:i, 2], color="red")
    trace3 = ax.plot(r3_sol_anim[:i, 0], r3_sol_anim[:i, 1], r3_sol_anim[:i, 2], color="gold")

    # Plot the current markers
    head1[0] = ax.scatter(r1_sol_anim[i - 1, 0], r1_sol_anim[i - 1, 1], r1_sol_anim[i - 1, 2], color="darkblue",
                          marker="o", s=100)
    head2[0] = ax.scatter(r2_sol_anim[i - 1, 0], r2_sol_anim[i - 1, 1], r2_sol_anim[i - 1, 2], color="darkred",
                          marker="o", s=100)
    head3[0] = ax.scatter(r3_sol_anim[i - 1, 0], r3_sol_anim[i - 1, 1], r3_sol_anim[i - 1, 2], color="goldenrod",
                          marker="o", s=100)

    plt.cla()
    plt.plot(trace1,trace2,trace3)
    return trace1, trace2, trace3, head1, head2, head3,


# Some beautifying
ax.set_xlabel("x-coordinate", fontsize=14)
ax.set_ylabel("y-coordinate", fontsize=14)
ax.set_zlabel("z-coordinate", fontsize=14)
ax.set_title("Visualization of orbits of stars in a 3-body system\n", fontsize=14)
ax.legend(loc="upper left", fontsize=14)

# If used in Jupyter Notebook, animation will not display only a static image will display with this command
# anim_2b = animation.FuncAnimation(fig,Animate_2b,frames=1000,interval=5,repeat=False,blit=False,fargs=(h1,h2))


# Use the FuncAnimation module to make the animation
repeatanim = animation.FuncAnimation(fig, Animate, frames=800, interval=10, repeat=False, blit=False,
                                     fargs=(head1, head2, head3))

plt.tight_layout()
plt.show()



