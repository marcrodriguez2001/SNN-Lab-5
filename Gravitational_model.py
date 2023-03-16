# -------------------------------------------------------
# NUMERICAL SOLUTION NEWTON EQUATIONS OF MOTION
# VELOCITY VERLET METHOD
# Python3 version 
# 
# 2D MODEL OF GRAVITATIONAL MOTION OF THE MOON AROUND EARTH
#
# Units (SI): kg, m, s
# By Marc Rodriguez Salazar
# -------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

print('\n--------------------------------------------------------')
print('2D MODEL OF GRAVITATIONAL MOTION OF THE MOON AROUND EARTH')
print('----------------------------------------------------------')


# Constants
G = 6.6743e-11  # gravitational constant (Nm^2/kg))
M_e = 5.972e24  # mass of Earth (kg)
M_m = 7.342e22  # mass of Moon (kg)
d = 3.844e8     # distance between Earth and Moon (m)
v_o= 1022       # velocity (m/s)

# Initial conditions
x_m = np.array([d, 0])           # initial vector position of Moon (m)
v_m = np.array([0, v_o])         # initial vector velocity of Moon (m/s)
a_m = -G * M_e * x_m / d**3      # initial vector acceleration of Moon (m/s^2)

# Time parameters
# input time step
dt = float(input("\n Time step dt (recommended 25 min):\n>"))*60
# Final time
ntot = int(input("\n Number of time steps (recommended 2000):\n>"))
t_use_it=dt*ntot
T_use_it=t_use_it/86400         #pass s to days
print('Simulation time will be',t_use_it,' s',"= ", T_use_it, " days")
T=27.32
print("\n>Number of revolutions the Moon will make around the Earth:", T_use_it/T)

t_start = 0       # start time
t_end = dt*ntot   # end time        


# Create empty array starting at zero with time, position, velocity and energies
t_array = np.arange(t_start, t_end, dt)    # time array
x_m_array = np.zeros((len(t_array), 2))    # position array of Moon
v_m_array = np.zeros((len(t_array), 2))    # velocity array of Moon
E_k_array = np.zeros(len(t_array))         # kinetic energy array
E_p_array = np.zeros(len(t_array))         # potential energy array
E_tot_array = np.zeros(len(t_array))       # total energy array

# Verlet Velocity algorithm
for i in range(len(t_array)):
    # update position
    x_m_array[i] = x_m + v_m*dt + 0.5*a_m*dt**2
    
    # calculate new acceleration
    a_m_new = -G * M_e * x_m_array[i] / np.linalg.norm(x_m_array[i])**3
    
    # update velocity
    v_m = v_m + 0.5*(a_m + a_m_new)*dt
    
    # update acceleration
    a_m = a_m_new
    
    # calculate energies
    E_k = 0.5 * M_m * np.linalg.norm(v_m)**2
    E_p = - G * M_e * M_m / np.linalg.norm(x_m_array[i])
    E_tot = E_k + E_p
    
    # store data
    v_m_array[i] = v_m
    E_k_array[i] = E_k
    E_p_array[i] = E_p
    E_tot_array[i] = E_tot
    
    # update position for next iteration
    x_m = x_m_array[i]

# Plot Moon's orbit around Earth
plt.plot(0, 0, "bo", markersize=40)                 # plot Earth
plt.plot(d, 0, "ro", markersize=10)                 # plot Moon
plt.plot(x_m_array[:, 0], x_m_array[:, 1], ":")     # plot Moon's orbit
plt.legend(handles=[plt.plot(0, 0, "bo", markersize=15, label="Earth")[0],  plt.plot(d, 0, "ro", markersize=5, label="Moon")[0],plt.plot(x_m_array[:, 0], x_m_array[:, 1], ":", label="Orbit")[0]], loc="upper right")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Moon's Orbit Around Earth")
plt.legend()
plt.show()

# Plot the energies of the system
plt.plot(E_k_array, label="Kinetic energy")
plt.plot(E_p_array, label="Potential energy")
plt.plot(E_tot_array, label="Total energy")
plt.xlabel("Time step")
plt.ylabel("Energy")
plt.legend()
plt.show()