#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:12:03 2025

@author: gabrielelecce
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import colors

x_array = []
y_array = []

# Show plots
plt.tight_layout()
plt.show()

# Take the TP image
img = np.array(Image.open('TP_raw_image.png'))
# Consider first channel (RGB)
img = img[:,:,0]
# pixel/m (scale of the image)
delta=1.01*10**4; 
# Create arrays for x and y in meters
x_img = np.arange(len(img[0]))/delta
y_img = np.arange(len(img))/delta

# Constants
N_q = 1                  # Net charge
N_a = 1                 # Mass number (A)
q = 1.602e-19*N_q        # Charge of proton [C]
m = 1.673e-27*N_a        # Mass of proton [kg]
E = (5800/0.0165)        # Electric field [V/m]
B = (0.4566)             # Magnetic field [T]
w = (q*B/m)              #Cyclotron frequency [rad/s]

ekin_min = 0.3 # [MeV]
ekin_max =  5
number_of_points = 150

ekin_points = np.linspace(ekin_min,ekin_max,number_of_points)

for Ek in ekin_points:
    
    v0 = np.sqrt((2*Ek)/(m*6.242*1e12)); 
    t_max = (0.075/v0)*3 #Simulation time 
    dt = 1e-15    # Time step
    
    # Derived values
    t = np.arange(0, t_max, dt)  # Time array
    
    #Equation of motion and initial conditions
    x0 = 0.00472
    y0 = 0.00467
    z0 = 0
    
    x = x0 + (q*E/(2*m))*t**2
    y = y0 + (v0/w)*(1-np.cos(w*t))   
    z = z0 + (v0/w)*np.sin(w*t)    
    z_p = 0.075
    z_gap = 0.1345              #zfin = 0.075 + 0.1345
   
    #Speed after the fields, calculated with difference quotient (rapporto incrementale)
    vx_2 = (x[np.argmin(abs(z-0.075))]-x[np.argmin(abs(z-0.075))-1])/(dt)
    vy_2 = (y[np.argmin(abs(z-0.075))]-y[np.argmin(abs(z-0.075))-1])/(dt)
    vz_2 = (z[np.argmin(abs(z-0.075))]-z[np.argmin(abs(z-0.075))-1])/(dt)
    
    #Final positions
    t_2 = 0.1345/vz_2
    x_f = x[np.argmin(abs(z-0.075))] + vx_2*t_2
    y_f = y[np.argmin(abs(z-0.075))] + vy_2*t_2
    
    #Save in array of final positions (multiple ekin)
    x_array.append(x_f)
    y_array.append(y_f)


#Save data
    np.save("data_x.npy", x_array)
    np.save("data_y.npy", y_array)

    
#Plot graph
fig, ax1 = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(4)
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
ax1.set_aspect('equal')
pos = ax1.pcolor(x_img, y_img, img, cmap="GnBu",
                 norm=colors.LogNorm(vmin = img[img > 0].min(), vmax = img[1:-1, 1:-1].max())) 
fig.colorbar(pos, ax=ax1, shrink=0.6) #location='top', anchor=(0.5, 0.3), 
fig.tight_layout()
ax1.scatter(x_array, y_array, s=5, color='red', zorder=11)

plt.savefig("TP_image_2.png")

l = np.sqrt(np.power(x_array,2)+np.power(y_array,2))


'''
#Approximated expressions and plots

ekin_points2 = np.linspace(ekin_min,ekin_max,1000)

x_arr = []
y_arr = []

for ek in ekin_points2:
    
    x_en = x0 + (q*E*z_p*(6.242*1e12)*(z_p/2 + z_gap))/ek;
    y_en = y0 + (w*z_p*(np.sqrt(6.242*1e12*m))*(z_p/2 + z_gap))/(np.sqrt(2*ek));
    x_arr.append(x_en)
    y_arr.append(y_en)


# Line plot
plt.figure(figsize=(8, 6))  # Seconda figura
plt.plot(ekin_points2, x_arr, label="Line", zorder=10)
# Scatter plot
plt.scatter(ekin_points, x_array, s=5, color='red', label="Scatter Points", zorder=11)

# Labels, title, and grid
plt.xlabel("Energy [MeV]")  
plt.ylabel("X [m]")  
plt.title("Boris pusher vs approximated expression (x-axis)")   
plt.grid(True)                    
plt.legend()  # Add legend for clarity

# Save the figure without cutting off labels
plt.savefig("plot_energy_x.png", bbox_inches='tight')
plt.show()  
    
plt.figure(figsize=(8, 6))  # Seconda figura
plt.plot(ekin_points2, y_arr, label="Line", zorder=10)
# Scatter plot
plt.scatter(ekin_points, y_array, s=5, color='red', label="Scatter Points", zorder=11)

# Labels, title, and grid
plt.xlabel("Energy [MeV]")  
plt.ylabel("Y [m]")  
plt.title("Boris pusher vs approximated expression (y-axis)")   
plt.grid(True)                    
plt.legend()  # Add legend for clarity

# Save the figure without cutting off labels
plt.savefig("plot_energy_y.png", bbox_inches='tight')
plt.show()  

######


x_arr = []

for ek in ekin_points2:
    
    x_en = x0 + (q*E*z_p*(6.242*1e12)*(z_p/2 + z_gap))/ek/2;
    x_arr.append(x_en)
    

a = (q*E*z_p*(6.242*1e12)*(z_p/2 + z_gap))/2
b = (w*z_p*(np.sqrt(6.242*1e12*m))*(z_p/2 + z_gap))/(np.sqrt(2))
l0 = np.sqrt(np.power(x0,2)+np.power(y0,2))

# Line plot
plt.figure(figsize=(8, 6))  # Seconda figura
plt.plot(ekin_points2, x_arr, label="Line", zorder=10)
# Scatter plot
plt.scatter(ekin_points, x_array, s=5, color='red', label="Scatter Points", zorder=11)

# Labels, title, and grid
plt.xlabel("Energy [MeV]")  
plt.ylabel("X [m]")  
plt.title("Energy vs X Values")   
plt.grid(True)                    
plt.legend()  # Add legend for clarity

# Save the figure without cutting off labels
plt.savefig("plot_energy.png", bbox_inches='tight')
plt.show()  


'''
