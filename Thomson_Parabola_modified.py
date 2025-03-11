import numpy as np
from scipy.constants import c, m_e, e as q_e, micron, femto, pi, eV 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image
from matplotlib import colors

rcParams.update({'font.size': 16})
cmap=plt.get_cmap('rainbow')

# UNITS 
lambda_SI = 0.8*micron # typical wavelength of laser [m]
omega_SI = 2.*pi*c/lambda_SI # reference frequency [1/s]
mc2 = m_e * c**2 / (1e6*eV) # MeV
unit_length = c/omega_SI # [m]
unit_time = 1./omega_SI # [s]
um = micron/unit_length # [1/um]
fs = femto/unit_time # [1/fs]
unit_B_field = m_e * omega_SI / q_e # [kg/C/s = Tesla]
unit_E_field = m_e * c * omega_SI / q_e

# PARTICLE PUSHER

def boris_algorithm(E, B, q, m, x0, p0, T, dt):   
    # Initialize times
    t = np.linspace(0,T,int(T/dt))
    # initialize arrays for results 
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t) 
    px, py, pz = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    Ex, Ey, Ez = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    Bx, By, Bz = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    # initial condition     
    x[0], y[0], z[0] = x0[0], x0[1], x0[2]
    px[0], py[0], pz[0] = p0[0], p0[1], p0[2]
    Ex[0], Ey[0], Ez[0] = E(x[0], y[0], z[0], t[0])
    Bx[0], By[0], Bz[0] = B(x[0], y[0], z[0], t[0]) 
    # Cycle over time steps
    for n in np.arange(int(T/dt)-1):
        # update position
        gamma = np.sqrt(1.+(px[n]**2+py[n]**2+pz[n]**2)/ m**2)
        x[n+1] = x[n] + dt * px[n] / gamma / m 
        y[n+1] = y[n] + dt * py[n] / gamma / m 
        z[n+1] = z[n] + dt * pz[n] / gamma / m 
        # get fields 
        Ex[n+1], Ey[n+1], Ez[n+1] = E(x[n+1], y[n+1], z[n+1], t[n+1]) 
        Bx[n+1], By[n+1], Bz[n+1] = B(x[n+1], y[n+1], z[n+1], t[n+1]) 
        # p tilde 
        p1x = px[n] + q*dt/2 * Ex[n+1]
        p1y = py[n] + q*dt/2 * Ey[n+1] 
        p1z = pz[n] + q*dt/2 * Ez[n+1] 
        # Lorentz factor approximation
        gamma1 = np.sqrt(1.+(p1x**2+p1y**2+p1z**2)/m**2)
        # Eval b
        bx = q/m*dt/2*Bx[n+1]/gamma1
        by = q/m*dt/2*By[n+1]/gamma1
        bz = q/m*dt/2*Bz[n+1]/gamma1
        # Eval p^(n)     
        b2 = bx**2+by**2+bz**2       
        p2x = (p1x + (p1y*bz - p1z*by) + (p1x*bx+p1y*by+p1z*bz)*bx ) / (1.+b2) 
        p2y = (p1y + (p1z*bx - p1x*bz) + (p1x*bx+p1y*by+p1z*bz)*by ) / (1.+b2)
        p2z = (p1z + (p1x*by - p1y*bx) + (p1x*bx+p1y*by+p1z*bz)*bz ) / (1.+b2)
        # update momentum
        px[n+1] = 2*p2x - px[n]
        py[n+1] = 2*p2y - py[n]
        pz[n+1] = 2*p2z - pz[n] 
    return x, y, z, px, py, pz,t


####################
# Thomson parabola #
####################

# Take the TP image
img = np.array(Image.open('TP_raw_image.png'))
# Consider first channel (RGB)
img = img[:,:,0]
# pixel/m (scale of the image)
delta=1.01*10**4; 
# Create arrays for x and y in meters
x_img = np.arange(len(img[0]))/delta
y_img = np.arange(len(img))/delta

# plot figure
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

# PARTICLE PARAMETERS
m_p = 1836.15
m_n = 1.00137845873*m_p
q = 1# 1 for protons 
m = m_n*0 + m_p*1 # in units of electron mass
ekin_min = 1 # [MeV] #0.29
ekin_max = 0.5
number_of_points = 1

# TIME
dt = 1000*fs # [-]
T =  100000000*fs # [-]

B_size_z = 0.075000/unit_length # [-]
Magnet_screen_gap = 0.1345/unit_length # [-]
x_start = 0.00472/unit_length  # [-]
y_start = 0.00467/unit_length  # [-]
z_start = 0/unit_length  # [-]

# magnetic field in units of m_e * omega_SI / e = Tesla
def B(x, y, z, t):
    if (z < B_size_z): Bx = 0.4566/unit_B_field
    else: Bx = 0
    By = 0
    Bz = 0 
    return [Bx,By,Bz]
    
def E(x,y,z,t):
    if (z < B_size_z): Ex = ((5800/0.0165)/unit_E_field)
    else: Ex = 0
    Ey=0
    Ez=0
    return [Ex,Ey,Ez]     

# Compute Boris Pusher

ekin_points = np.linspace(ekin_min,ekin_max,number_of_points)
x_array = []
y_array = []

for ekin in ekin_points:
    gamma0 = ekin/(m*mc2) + 1
    v0 = np.sqrt(1-1/gamma0**2)
    p0 = [0,0, gamma0 * m * v0]
    x0 = np.array([x_start, y_start, z_start]) # initial position
    x,y,z,px,py,pz,t= boris_algorithm(E, B, q, m, x0, p0, T, dt)
    gamma = np.sqrt(1+(px**2+py**2+pz**2)/m**2)
    ekin = m*mc2*(gamma-1)
    
    # Compute the position of the particle on the screen (at z = B_size_z+Magnet_screen_gap)
    y_on_screen = y[np.argmin(np.abs(z - B_size_z - Magnet_screen_gap))]*unit_length
    x_on_screen = x[np.argmin(np.abs(z - B_size_z - Magnet_screen_gap))]*unit_length
    # Add point to figure
    x_array.append(x_on_screen)
    y_array.append(y_on_screen)
       
ax1.scatter(x_array, y_array, s=5, color='red', zorder=11)
plt.savefig("TP_image.png")
# plt.close()

# PLot the thajectory
fig, ax = plt.subplots(1,1, figsize=(4,3), dpi=300., subplot_kw=dict(projection='3d'))
ax.plot(x*unit_length, y*unit_length, z*unit_length) 
ax.scatter(x[0]*unit_length, y[0]*unit_length, z[0]*unit_length, s=5, color='red', zorder=11)
ax.scatter(x[-1]*unit_length, y[-1]*unit_length, z[-1]*unit_length, s=5, color='black', zorder=11)
ax.set_xlabel(r'x [m]', labelpad=15.)
ax.set_ylabel(r'y [m]', labelpad=15.)
ax.set_zlabel(r'z [m]', labelpad=25.)      
ax.tick_params(axis='z', pad=15.)
ax.tick_params(axis='x', pad=5.)
plt.tight_layout()
plt.savefig("Trajectory.png")
# plt.close()

#Uncomment to graph the approximated expressions or exect expression to the boris pusher
'''
#Approximated expressions and plots (Boris pusher vs approximated)
N_q = 1                  # Net charge
N_a = 1                  # Mass number (A)
q = 1.602e-19*N_q        # Charge of proton [C]
m = 1.673e-27*N_a        # Mass of proton [kg]
E = (5800/0.0165)        # Electric field [V/m]
B = (0.4566)             # Magnetic field [T]
w = (q*B/m)              #Cyclotron frequency [rad/s]
x0 = 0.00472
y0 = 0.00467
z0 = 0  
z_p = 0.075
z_gap = 0.1345       

ekin_points2 = np.linspace(ekin_min,ekin_max,2000)

x_arr = []
y_arr = []

for ek in ekin_points2:
    
    x_en = x0 + (q*E*z_p*(6.242*1e12)*(z_p/2 + z_gap))/ek/2;
    y_en = y0 + (w*z_p*(np.sqrt(6.242*1e12*m))*(z_p/2 + z_gap))/(np.sqrt(2*ek));
    x_arr.append(x_en)
    y_arr.append(y_en)


# Line plot
plt.figure(figsize=(8, 6))  # Seconda figura
plt.plot(ekin_points2, x_arr, label="Approximated Expression", zorder=10)
# Scatter plot
plt.scatter(ekin_points, x_array, s=5, color='red', label="Boris Pusher", zorder=11)

# Labels, title, and grid
plt.xlabel("Energy [MeV]")  
plt.ylabel("y [m]")  
plt.title("Boris pusher vs approximated expression (x-axis)")   
plt.grid(True)                    
plt.legend()  # Add legend for clarity

# Save the figure without cutting off labels
plt.savefig("plot_energy_x.png", bbox_inches='tight')
plt.show()  
    
plt.figure(figsize=(8, 6))  # Seconda figura
plt.plot(ekin_points2, y_arr, label="Approximated Expression", zorder=10)
# Scatter plot
plt.scatter(ekin_points, y_array, s=5, color='red', label="Boris Pusher", zorder=11)

# Labels, title, and grid
plt.xlabel("Energy [MeV]")  
plt.ylabel("y [m]")  
plt.title("Boris pusher vs approximated expression (y-axis)") 
plt.grid(True)                    
plt.legend()  # Add legend for clarity

# Save the figure without cutting off labels
plt.savefig("plot_energy_y.png", bbox_inches='tight')
plt.show()  

######
A = (q*E*z_p*(6.242*1e12)*(z_p/2 + z_gap))/2
B = (w*z_p*(np.sqrt(6.242*1e12*m))*(z_p/2 + z_gap))/(np.sqrt(2))
'''
'''
#Exact expressions and plots (Boris pusher vs exact)
N_q = 1                  # Net charge
N_a = 1                  # Mass number (A)
q = 1.602e-19*N_q        # Charge of proton [C]
m = 1.673e-27*N_a        # Mass of proton [kg]
E = (5800/0.0165)      # Electric field [V/m]
B = (0.4566)             # Magnetic field [T]
w = (q*B/m)              #Cyclotron frequency [rad/s]
x0 = 0.00472
y0 = 0.00467
z0 = 0  
z_p = 0.075
z_gap = 0.1345       

ekin_points2 = np.linspace(ekin_min,ekin_max,300)

#I dati sono presi da thomson_analytical.py

x_arr = np.load("data_x.npy")
y_arr = np.load("data_y.npy")


# Line plot
plt.figure(figsize=(8, 6))  # Seconda figura
plt.plot(ekin_points2, x_arr, label="Exact Expression", zorder=10)
# Scatter plot
plt.scatter(ekin_points, x_array, s=5, color='red', label="Boris Pusher", zorder=11)

# Labels, title, and grid
plt.xlabel("Energy [MeV]")  
plt.ylabel("X [m]")  
plt.title("Boris pusher vs exact expression (x-axis)")   
plt.grid(True)                    
plt.legend()  # Add legend for clarity

# Save the figure without cutting off labels
plt.savefig("plot_energy_x_2.png", bbox_inches='tight')
plt.show()  
    
plt.figure(figsize=(8, 6))  # Seconda figura
plt.plot(ekin_points2, y_arr, label="Exact Expression", zorder=10)
# Scatter plot
plt.scatter(ekin_points, y_array, s=5, color='red', label="Boris Pusher", zorder=11)

# Labels, title, and grid
plt.xlabel("Energy [MeV]")  
plt.ylabel("Y [m]")  
plt.title("Boris pusher vs exact expression (y-axis)")   
plt.grid(True)                    
plt.legend()  # Add legend for clarity

# Save the figure without cutting off labels
plt.savefig("plot_energy_y_2.png", bbox_inches='tight')
plt.show()  

######


x_arr = []

for ek in ekin_points2:
    
    x_en = x0 + (q*E*z_p*(6.242*1e12)*(z_p/2 + z_gap))/ek;
    x_arr.append(x_en)
    


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


