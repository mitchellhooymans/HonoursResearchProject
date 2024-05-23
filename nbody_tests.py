# My first nbody simulations 
# This is the code for my first n-body simulation it takes inspiration from
# an n-body simulation solution from Unimelbourne

# Begin by importing required packages
import numpy as np
import matplotlib.pyplot as plt

# Setup units


m_unit = 2.e30		# Mass unit in kg
l_unit = 1.496e11	# Length unit in m
t_unit = 86400.		# Time unit in s , ie. second per day
G = 6.67e-11		# The gravitational constant in SI units (m^3 kg^-1 s^-2) -- **convert to rescaled units**


n_body = 2     		# Number of bodies in the simulation
delta_t = 0.01		# Length of time step (in time units)
max_step = int(4e5)	# Maximum number of time steps
show_step = 10	# Show every nth step


# Define a class
class Particle:
    
    # This is the inital condition for the particle
    def __init__(self, name, mass, x, y, vx, vy):
        # For a simple system, initalize the variables like so
        self.name = name
        self.mass = mass
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        

    # function to show the particle's parameters
    def show(self):
        print("Body: %s"%self.name)
        print("Mass = %6.2e mass units = %6.2e kg"%(self.mass, self.mass*m_unit))
        print("Position = (%6.2e,%6.2e) length units"%(self.x, self.y))
        print("Velocity = (%6.2e,%6.2e) velocity units"%(self.vx,self.vy))
        print("")
        
    
    
    
# Depending on how many particles we have we can define an array
# to contain these particles
bodies = np.array([])


# Define the initial conditions of the Sun
name1 = 'Sun'
mass1 = 1. # 1 solar mass
x1 = 0
y1 = 0
vx1 = 0
vy1 = 0


# Define the conditions for the earth
name2 = 'Earth'
mass2 = 3.e-6 # 0000003 solar masses
x2 = 1
y2 = 0
vx2 = 0
vy2 = 0.000172

# We will latter define a third body (black hole)
# define the initial conditions of the third body
name3 = 'Black Hole'
mass3 = 10000
x3 = 0
y3 = 5.2
vx3 = 0
vy3 = 0


# Add the two bodies to the array
bodies = np.append(bodies, Particle(name1, mass1, x1, y1, vx1, vy1))
bodies = np.append(bodies, Particle(name2, mass2, x2, y2, vx2, vy2))


for i in range(0, n_body):
    bodies[i].show()


# Now that we have setup the parameters we can try to run the simulation
time = 0


f = open("outfile.csv", "w")
f.write(", ".join(["Xpos body%i, Ypos body%i"%(i+1, i+1) for i in range(n_body)])+", step, time,\n")



for step in range(0, max_step):
    
    # Because we only know how to compute a force between two bodies
    # we setup a matrix to store the acceleration factor
    # Between each pair of all bodies within the simulation
    aFactor = np.zeros((n_body, n_body)) # a 2x2 matrix for us to update
    
    # Calculate the seperations and determine the accelerations
    for i in range(0, n_body):
        for j in range(0, n_body):
            if i != j: #This is so we don't calculate the self-gravity force
                xsq = (bodies[i].x - bodies[j].x)**2 # Calculate the x position
                ysq = (bodies[i].y - bodies[j].y)**2 # calculate the y position
                
                rsq = xsq + ysq
                factor = rsq**-1.5 # use numpy for now
                
                # Update the acceleration factor array
                aFactor[i][j] = G * bodies[j].mass*factor
                aFactor[j][i] = G * bodies[i].mass*factor
                
    # Now we would like to ensure that we have an acceleration initilizaed
    # for each particle
    for i in range(0, n_body):
        # Initalize each body with an acceleration
        bodies[i].ax = 0
        bodies[i].ay = 0
        
        
    # Update each acceleration for each pair of bodies
    for i in range(0,n_body):
        for j in range(0,n_body):
            if i != j:
                # Calculate the x and y components of the acceleration
                bodies[i].ax -= aFactor[i][j] * (bodies[i].x - bodies[j].x)
                bodies[i].ay -= aFactor[i][j] * (bodies[i].y - bodies[j].y)
                
    # For each body, update the position and velocity
    for i in range(0,n_body):
        
        # Update the velocity
        bodies[i].vx += bodies[i].ax * delta_t
        bodies[i].vy += bodies[i].ay * delta_t
        
        # Update the position
        bodies[i].x += bodies[i].vx * delta_t
        bodies[i].y += bodies[i].vy * delta_t
        
        
        
    # we want to be able to output the results
    if(step % show_step == 0):
        
        for i in range(0, n_body):
            f.write("%0.10f,%0.10f,"%(bodies[i].x, bodies[i].y))
        f.write("%5d,%6.4f\n"%(step, time))
    time+=delta_t

f.close()

# Finally plot the result import matplotlib.pyplot as plt

f = open("outfile.csv"); k = f.readlines(); f.close()
x_body1 = [float(i.split(',')[0]) for i in k[1:]]
y_body1 = [float(i.split(',')[1]) for i in k[1:]]
x_body2 = [float(i.split(',')[2]) for i in k[1:]]
y_body2 = [float(i.split(',')[3]) for i in k[1:]]
plt.plot(x_body1, y_body1)
plt.plot(x_body2, y_body2)

plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("N-body simulation")
plt.legend(["Sun", "Earth"])
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.show()
