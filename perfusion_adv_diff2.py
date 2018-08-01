
# coding: utf-8

# Imports

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

parameters['allow_extrapolation'] = True
WARNING = 30
set_log_level(WARNING)

import pandas as pd

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# Importing the mesh
mesh=Mesh()

f = XDMFFile(mesh.mpi_comm(),"Files/pressure_mesh.xdmf")
f.read(mesh)
f.close()

plot(mesh)

# Creating files that will be saved to,

## Files for pressure
xdmffile_p1 = XDMFFile(mesh.mpi_comm(), 'perfusion_adv_diff2_results/p1.xdmf')
xdmffile_p2 = XDMFFile(mesh.mpi_comm(), 'perfusion_adv_diff2_results/p2.xdmf')
xdmffile_p3 = XDMFFile(mesh.mpi_comm(), 'perfusion_adv_diff2_results/p3.xdmf')

## Files for darcy velocity

xdmffile_v1 = XDMFFile(mesh.mpi_comm(), 'perfusion_adv_diff2_results/v1.xdmf')
xdmffile_v2 = XDMFFile(mesh.mpi_comm(), 'perfusion_adv_diff2_results/v2.xdmf')
xdmffile_v3 = XDMFFile(mesh.mpi_comm(), 'perfusion_adv_diff2_results/v3.xdmf')




## Files for concentration

xdmffile_c1 = XDMFFile(mesh.mpi_comm(), 'perfusion_adv_diff2_results/c1.xdmf')
xdmffile_c2 = XDMFFile(mesh.mpi_comm(), 'perfusion_adv_diff2_results/c2.xdmf')
xdmffile_c3 = XDMFFile(mesh.mpi_comm(), 'perfusion_adv_diff2_results/c3.xdmf')

# Importing initial pressure

path = "Files/coronary_pressure.csv"
df = pd.read_csv(path,names=['time','pressure'])

# Fixing units
time = np.array(df['time'])
pressure = np.array(df['pressure']) # mmHg, 0.133322368 mmHg = 1 Kpa
pressure = pressure * 0.133322368 # KPa

#Interpolation of time and pressure
num_time_steps = 101
new_time = np.linspace(0,1,num_time_steps)
func_interpol_p = interp1d(time,pressure)
new_pressure = func_interpol_p(new_time)

time = new_time
pressure = new_pressure


# Define function spaces. Define trial and test functions

el = tetrahedron

P = FiniteElement('P',el,2)

element = MixedElement([P,P,P])
FS = FunctionSpace(mesh,element)

# Define test function
q1,q2,q3 = TestFunctions(FS)
v1, v2, v3 = TestFunctions(FS)

# Define constants
#diff_o2 = 0.9*10**-9
diff_o2 = 10**-5

dt = 0.01
K1 = Constant(1) #(mm^2)/(kPa*s)
K2 = Constant(10) #(mm^2)/kPa*s
K3 = Constant(20) #(mm^2)/kPa*s
beta12 = Constant(0.02) #1/(kPa*s)
beta23 = Constant(0.05) #1/(kPa*s)

D = Constant(diff_o2)
k = Constant(dt)

# Define functions 

p = Function(FS) #Pressure
v_d1 = Function(FS) #Darcy-velocity
v_d2 = Function(FS) #Darcy-velocity
v_d3 = Function(FS) #Darcy-velocity

c = Function(FS) #Concentration this timestep
c_n = Function(FS) #Consentration last timestep

p1, p2, p3 = split(p)
c1, c2, c3 = split(c)
c_n1, c_n2, c_n3 = split(c_n)

S3 = - Constant(0.1)*(p3-Constant(3.0)) #Sink term in the third apartment

#Define Equations

##Reduced darcy equation
F = -K1 * dot(grad(p1), grad(q1))*dx + -K2 * dot(grad(p2), grad(q2))*dx + -K3 * dot(grad(p3), grad(q3))*dx + \
        dot(beta12*(p1-p2),q1)*dx + dot(beta12*(p2-p1),q2)*dx + dot(beta23*(p2-p3),q2)*dx + dot(beta23*(p3-p2),q3)*dx -\
        dot(S3,q3)*dx

## Advection-diffusion reaction
F2 = ((c1 - c_n1) / k)*v1*dx + dot(v_d1, grad(c1))*v1*dx - D*dot(grad(c1),grad(v1))*dx + \
      ((c2 - c_n2) / k)*v2*dx + dot(v_d2, grad(c2))*v2*dx - D*dot(grad(c2),grad(v2))*dx + \
      ((c3 - c_n3) / k)*v3*dx + dot(v_d3, grad(c3))*v3*dx - D*dot(grad(c3),grad(v3))*dx

# Setting boundry conditions
markers = MeshFunction("size_t",mesh,"Files/pressure_markers.xml")

pD = Expression("p",p=0.0,degree=2)
bc = DirichletBC(FS.sub(0),pD,markers,1)
bcs = [bc]

# Setting initial condition
c_0 = Expression(('abs(sin(x[0]))',0,0),degree=1)
c_n_ = project(c_0,FS)
c_n.assign(c_n_)

for t, i_p in zip(time,pressure):
    if t%0.1 == 0:
        print('Time run',t)
    if sum(np.array(c_n.vector())!=0) == 0:
        print("c_n is zero",t)

    if sum(np.array(c.vector())!=0) == 0:
        print("c is zero",t)
    

    pD.p = i_p #Updating initial pressure

    # Solve for pressure
    solve(F==0, p, bc)
    p1_,p2_,p3_ = p.split()
    
    # Calculate the velocity
    v_d1_ = project(-K1 *  grad(p1))
    v_d1.assign(v_d1_)
    v_d2_ = project(-K2*grad(p2))
    v_d2.assign(v_d2_)
    v_d3_ = project(-K3 * grad(p3))
    v_d3.assign(v_d3_)
 
    # Solve for concentration
    solve(F2==0,c)
    c1_, c2_, c3_ = c.split() 
    #Update previous solution
    c_n.assign(c)

    # Save parameters
    xdmffile_p1.write(p1_, t)
    xdmffile_p2.write(p2_,t)
    xdmffile_p3.write(p3_,t)
      
    xdmffile_v1.write(v_d1,t)
    xdmffile_v2.write(v_d2,t)
    xdmffile_v3.write(v_d3,t)

    xdmffile_c1.write(c1_,t)
    xdmffile_c2.write(c2_,t)
    xdmffile_c3.write(c3_,t)

# Close all files

xdmffile_p1.close()
xdmffile_p2.close()
xdmffile_p3.close()
xdmffile_v1.close()
xdmffile_v2.close()
xdmffile_v3.close()
xdmffile_c1.close()
xdmffile_c2.close()
xdmffile_c3.close()
