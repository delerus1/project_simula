
# coding: utf-8

# In[46]:


from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

parameters['allow_extrapolation'] = True
set_log_level(WARNING)

import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


# Importing the mesh
mesh=Mesh()

f = XDMFFile(mpi_comm_world(),"Files/pressure_mesh.xdmf")
f.read(mesh)
f.close()

plot(mesh)


# In[48]:


# Importing initial pressure

path = "Files/coronary_pressure.csv"

df = pd.read_csv(path,names=['time','pressure'])


# In[49]:


time = np.array(df['time'])
pressure = np.array(df['pressure']) # mmHg


# In[50]:


pressure = pressure * 0.133322368 # KPa


# In[51]:


plt.plot(time,pressure)
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('Initial pressure')


# In[52]:


# Define function spaces. Define trial and test functions

el = tetrahedron

P = FiniteElement('P',el,2)


element = MixedElement([P,P,P])
FS = FunctionSpace(mesh,element)


# Define test functions
q1,q2,q3 = TestFunctions(FS)


K1 = Constant(1) #(mm^2)/(kPa*s)
K2 = Constant(10) #(mm^2)/kPa*s
K3 = Constant(20) #(mm^2)/kPa*s

beta12 = Constant(0.02) #1/(kPa*s)
beta23 = Constant(0.05) #1/(kPa*s)

p = Function(FS)

p1, p2, p3 = split(p)

S3 = - Constant(0.1)*(p3-Constant(3.0)) #Sink term in the third apartment
F = -K1 * dot(grad(p1), grad(q1))*dx + -K2 * dot(grad(p2), grad(q2))*dx + -K3 * dot(grad(p3), grad(q3))*dx     + dot(beta12*(p1-p2),q1)*dx + dot(beta12*(p2-p1),q2)*dx + dot(beta23*(p2-p3),q2)*dx + dot(beta23*(p3-p2),q3)*dx    + dot(S3,q3)*dx

markers = MeshFunction("size_t",mesh,"Files/pressure_markers.xml")


# In[53]:



pD = Expression("p",p=0.0,degree=2)
bc = DirichletBC(FS.sub(0),pD,markers,1)
bcs = [bc]


TOL = 1e-8

# Set up nonlinear solver
J = derivative(F, p)
prob = NonlinearVariationalProblem(F, p, bcs, J=J, form_compiler_parameters={"optimize": True})
sol = NonlinearVariationalSolver(prob)
sol.parameters["newton_solver"]["linear_solver"] = "minres"
sol.parameters["newton_solver"]["preconditioner"] = "jacobi"
sol.parameters["newton_solver"]["absolute_tolerance"] = TOL
sol.parameters["newton_solver"]["relative_tolerance"] = TOL

for t,i_p in zip(time,pressure):
    
    pD.p = i_p
    
    sol.solve()

