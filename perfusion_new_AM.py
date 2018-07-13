
# coding: utf-8

# In[14]:


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
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


# Importing the mesh
mesh=Mesh()

f = XDMFFile(mesh.mpi_comm(),"pressure_mesh.xdmf")
f.read(mesh)
f.close()

plot(mesh)


# In[16]:


xdmffile_p1 = XDMFFile(mesh.mpi_comm(), 'perfusion_results/p1.xdmf')
xdmffile_p2 = XDMFFile(mesh.mpi_comm(), 'perfusion_results/p2.xdmf')
xdmffile_p3 = XDMFFile(mesh.mpi_comm(), 'perfusion_results/p3.xdmf')


# In[17]:


# Importing initial pressure

path = "coronary_pressure.csv"

df = pd.read_csv(path,names=['time','pressure'])


# In[18]:


time = np.array(df['time'])
pressure = np.array(df['pressure']) # mmHg, 0.133322368 mmHg = 1 Kpa
pressure = pressure * 0.133322368 # KPa


# In[19]:


num_time_steps = 100
new_time = np.linspace(0,1,num_time_steps)
func_interpol_p = interp1d(time,pressure)

new_pressure = func_interpol_p(new_time)


time = new_time
pressure = new_pressure


# In[20]:


plt.plot(time,pressure)
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('Initial pressure')


# In[21]:


# Define function spaces. Define trial and test functions

el = tetrahedron

P = FiniteElement('P',el,2)


element = MixedElement([P,P,P])
FS = FunctionSpace(mesh,element)


# Define test functions
q1,q2,q3 = TestFunctions(FS)
v1, v2, v3 = TestFunctions(FS)

K1 = Constant(1) #(mm^2)/(kPa*s)
K2 = Constant(10) #(mm^2)/kPa*s
K3 = Constant(20) #(mm^2)/kPa*s
dt = Constant(0.01)
diff_o2 = 0.9*10**-9
D = Constant(diff_o2)

beta12 = Constant(0.02) #1/(kPa*s)
beta23 = Constant(0.05) #1/(kPa*s)

p = Function(FS)
c = Function(FS)
c1, c2, c3 = split(c)
c_n = Function(FS)
c_n1, c_n2, c_n3 = split(c_n)

p1, p2, p3 = split(p)
k = Constant(dt)
S3 = - Constant(0.1)*(p3-Constant(3.0)) #Sink term in the third apartment
F = -K1 * dot(grad(p1), grad(q1))*dx + -K2 * dot(grad(p2), grad(q2))*dx + -K3 * dot(grad(p3), grad(q3))*dx     + dot(beta12*(p1-p2),q1)*dx + dot(beta12*(p2-p1),q2)*dx + dot(beta23*(p2-p3),q2)*dx + dot(beta23*(p3-p2),q3)*dx    - dot(S3,q3)*dx
F2 = ((c1 - c1_n) / k)*v1*dx + dot(v_d1, grad(c1))*v1*dx - D*dot(grad(c1), grad(v1))*dx
F3 = ((c2 - c2_n) / k)*v2*dx + dot(v_d2, grad(c2))*v2*dx - D*dot(grad(c2), grad(v2))*dx
F4 = ((c3 - c3_n) / k)*v3*dx + dot(v_d3, grad(c3))*v3*dx - D*dot(grad(c3), grad(v3))*dx

markers = MeshFunction("size_t",mesh,"pressure_markers.xml")


# In[22]:



pD = Expression("p",p=0.0,degree=2)
bc = DirichletBC(FS.sub(0),pD,markers,1)
bcs = [bc]


TOL = 1e-8

# Set up nonlinear solver
#J = derivative(F, p)
#prob = NonlinearVariationalProblem(F, p, bcs, J=J, form_compiler_parameters={"optimize": True})
#sol = NonlinearVariationalSolver(prob)
#sol.parameters["newton_solver"]["linear_solver"] = "minres"
#sol.parameters["newton_solver"]["preconditioner"] = "jacobi"
#sol.parameters["newton_solver"]["absolute_tolerance"] = TOL
#sol.parameters["newton_solver"]["relative_tolerance"] = TOL

for t,i_p in zip(time,pressure):
    
    pD.p = i_p
    
#    sol.solve()
    solve(F==0, p, bc)
	p1,p2,p3 = p.split()   
	v_d1 = project(grad(-K1*grad(p1)),FS)
	v_d2 = project(grad(-K2*grad(p2)),FS)
	v_d3 = project(grad(-K3*grad(p3)),FS)
	solve(F2==0, c1)
	solve(F3==0, c2)
	solve(F4==0, c3)
    xdmffile_p1.write(p1,t)
    xdmffile_p2.write(p2,t)
    xdmffile_p3.write(p3,t)
	c_n1.assign(c1)
	c_n2.assign(c2)
	c_n3.assign(c3)
	
xdmffile_p1.close()
xdmffile_p2.close()
xdmffile_p3.close()