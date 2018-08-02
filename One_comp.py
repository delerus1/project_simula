
# coding: utf-8

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

# Create mesh

nx = 12
ny = 12
nz = 12

mesh = UnitCubeMesh(nx,ny,nz) # 10 is number of intervals Omega is divided into
plot(mesh);

xdmffile_p = XDMFFile(mesh.mpi_comm(), 'one_comp_results/pressure.xdmf')
xdmffile_v = XDMFFile(mesh.mpi_comm(), 'one_comp_results/velocity.xdmf')
xml_mesh = File('one_comp_results/mesh.xml')
xml_mesh << mesh

# Define function spaces. Define trial and test functions

V = FunctionSpace(mesh,"Lagrange",2)
q = TestFunction(V)
p = TrialFunction(V)

p_ = Function(V)


# Define boundaries
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[2], 1)'

# Define inflow profile
inflow_profile = ('2*pi*sin(2*pi*x[1])')

bcp_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)

bcp_outflow = DirichletBC(V, Constant(0), outflow)

bcp = [bcp_inflow,bcp_outflow]

K = Constant(0.5)

f = Constant(0)
s = Constant(0)


p = Function(V)
F = -K * dot(grad(p), grad(q))*dx + f*q*dx + s*q*dx

#Solve equations
solve(F==0, p, bcp)

W = VectorFunctionSpace(mesh, "P", 2)
w = project(-K*grad(p_), W)


xdmffile_p.write_checkpoint(p, 'p')
xdmffile_v.write_checkpoint(w, 'w')

xdmffile_p.close()
xdmffile_v.close()

