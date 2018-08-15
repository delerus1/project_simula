from fenics import *
 
mesh=Mesh()

f = XDMFFile(mesh.mpi_comm(),"Files/pressure_mesh.xdmf")
f.read(mesh)
f.close()



markers = MeshFunction("size_t",mesh,"Files/pressure_markers.xml")
