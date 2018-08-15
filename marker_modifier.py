from fenics import *


def marker_modifier(markers, values, output_filename):
    """
    Modifies markers for the left ventrical mesh used at the SSCP18 project

    param:
        markers: list of the Cell_id of the markers to be changed
        values: list of the new values for the markers
    """

    if output_filename[-4:] != 'xdmf':
        print("you have to save the resulst as a xdmf file")
    elif output_filename == "Files/pressure_markers.xdmf":
        print("You must choose another output_filename")
    else:

        mesh = Mesh()
        f =  XDMFFile(mesh.mpi_comm(),"Files/pressure_mesh.xdmf")
        f.read(mesh)
        f.close()

        mf = MeshFunction("size_t", mesh, "Files/pressure_markers.xml")

        for m,v in zip(markers,values):
            mf.array()[m] = v

        xd = XDMFFile(mesh.mpi_comm(),output_filename)
        xd.write(mf)
        xd.close()
        print("The file was created")


if __name__ == "__main__":
    

    markers = [6172,6205,6258,6260,6262,6292,6338]
    values = [22,22,22,22,22,22,22]
    output_filename = "Files/modified_markers.xdmf"
    marker_modifier(markers=markers,values=values,output_filename=output_filename)
