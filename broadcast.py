import mpi4py.MPI as MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()

data = None
if rank == 0:
    data = np.array([1, 2, 3, 4, 5])
    print(f"Process {rank} broadcasting {data}")
else:
    print(f"Process {rank} has data {data}")


data = comm.bcast(data, root=0)

print(f"Process {rank} received {data}")
