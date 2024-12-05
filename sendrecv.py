import numpy as np
import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()


if rank == 0:
    data = np.arange(100, dtype=np.float64)
    print(f"Process {rank} sending {data}")
    comm.Send(data, dest=1, tag=13)
elif rank == 1:
    data = np.empty(100, dtype=np.float64)
    comm.Recv(data, source=0, tag=13)
    print(f"Process {rank} received {data}")
