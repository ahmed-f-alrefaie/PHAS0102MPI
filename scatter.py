import numpy as np
import mpi4py.MPI as MPI

# Scatter example
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

N = 2


if rank == 0:
    sendbuf = np.arange(N, dtype=np.float64)
    ave, res = divmod(sendbuf.size, nprocs)
    count = np.full(nprocs, ave)
    count[:res] += 1
    displ = np.array([sum(count[:p]) for p in range(nprocs)])

    print(f"Process {rank} scattering {sendbuf}")
else:
    sendbuf = None
    count = np.zeros(nprocs, dtype=np.int64)
    displ = None

# Expected_size
comm.Bcast(count, root=0)

recvbuf = np.empty(count[rank], dtype=np.float64)

comm.Scatterv([sendbuf, count, displ, MPI.DOUBLE], recvbuf, root=0)

print(f"Process {rank} received {recvbuf} count {count[rank]}")
