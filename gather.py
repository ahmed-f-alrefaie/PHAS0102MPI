import numpy as np
import mpi4py.MPI as MPI

# Scatter example
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


if rank == 0:
    N = 20
    sendbuf = np.arange(N, dtype=np.float64)
    ave, res = divmod(sendbuf.size, nprocs)
    count = [ave + 1 if p < res else ave for p in range(nprocs)]
    count = np.array(count)
    displ = np.array([sum(count[:p]) for p in range(nprocs)])

else:
    sendbuf = None
    count = np.zeros(nprocs, dtype=np.int64)
    displ = None

# Expected_size
comm.Bcast(count, root=0)

recvbuf = np.empty(count[rank], dtype=np.float64)

comm.Scatterv([sendbuf, count, displ, MPI.DOUBLE], recvbuf, root=0)

recvbuf *= rank

print(f"My process {rank} now has {recvbuf}")

sendbuf2 = recvbuf
recvbuf2 = np.empty(count.sum(), dtype=np.float64)

comm.Gatherv(sendbuf2, [recvbuf2, count, displ, MPI.DOUBLE], root=0)

if rank == 0:
    print(f"Process {rank} received {recvbuf2} count {count.sum()}")
