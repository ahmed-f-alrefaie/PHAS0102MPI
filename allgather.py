import numpy as np
import mpi4py.MPI as MPI

# Scatter example
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


if rank == 0:
    N = 22
    sendbuf = np.arange(N, dtype=np.float64)
    ave, res = divmod(sendbuf.size, nprocs)
    count = np.full(nprocs, ave)

    count[:res] += 1
    print(count)
    displ = np.array([sum(count[:p]) for p in range(nprocs)])

else:
    sendbuf = None
    count = np.zeros(nprocs, dtype=np.int64)
    displ = np.zeros(nprocs, dtype=np.int64)

# Expected_size
comm.Bcast(count, root=0)
comm.Bcast(displ, root=0)
# print(f"Process {rank} has count {count[rank]}")
recvbuf = np.empty(count[rank], dtype=np.float64)

comm.Scatterv([sendbuf, count, displ, MPI.DOUBLE], recvbuf, root=0)
# print(f"Process {rank} received {recvbuf} count {count[rank]}")
recvbuf *= rank

# print(f"My process {rank} now has {recvbuf}")


sendbuf2 = recvbuf
recvbuf2 = np.empty(count.sum(), dtype=np.float64)

comm.Allgatherv(sendbuf2, [recvbuf2, count, displ, MPI.DOUBLE])

print(f"Process {rank} received {recvbuf2} count {count.sum()}")
