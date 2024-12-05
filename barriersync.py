import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
nprocs = comm.Get_size()

for i in range(nprocs):
    if rank == i:
        print(f"Hello I am process {rank} out of {nprocs}")
    comm.Barrier()
