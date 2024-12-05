import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
nprocs = comm.Get_size()

print(f"Hello I am process {rank} out of {nprocs}")

print("We all execute the same thing!")

if rank == 0:
    print("But I am the better!")
