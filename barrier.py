import mpi4py.MPI as MPI
import time

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
nprocs = comm.Get_size()

print(f"[{rank}] Hello I am process {rank} out of {nprocs}")


if rank == 0:
    time.sleep(5)

print(f"[{rank}] Lets wait for process 0 to finish")
comm.Barrier()

print(f"[{rank}] Process 0 has finished")
