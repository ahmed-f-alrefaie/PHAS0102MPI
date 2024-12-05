import numpy as np

from mpi4py import MPI


comm = MPI.COMM_WORLD

rank = comm.Get_rank()

square_points = 10000

data = np.random.uniform(low=-1.0, high=1.0, size=(square_points, 2))

in_circle = np.sum(data**2, axis=1) < 1.0


circle_points = np.sum(in_circle)


square_points = comm.reduce(square_points, op=MPI.SUM, root=0)

circle_points = comm.reduce(circle_points, op=MPI.SUM, root=0)


if square_points and circle_points:
    pi = 4.0 * circle_points / (square_points)
else:
    pi = None
print(
    f"[{rank}] Total square points {square_points} Total circle points {circle_points} Estimation of pi is {pi}"
)
