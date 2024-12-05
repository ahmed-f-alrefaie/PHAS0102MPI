import mpi4py.MPI as MPI
import numpy as np
import numba as nb
import time

comm = MPI.COMM_WORLD


rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()


G = 1


def generate_initial_conditions(n):
    positions = np.random.randn(n, 3) * 10
    velocities = np.random.randn(n, 3) * 5

    masses = np.random.rand(n) * 10
    positions[0, :] = 0
    velocities[0, :] = 0
    masses[0] = 400
    return positions, velocities, masses


def generate_solar_system():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [
                57.909e9,
                0.0,
                0.0,
            ],
            [108.2e9, 0.0, 0.0],
            [149.6e9, 0.0, 0.0],
            [227.9e9, 0.0, 0.0],
            [778.5e9, 0.0, 0.0],
            [1433.5e9, 0.0, 0.0],
            [2872.5e9, 0.0, 0.0],
            [4495.1e9, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 47.87e3, 0.0],
            [0.0, 35.02e3, 0.0],
            [0.0, 29.78e3, 0.0],
            [0.0, 24.07e3, 0.0],
            [0.0, 13.07e3, 0.0],
            [0.0, 9.69e3, 0.0],
            [0.0, 6.81e3, 0.0],
            [0.0, 5.43e3, 0.0],
        ]
    )
    masses = np.array(
        [
            1.989e30,
            3.285e23,
            4.867e24,
            5.972e24,
            6.39e23,
            1.898e27,
            5.683e26,
            8.681e25,
            1.024e26,
        ]
    )
    return positions, velocities, masses


@nb.njit(parallel=True, fastmath=True, error_model="numpy")
def compute_acceleration(start_index, count, positions, masses, softening):
    accel = np.empty((count, 3), dtype=np.float64)
    n_accel = accel.shape[0]
    n_positions = positions.shape[0]

    for i_local in nb.prange(n_accel):
        accel[i_local] = 0.0
        current_accel = accel[i_local]
        i = i_local + start_index

        for j in range(0, i):
            r = positions[i] - positions[j]

            # Softening version
            r_squared = np.dot(r, r) + softening * softening
            r_soft = np.sqrt(r_squared)
            # Compute acceleration
            current_accel += -G * masses[j] * r / (r_soft * r_squared)
        for j in range(i + 1, n_positions):
            r = positions[i] - positions[j]
            r_squared = np.dot(r, r) + softening * softening
            r_soft = np.sqrt(r_squared)
            current_accel += (
                -G * masses[j] * r / (r_soft * r_squared)
            )  # Compute acceleration

        accel[i_local] = current_accel

    return accel


def integrate_euler(positions, velocities, masses, start_index, count, dt, softening):
    accel = compute_acceleration(start_index, count, positions, masses, softening)
    myslice = slice(start_index, start_index + count)

    new_velocities = velocities + accel * dt
    new_positions = positions[myslice] + new_velocities * dt

    return new_positions, new_velocities


def integrate_leapfrog(
    positions, velocities, masses, start_index, count, dt, softening
):
    """
    Perform one Leapfrog step for the local particles.
    """
    # Step 1: Compute accelerations at the current positions
    accel = compute_acceleration(start_index, count, positions, masses, softening)
    myslice = slice(start_index, start_index + count)

    # Step 2: Update velocities by half a timestep
    velocities += accel * (dt / 2)

    # Step 3: Update positions by a full timestep
    new_positions = positions[myslice] + velocities * dt

    # Step 4: Compute new accelerations at updated positions
    accel_new = compute_acceleration(start_index, count, positions, masses, softening)

    # Step 5: Update velocities by another half timestep
    new_velocities = velocities + accel_new * (dt / 2)

    return new_positions, new_velocities


# Generate and scatter data
def scatter_data(N):
    if rank == 0:
        positions, velocities, masses = generate_initial_conditions(N)

        N = positions.shape[0]
        ave, res = divmod(N, nprocs)
        count = np.full(nprocs, ave)
        count[:res] += 1
        displ = np.array([sum(count[:p]) for p in range(nprocs)])
    else:
        positions = None
        velocities = None
        masses = None
        count = np.zeros(nprocs, dtype=np.int64)
        displ = np.zeros(nprocs, dtype=np.int64)

    comm.Bcast(count, root=0)
    comm.Bcast(displ, root=0)

    N = count.sum()
    if rank != 0:
        positions = np.empty((N, 3), dtype=np.float64)
        masses = np.empty(N, dtype=np.float64)

    comm.Bcast(positions, root=0)
    comm.Bcast(masses, root=0)
    recvbuf = np.empty(shape=(count[rank], 3), dtype=np.float64)
    comm.Scatterv([velocities, count * 3, displ * 3, MPI.DOUBLE], recvbuf, root=0)
    return positions, recvbuf, masses, count, displ


def nbody(N, dt, softening, end_time):

    nsteps = int(end_time / dt) + 1

    positions, velocities, masses, count, displ = scatter_data(N)

    all_positions = []
    start = time.perf_counter()
    for i in range(nsteps):
        if rank == 0:
            print(f"Step {i}/{nsteps}")
        new_positions, new_velocities = integrate_leapfrog(
            positions, velocities, masses, displ[rank], count[rank], dt, softening
        )
        comm.Allgatherv(new_positions, [positions, count * 3, displ * 3, MPI.DOUBLE])
        velocities = new_velocities
        if rank == 0:
            all_positions.append(positions.copy())
    end = time.perf_counter()
    if rank == 0:
        print(f"Simulation took {end - start} seconds")
        print(f"Time per step { (end - start) / nsteps} seconds")

    return np.array(all_positions), masses, nsteps


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.01)

    parser.add_argument("--softening", type=float, default=1e-1)

    parser.add_argument("--end_time", type=float, default=1)

    parser.add_argument("--nthreads", type=int, default=1)

    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    nb.set_num_threads(args.nthreads)

    positions, masses, nsteps = nbody(args.N, args.dt, args.softening, args.end_time)

    if rank == 0 and args.output is not None:
        np.savez(args.output, positions=positions, masses=masses)
