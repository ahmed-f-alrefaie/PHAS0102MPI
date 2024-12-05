import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse
import tqdm


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", type=str, required=True)
    parser.add_argument("-o", type=str, required=True)
    args = parser.parse_args()

    data = np.load(args.i)
    masses = data["masses"]
    positions = data["positions"]
    nsteps = positions.shape[0]
    N = positions.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # size should be proportional to mass

    sc = ax.scatter(
        positions[0][:, 0], positions[0][:, 1], positions[0][:, 2], s=masses * 1
    )
    trails = []
    # for i in range(N):
    #     trails.append(ax.plot([], [], [])[0])
    tqdm_bar = tqdm.tqdm(total=nsteps)

    com = np.sum(positions[0] * masses[:, None], axis=0) / np.sum(masses)
    std = np.sqrt(
        np.sum((positions[0] - com) ** 2 * masses[:, None], axis=0) / np.sum(masses)
    )

    x_min = com[0] - 2 * std[0]

    x_max = com[0] + 2 * std[0]

    # x_min = -2e11
    # x_max = 2e11

    # relimit the axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_zlim(x_min, x_max)

    def update(i):
        sc._offsets3d = (
            positions[i][:, 0],
            positions[i][:, 1],
            positions[i][:, 2],
        )
        tqdm_bar.update(1)

        com = np.sum(positions[0] * masses[:, None], axis=0) / np.sum(masses)

        x_min = com[0] - 2 * std[0]

        x_max = com[0] + 2 * std[0]

        y_min = com[1] - 2 * std[0]

        y_max = com[1] + 2 * std[0]

        z_min = com[2] - 2 * std[0]

        z_max = com[2] + 2 * std[0]

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Set size

        # for j in range(N):
        #     trails[j].set_data(positions[:i, j, 0], positions[:i, j, 1])
        #     trails[j].set_3d_properties(positions[:i, j, 2])
        # Put x,y,z limits in center of mass and std dev

        return sc

    ani = animation.FuncAnimation(fig, update, frames=nsteps, interval=20)
    ani.save(args.o, writer="ffmpeg", fps=60)
    tqdm_bar.close()


if __name__ == "__main__":
    main()
