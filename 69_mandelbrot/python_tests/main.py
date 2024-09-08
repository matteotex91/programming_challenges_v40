import sys
import numpy as np
from time import time
from matplotlib import colormaps
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backend_bases import KeyEvent

re0 = -1
re1 = 1
cx0 = -1
cx1 = 1
add_const = 0.05
zoom_const = 1.1
N = 250


def replot_fractal():
    global re0, re1, cx0, cx1, N
    real_range = np.linspace(re0, re1, N).T
    comp_range = np.linspace(cx0, cx1, N)
    real = np.resize(real_range, (N, N))
    comp = np.resize(comp_range, (N, N)).T
    mat = real + 1j * comp

    include_map = np.ones_like(mat)
    abs_mat = np.abs(mat)
    iterMat = np.zeros_like(mat)
    new_divergent_numbers = 1

    while new_divergent_numbers != 0:
        indexes = np.where(include_map == 1)
        iterMat[indexes] = np.power(iterMat[indexes], 2) + mat[indexes]
        abs_mat = np.abs(iterMat)
        new_divergent_numbers = np.sum(
            np.where(
                np.logical_and(
                    abs_mat > np.average(abs_mat) + np.std(abs_mat), include_map == 1
                )
            )
        )
        include_map[
            np.where(
                np.logical_and(
                    abs_mat > np.average(abs_mat) + np.std(abs_mat), include_map == 1
                )
            )
        ] = 0
    min_abs = np.average(abs_mat)
    std_abs = np.std(abs_mat)

    conv_mat = np.zeros_like(abs_mat)
    conv_mat[np.where(abs_mat < min_abs + std_abs)] = 1

    # ax.pcolormesh(np.abs(conv_mat))
    ax.pcolormesh(np.abs(abs_mat))
    fig.show()


def interactive_plot():
    global fig, ax
    fig, ax = plt.subplots()

    def on_release(event: KeyEvent):
        global re0, re1, cx0, cx1, add_const, zoom_const
        print(event.key)
        match event.key:
            case "right":
                re0 += add_const
                re1 += add_const
            case "up":
                cx0 += add_const
                cx1 += add_const
            case "down":
                cx0 -= add_const
                cx1 -= add_const
            case "left":
                re0 -= add_const
                re1 -= add_const
            case "-":
                add_const /= 2
                zoom_const = 1 + (zoom_const - 1) / 2
            case "+":
                add_const *= 2
                zoom_const = 1 * (zoom_const - 1) * 2
            case "z":
                new_re0 = (re1 + re0) / 2 - (re1 - re0) / 2 / zoom_const
                new_re1 = (re1 + re0) / 2 + (re1 - re0) / 2 / zoom_const
                new_cx0 = (cx1 + cx0) / 2 - (cx1 - cx0) / 2 / zoom_const
                new_cx1 = (cx1 + cx0) / 2 + (cx1 - cx0) / 2 / zoom_const
                re0 = new_re0
                re1 = new_re1
                cx0 = new_cx0
                cx1 = new_cx1

            case "x":
                new_re0 = (re1 + re0) / 2 - (re1 - re0) / 2 * zoom_const
                new_re1 = (re1 + re0) / 2 + (re1 - re0) / 2 * zoom_const
                new_cx0 = (cx1 + cx0) / 2 - (cx1 - cx0) / 2 * zoom_const
                new_cx1 = (cx1 + cx0) / 2 + (cx1 - cx0) / 2 * zoom_const
                re0 = new_re0
                re1 = new_re1
                cx0 = new_cx0
                cx1 = new_cx1
        replot_fractal()

    replot_fractal()
    conn_id = fig.canvas.mpl_connect("key_release_event", on_release)
    plt.show()
    fig.canvas.mpl_disconnect(conn_id)


def test_plot():
    N = 250
    real_range = np.linspace(-1, 1, N).T
    comp_range = np.linspace(-1, 1, N)
    real = np.resize(real_range, (N, N))
    comp = np.resize(comp_range, (N, N)).T
    mat = real + 1j * comp

    include_map = np.ones_like(mat)
    abs_mat = np.abs(mat)
    iterMat = np.zeros_like(mat)

    t0 = time()
    divergent_list = []
    for i in range(500):
        indexes = np.where(include_map == 1)
        iterMat[indexes] = np.power(iterMat[indexes], 2) + mat[indexes]
        abs_mat = np.abs(iterMat)
        new_divergent_numbers = np.sum(
            np.where(
                np.logical_and(
                    abs_mat > np.average(abs_mat) + np.std(abs_mat), include_map == 1
                )
            )
        )
        print(new_divergent_numbers)
        if new_divergent_numbers == 0:
            break
        divergent_list.append(new_divergent_numbers)
        include_map[
            np.where(
                np.logical_and(
                    abs_mat > np.average(abs_mat) + np.std(abs_mat), include_map == 1
                )
            )
        ] = 0
    # plt.plot(divergent_list)
    # plt.show()
    print(time() - t0)

    min_abs = np.average(abs_mat)
    std_abs = np.std(abs_mat)

    conv_mat = np.zeros_like(abs_mat)
    conv_mat[np.where(abs_mat < min_abs + std_abs)] = 1

    plt.pcolormesh(np.abs(conv_mat))
    plt.show()


comp_num = 0 + 0.0j
add_const = 0.05


def replot():
    abs_list = []
    ax.clear()
    z = 0
    for i in range(100):
        abs_list.append(np.abs(z))
        z = np.power(z, 2) + comp_num
    plt.cla()
    ax.plot(abs_list)
    fig.show()
    # plt.show()


def convergence_plot():
    global fig, ax, comp_num
    fig, ax = plt.subplots()

    def onclick(event: KeyEvent):
        global comp_num, add_const
        print(event.key)
        match event.key:
            case "right":
                comp_num += add_const
            case "up":
                comp_num += add_const * 1j
            case "down":
                comp_num -= add_const * 1j
            case "left":
                comp_num -= add_const
            case "-":
                add_const /= 2
            case "+":
                add_const *= 2

        replot()

    conn_id = fig.canvas.mpl_connect("key_release_event", onclick)

    plt.show()

    fig.canvas.mpl_disconnect(conn_id)


if __name__ == "__main__":
    # test_plot()
    # convergence_plot()
    interactive_plot()
