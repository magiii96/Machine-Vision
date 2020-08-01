from IPython import display
import math
import matplotlib.pyplot as plt
import numpy as np


def plot_binary_vector(vector, axes):
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    axes.set_title('{:d}{:d}{:d}{:d}{:d}'.format(*vector))
    axes.imshow(vector.reshape(1, -1), cmap='gray', vmin=0, vmax=1)


def get_gaussian_2sd(mean, covariance, angle):
    vec = np.array([[np.cos(angle), np.sin(angle)]])

    factor = 4 / (vec @ np.linalg.inv(covariance) @ vec.T)

    location = vec * np.sqrt(factor) + mean
    return tuple(location.squeeze())


def draw_gaussian_outline_2d(mean, std):
    angle_step = 0.01

    for angle in np.arange(0, 2 * np.pi, angle_step):
        point_1 = get_gaussian_2sd(mean, std, angle)
        point_2 = get_gaussian_2sd(mean, std, angle + angle_step)

        plt.plot(*zip(point_1, point_2), 'k-')


def draw_gaussian(mean, std):
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.xlim([-5, 5])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.ylim([-5, 5])
    draw_gaussian_outline_2d(mean, std)


def display_image(image, iteration):
    ax = plt.gca()
    ax.imshow(image, cmap='gray')
    ax.set_title('Iteration {:4d}'.format(iteration))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    display.display(plt.gcf())
    display.clear_output(wait=True)


def split(full_list, n):
    segment_length = math.ceil(len(full_list) / n)
    for i in range(0, len(full_list), segment_length):
        yield full_list[i:i + segment_length]


class display_table:
    def __init__(self, column_names, splits=4):
        self.column_names = column_names
        self.data = []
        self.splits = splits

    def add_row(self, data):
        self.data.append(data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        fig, axs = plt.subplots(1, self.splits, figsize=(15, 1))
        for ax, data in zip(axs, split(self.data, self.splits)):
            ax.axis('off')
            the_table = ax.table(cellText=data, colLabels=self.column_names, loc='center')
            the_table.set_fontsize(14)
            the_table.scale(1, 2)
            cells = the_table.properties()["celld"]
            for i in range(len(data) + 1):
                cells[i, 0]._loc = 'center'
