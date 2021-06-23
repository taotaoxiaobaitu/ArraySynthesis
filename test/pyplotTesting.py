import unittest

import matplotlib.pyplot as plt
import numpy as np


class MatplotlibTestCase(unittest.TestCase):
    cmaps = [('Perceptually Uniform Sequential', [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
             ('Sequential', [
                 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
             ('Sequential (2)', [
                 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                 'hot', 'afmhot', 'gist_heat', 'copper']),
             ('Diverging', [
                 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
             ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
             ('Qualitative', [
                 'Pastel1', 'Pastel2', 'Paired', 'Accent',
                 'Dark2', 'Set1', 'Set2', 'Set3',
                 'tab10', 'tab20', 'tab20b', 'tab20c']),
             ('Miscellaneous', [
                 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
                 'gist_ncar'])]

    def test_simple_example(self):
        fig, axs = plt.subplots(1, 2)
        axs[0].plot([1, 2, 3, 4], [1, 4, 2, 3])

        x = np.linspace(0, 2, 100)

        plt.plot(x, x, label="linear")
        plt.plot(x, x * x, label="quadratic")
        plt.plot(x, x ** 3, label="cubic")
        plt.xlabel("t/s")
        plt.ylabel("distance/m")
        plt.title("Movement")
        plt.legend()

        plt.show()

    def test_multiple_plot(self):
        x1 = np.linspace(0, 5)
        x2 = np.linspace(0, 2)

        y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
        y2 = np.sin(2 * np.pi * x2)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle("Two plots in one picture")

        ax1.plot(x1, y1, 'o-')
        ax1.set_ylabel("Damped oscillation")

        ax2.plot(x2, y2, '.-')
        ax2.set_xlabel('times/s')
        ax2.set_ylabel('Undamped')

        # plt.subplot(2, 1, 1)
        # plt.plot(x1, y1, 'o-')
        # plt.title('Tow plots in one picture')
        # plt.ylabel("Damped oscillation")
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(x2, y2, '.-')
        # plt.xlabel('time/s')
        # plt.ylabel("Undamped")

        plt.show()

    def test_simple_picture(self):
        delta = 0.025
        x = y = np.arange(-3., 3., delta)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X ** 2 - Y ** 2)
        Z2 = np.exp(-(X - -1) ** 2 - (Y - -1) ** 2)
        Z = (Z1 - Z2) * 2

        fig, ax = plt.subplots()
        im = ax.imshow(Z, interpolation='bilinear', cmap='rainbow',
                       origin='lower', extent=[-3, 3, -3, 3],
                       vmax=abs(Z).max(), vmin=-abs(Z).max())

        plt.show()

    def test_polar_plot(self):
        r = np.arange(0, 2, 0.001)
        theta = 2 * np.pi * r

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(theta, r)
        ax.set_rmax(2)
        ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)

        ax.set_title(r'A line plot on a polar axis with $(\theta, \rho)$')
        plt.show()

    def test_zoom_figure(self):
        delta = 0.001
        x = y = np.arange(-3., 3., delta)
        extent = [-3, 3, -3, 3]
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X ** 2 - Y ** 2)
        Z2 = np.exp(-(X - -1) ** 2 - (Y - -1) ** 2)
        Z = (Z1 - Z2) * 2

        fig, ax = plt.subplots()

        im = ax.imshow(Z, interpolation='bilinear', cmap='rainbow',
                       origin='lower', extent=extent,
                       vmax=abs(Z).max(), vmin=-abs(Z).max())

        # inset axes
        axins = ax.inset_axes([0.6, 0.6, 0.4, 0.4])  # location and enlarge factor
        axins.imshow(Z, extent=extent, origin='lower', cmap='rainbow')
        # subregion of the original image
        x1, x2, y1, y2 = -0.75, 0.25, -1.25, -0.25
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels('')
        axins.set_yticklabels('')

        ax.indicate_inset_zoom(axins, edgecolor="black")

        plt.show()

    def test_contour_figure(self):

        generate_data = lambda x, y: (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

        number = 1024
        x = np.linspace(-3, 3, number)
        y = np.linspace(-3, 3, number)

        X, Y = np.meshgrid(x, y)

        fig = plt.figure(num=1, figsize=(8, 6), dpi=100)

        ax = fig.add_subplot(111)
        # Draw picture
        ax.contourf(X, Y, generate_data(X, Y), 15, alpha=0.75, cmap='hot')
        # Draw line. Here value 8 is the gradient of height
        C = ax.contour(X, Y, generate_data(X, Y), 15, colors='black', linewidth=.5)

        ax.clabel(C, inline=True, fontsize=10)

        ax.set_xticks(())
        ax.set_yticks(())

        plt.show()

    def test_colormap_reference(self):
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        for cmap_category, cmap_list in self.cmaps:
            nrows = len(self.cmaps)
            figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
            fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
            fig.subplots_adjust(top=1 - .35 / figh, bottom=.15 / figh, left=0.2, right=0.99)

            axs[0].set_title(cmap_category + ' colormaps', fontsize=14)

            for ax, name in zip(axs, cmap_list):
                ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
                ax.text(-.01, .5, name, va='center', ha='right', fontsize=10, transform=ax.transAxes)

            for ax in axs:
                ax.set_axis_off()

        plt.show()

    def test_mask_plot(self):
        phi = np.linspace(0, 2 * np.pi, 360 * 4)
        radius = 8
        interval = 0.5
        number = 33

        # array mask

        half = int(number / 2)
        mask_line = np.arange(-half, half + 1, 1) * interval
        temp_mask = np.einsum("i,j->ij", np.ones([2 * half + 1], dtype=float), mask_line * mask_line)
        dis_mask = temp_mask + temp_mask.T - radius * radius

        array_mask = np.array(np.where(dis_mask > 0, 0, 1), dtype=np.int8)

        idx, idy = np.where(array_mask == 1)

        # draw
        fig = plt.figure(num=1, figsize=(6, 6), dpi=200)

        ax = fig.add_subplot(111)

        circle = ax.fill(radius * np.cos(phi), radius * np.sin(phi), color='lightblue', zorder=0)
        elements = ax.scatter(x=mask_line[idx], y=mask_line[idy], s=50, c='gray', zorder=1)

        # Arrow
        arrow_width = 0.03
        arrow_head_width = 0.3
        arrow_head_length = 0.5
        fontsize = 15

        ax.arrow(x=0, y=0, dx=radius - 0.5, dy=0,
                 width=arrow_width, head_width=arrow_head_width, head_length=arrow_head_length, color='k')
        ax.text(x=radius / 2, y=0.3, s="r", size=fontsize, family='times new roman', style='italic')

        # ax.annotate(text=str(radius*2)+r'$\lambda$',
        #             xy=(-radius, 0),
        #             xytext=(radius, 0),
        #             arrowprops=dict(arrowstyle='<->'))

        ax.set_aspect('equal', 'box')
        ax.set_axis_off()
        ax.grid(True)

        ax.set_title("Mask of Array with " + str(len(idx)) + " elements", fontsize=20, family='times new roman')

        plt.show()




if __name__ == '__main__':
    unittest.main()
