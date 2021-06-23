import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve2d

import models.array_synthesis.draw as dw
from models.array_synthesis.IFTSynthesis import IFTSynthesis as IFT


class LoadMatTestCase(unittest.TestCase):
    def test_loadmat(self):
        filename = './number25.mat'
        data = loadmat(filename)

        number = data['number_diameter'].flatten()[0]
        radius = data['radius'].flatten()[0]
        interval = data['dx'].flatten()[0]
        theta = data['theta'][0]
        phi = data['phi'][0]
        AF = data['AF']

        fig = plt.figure(num=1, figsize=(10, 8), dpi=300)

        dw.IFT_MaskArray(radius=radius, number=number, interval=interval, ax=fig.add_subplot(221))
        surf = dw.IFT_3D_surface(theta=theta, phi=phi, AF=AF, ax=fig.add_subplot(222, projection='3d'), title='')
        fig.colorbar(surf, shrink=0.7, pad=0.15)

        dw.IFT_line_plot(x=theta, y=AF[0], ax=fig.add_subplot(223), title=r"$\phi=0$")
        dw.IFT_line_plot(x=theta, y=AF[int(len(AF) / 2)], ax=fig.add_subplot(224), title=r"$\phi=\pi/2$")

        fig.suptitle(str(radius * 2) + r"$\lambda$ circle aperture array", family="times new roman", fontsize=15)

        plt.show()

    def test_circular_kernel(self):
        # 8 circular kernels (kernel is inversed in fact)
        picture_size = 1024

        window = np.zeros([8, 3, 3], dtype=int)
        window[:, 1, 1] = 1
        window[0, 0, 0] = -1
        window[1, 0, 1] = -1
        window[2, 0, 2] = -1
        window[3, 1, 0] = -1
        window[4, 1, 2] = -1
        window[5, 2, 0] = -1
        window[6, 2, 1] = -1
        window[7, 2, 2] = -1

        # load picture
        # picture = np.random.rand(picture_size, picture_size)
        picture = generate_test_picture(num=picture_size)
        res_sign = np.zeros([8, picture.shape[0], picture.shape[1]], dtype=int)

        for zmc in np.arange(8):
            temp = convolve2d(picture, window[zmc, :, :], mode="same", boundary="symm")
            res_sign[zmc, 1:-1, 1:-1] = np.sign(temp)[1:-1, 1:-1]

        flux = np.einsum("ijk->jk", res_sign)

        peak_ind = np.where(flux == 8)
        peak_val = picture[peak_ind]
        peak_num = len(peak_val)
        peak_max = np.msort(peak_val)[-1]

        null_ind = np.where(flux == -8)
        null_val = picture[null_ind]
        null_num = len(null_val)
        null_min = np.msort(null_val)[0]

        # peak_num = np.einsum("ij->", np.where(flux == 8, 1, 0))
        # null_num = np.einsum("ij->", np.where(flux == -8, 1, 0))

        # Draw
        fig = plt.figure(num=1, figsize=(10, 10), dpi=100)

        ax = fig.add_subplot(221)
        # Draw picture
        im1 = ax.imshow(picture, cmap='bwr')
        ax.set_title("Original Picture")

        ax = fig.add_subplot(222)
        im2 = ax.imshow(flux, cmap='bwr')
        ax.set_title("The flux (Peak and null are " + str(peak_num) + " and " + str(null_num) + ")")

        XX, YY = np.meshgrid(np.arange(picture_size), np.arange(picture_size))

        ax = fig.add_subplot(223, projection='3d')
        surf = ax.plot_surface(X=XX, Y=YY, Z=picture, cmap='bwr')
        # ax.contour(X, Y, val, zdir='z', levels=8, offset=20, cmap="coolwarm")  # 生成z方向投影，投到x-y平面
        ax.set_title('3D Original Picture', family="times new roman", fontsize=15)

        ax = fig.add_subplot(224, projection='3d')
        # ax.scatter(peak_ind[0], peak_ind[1], peak_val)
        ax.stem(peak_ind[0], peak_ind[1], peak_val, linefmt='grey', markerfmt='D', bottom=-10)
        # surf = ax.stem(X=XX, Y=YY, Z=flux, cmap='bwr')
        # ax.contour(X, Y, val, zdir='z', levels=8, offset=20, cmap="coolwarm")  # 生成z方向投影，投到x-y平面
        ax.set_title('3D Flux', family="times new roman", fontsize=15)

        plt.show()

        self.assertAlmostEqual(np.max(picture), peak_max, 4, "Peak Max is not Equal", 1e-3)
        self.assertAlmostEqual(np.min(picture), null_min, 4, "Null min is not Equal", 1e-3)

        print("max and min: " + str(np.max(picture)) + "\t" + str(np.min(picture)))

    def test_circular_equal_exciation(self):
        sidelobe = 30
        scan = np.array([np.pi / 3, np.pi / 3])
        omega = np.array([3, 3]) / 180 * np.pi
        interval = np.array([0.45, 0.45], dtype=float)
        aperture = np.array([20, 20], dtype=float)

        sample = IFT(sidelobe, interval, aperture)

        theta = np.arange(30, 150, 1)
        phi = np.arange(-60, 60, 1)
        AF = sample.array_factor(theta * np.pi / 180, phi * np.pi / 180)

        AF_abs = np.abs(AF)
        AFnormal = 20 * np.log10(AF_abs / np.max(AF_abs))

        sample.show()
        picture_title = "(Gain: " + str(np.round(10 * np.log10(sample.max_gain), 2)) + " dB )"
        dw.surface_3D(theta, phi, AFnormal, picture_title)


def generate_test_picture(num=1024):
    fix_point = int(num / 2)
    line = 2 * np.abs(np.arange(-num / 2, num / 2, 1)) / num
    line2 = line * line
    temp = np.einsum("i,j->ij", line2, np.ones([num]))
    radius = 8 * np.pi * np.sqrt(temp + temp.T)
    radius[fix_point, fix_point] = 1
    sina = np.sin(radius) / radius
    sina[fix_point, fix_point] = 1
    mag = 10
    return mag * mag * sina


if __name__ == '__main__':
    unittest.main()
