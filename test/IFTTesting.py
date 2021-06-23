import unittest

import matplotlib.pyplot as plt
import numpy as np

import models.array_synthesis.draw as dw
from models.array_synthesis.IFTSynthesis import IFTSynthesis as IFT


class IFTTestCase(unittest.TestCase):
    def test_circular_equal_exciation(self):
        sidelobe = 30
        scan = np.array([np.pi / 3, np.pi / 3])
        omega = np.array([3, 3]) / 180 * np.pi
        interval = np.array([0.45, 0.45], dtype=float)
        aperture = np.array([22, 22], dtype=float)
        theta = np.arange(30, 150, 1)
        phi = np.arange(-60, 60, 1)

        # calculate factor
        sample = IFT(sidelobe, interval, aperture)

        AF = sample.array_factor(theta * np.pi / 180, phi * np.pi / 180)
        AF_abs = np.abs(AF)
        AFnormal = 20 * np.log10(AF_abs / np.max(AF_abs))

        sample.show()

        # Draw pictures
        fig = plt.figure(num=1, figsize=(10, 8), dpi=300)

        # subfigure1

        dw.IFT_MaskArray(radius=sample.aperture[0] / 2, number=sample.numberUV[0], interval=sample.interval[0],
                         ax=fig.add_subplot(221))

        # subfigure2

        ax = fig.add_subplot(222, projection='3d')
        Theta, Phi = np.meshgrid(theta, phi)
        AFnormal[AFnormal < -90] = -90
        surf = ax.plot_surface(Theta, Phi, AFnormal, cmap='coolwarm')
        # ax.contour(Theta, Phi, AFnormal, zdir='z', levels=8, offset=20, cmap="coolwarm")
        picture_title = "(Gain: " + str(np.round(10 * np.log10(sample.max_gain), 2)) + " dB )"
        ax.set_title(picture_title, family='times new roman', fontsize=15)
        ax.set_ylabel('Theta(degree)', family='times new roman')
        ax.set_xlabel('Phi(degree)', family='times new roman')
        ax.set_zlabel('/dB', family='times new roman')
        ax.set_zlim([-90, 0])

        fig.colorbar(surf, shrink=0.7, pad=0.15)

        # subfigure3
        ax = fig.add_subplot(223)
        y = AF_abs[:, int(AF_abs.shape[1] / 2)]
        y = 20 * np.log10(y / y.max())

        ax.plot(phi, y)

        ax.set_title(r"$\phi=0$", family="times new roman", style='italic')

        ax.set_xticks([-60, -45, -30, -15, -3, 3, 15, 30, 45, 60])
        ax.set_xlabel(r'$\theta$ (degree)', family="times new roman", style='italic', fontsize=15)
        ax.set_xlim([-60, 60])

        ax.set_ylabel('normalized pattern/dB', family="times new roman", fontsize=15)
        ax.set_ylim([-70, 0])
        ax.set_yticks([-70, -60, -50, -40, -30, -20, -10, -3, 0])

        ax.grid(True)

        # subfigure4
        ax = fig.add_subplot(224)
        y = AF_abs[int(len(AF_abs) / 2)]
        y = 20 * np.log10(y / y.max())

        ax.plot(phi, y)

        ax.set_title(r"$\phi=\pi/2$", family="times new roman", style='italic')

        ax.set_xticks([-60, -45, -30, -15, -3, 3, 15, 30, 45, 60])
        ax.set_xlabel(r'$\theta$ (degree)', family="times new roman", style='italic', fontsize=15)
        ax.set_xlim([-60, 60])

        ax.set_ylabel('normalized pattern/dB', family="times new roman", fontsize=15)
        ax.set_ylim([-70, 0])
        ax.set_yticks([-70, -60, -50, -40, -30, -20, -10, -3, 0])

        ax.grid(True)

        fig.suptitle(str(sample.aperture[0]) + r" $\lambda$ circular aperture array", family="times new roman",
                     fontsize=15)

        plt.show()


if __name__ == '__main__':
    unittest.main()
