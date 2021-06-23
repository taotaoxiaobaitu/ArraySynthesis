import unittest

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.chebyshev as cbs

from models.array_synthesis.chebyshev_syn import ChebyshevPlaneSyn
from models.array_synthesis.draw import surface_3D


class ChebyshevTestCase(unittest.TestCase):
    def test_chebyshev_arithmetic(self):
        print("chebdomain:\t" + str(cbs.chebdomain))
        print("chebzero:\t" + str(cbs.chebzero))
        print("chebone\t:" + str(cbs.chebone))
        print("chebx\t:" + str(cbs.chebx))

        p1 = cbs.chebval(0.2, 4)
        print("P(0.5) = " + str(p1))

        N = 100
        chebyshevMatrix = np.zeros([N, N], dtype=float)
        for zmc in np.arange(N):
            vec = np.zeros([zmc + 1], dtype=float)
            vec[zmc] = 1.
            chebyshevMatrix[zmc, 0:zmc + 1] = cbs.cheb2poly(vec)

        print("Size of Matrix:")
        print(chebyshevMatrix.shape)

        # for zmc in np.arange(chebyshevMatrix.shape[0]):
        #     print(chebyshevMatrix[zmc])
        np.save("chebyshevCoeficientsMatrix100", chebyshevMatrix)

    def test_chebyshev_load(self):
        chebyshevMatrix = np.load("chebyshevCoeficientsMatrix100.npy")
        print(chebyshevMatrix.shape)

    def test_chebyshev_sample(self):
        sidelobe = 30
        scan = np.array([np.pi / 4, np.pi / 3])
        omega = np.array([3, 3]) / 180 * np.pi
        number = np.array([1, 1], dtype=int)

        theta = np.arange(30, 150, 1)
        phi = np.arange(-60, 60, 1)

        cbs_sample = ChebyshevPlaneSyn(sidelobe, scan, omega, number, True)
        cbs_sample.synthesis()
        cbs_sample.show()

        theta_degree = theta * np.pi / 180
        phi_degree = phi * np.pi / 180

        AF = cbs_sample.array_factor(theta_degree, phi_degree)
        AF_abs = np.abs(AF)
        AFnormal = 20 * np.log10(AF_abs / np.max(AF_abs))

        size = cbs_sample.get_size()

        gain = np.round(10 * np.log10(cbs_sample.direct), 2)

        picture_title = str(size[0]) + "*" + str(size[1]) + " Array Factor" + "(Gain: " \
                        + str(gain) + "dB )"

        surface_3D(theta, phi, AFnormal, picture_title)

    def test_chebyshev_sample_undepart(self):
        sidelobe = 30
        scan = np.array([np.pi / 4, np.pi / 3])
        omega = np.array([3, 3]) / 180 * np.pi
        number = np.array([1, 1], dtype=int)

        theta = np.arange(30, 150, 1)
        phi = np.arange(-60, 60, 1)

        cbs_sample = ChebyshevPlaneSyn(sidelobe, scan, omega, number)
        cbs_sample.synthesis()
        cbs_sample.show()

        theta_degree = theta * np.pi / 180
        phi_degree = phi * np.pi / 180

        AF = cbs_sample.array_factor(theta_degree, phi_degree)
        AF_abs = np.abs(AF)
        AFnormal = 20 * np.log10(AF_abs / np.max(AF_abs))

        size = cbs_sample.get_size()

        gain = np.round(10 * np.log10(cbs_sample.direct), 2)

        picture_title = str(size[0]) + "*" + str(size[0]) + " Array Factor" + "(Gain: " \
                        + str(gain) + "dB )"

        surface_3D(theta, phi, AFnormal, picture_title)

    def test_report_pictures(self):
        title2 = [''] * 4
        title = [''] * 4
        design = [None] * 4
        design_undepart = [None] * 4
        AFnormal_undepart = [None] * 4
        AFnormal = [None] * 4
        case = [None] * 4
        case[0] = dict(sidelobe=30,
                       scan=np.array([np.pi / 3, np.pi / 3]),
                       omega=np.array([3, 3]) / 180 * np.pi,
                       number=np.array([1, 1], dtype=int),
                       theta=np.arange(30, 150, 1),
                       phi=np.arange(-60, 60, 1)
                       )

        case[1] = dict(sidelobe=40,
                       scan=np.array([np.pi / 3, np.pi / 3]),
                       omega=np.array([3, 3]) / 180 * np.pi,
                       number=np.array([1, 1], dtype=int),
                       theta=np.arange(30, 150, 1),
                       phi=np.arange(-60, 60, 1)
                       )

        case[2] = dict(sidelobe=30,
                       scan=np.array([np.pi / 3, np.pi / 3]),
                       omega=np.array([5, 5]) / 180 * np.pi,
                       number=np.array([1, 1], dtype=int),
                       theta=np.arange(30, 150, 1),
                       phi=np.arange(-60, 60, 1)
                       )

        case[3] = dict(sidelobe=30,
                       scan=np.array([np.pi / 4, np.pi / 3]),
                       omega=np.array([3, 3]) / 180 * np.pi,
                       number=np.array([1, 1], dtype=int),
                       theta=np.arange(30, 150, 1),
                       phi=np.arange(-60, 60, 1)
                       )

        for zmc in np.arange(4):
            design[zmc] = ChebyshevPlaneSyn(case[zmc]['sidelobe'], case[zmc]['scan'], case[zmc]['omega'],
                                            case[zmc]['number'], True)
            design[zmc].synthesis()
            AF = design[zmc].array_factor(case[zmc]['theta'] * np.pi / 180, case[zmc]['phi'] * np.pi / 180)
            AFnormal[zmc] = 20 * np.log10(np.abs(AF) / np.max(np.abs(AF)))

            gain = np.round(10 * np.log10(design[zmc].direct), 2)
            title[zmc] = "Size: " + str(design[zmc].get_size()[0]) + "*" \
                         + str(design[zmc].get_size()[1]) + "  (Gain: " \
                         + str(gain) + "dB )"

            design_undepart[zmc] = ChebyshevPlaneSyn(case[zmc]['sidelobe'], case[zmc]['scan'], case[zmc]['omega'],
                                                     case[zmc]['number'])
            design_undepart[zmc].synthesis()
            AF2 = design_undepart[zmc].array_factor(case[zmc]['theta'] * np.pi / 180, case[zmc]['phi'] * np.pi / 180)
            AFnormal_undepart[zmc] = 20 * np.log10(np.abs(AF2) / np.max(np.abs(AF2)))

            gain = np.round(10 * np.log10(design[zmc].direct), 2)
            title2[zmc] = "Size: " + str(design[zmc].get_size()[0]) + "*" \
                          + str(design[zmc].get_size()[0]) + "  (Gain: " \
                          + str(gain) + "dB )"

        # fig = plt.figure(num=1, figsize=(8, 6), dpi=300)
        #
        # for zmc in [0, 1, 2, 3]:
        #     ax = fig.add_subplot(2, 2, zmc+1, projection='3d', constrained_layout=True)
        #     X, Y = np.meshgrid(case[zmc]['theta'], case[zmc]['phi'])
        #     surf = ax.plot_surface(X, Y, AFnormal[zmc], cmap='coolwarm')
        #     ax.contour(X, Y, AFnormal[zmc], zdir='z', offset=20, cmap="coolwarm")  # 生成z方向投影，投到x-y平面
        #
        #     ax.set_title(title[zmc])
        #     ax.set_xlabel('Theta(degree)')
        #     ax.set_ylabel('Phi(degree)')
        #     ax.set_zlabel('dB')
        #     fig.colorbar(surf, ax=ax, shrink=0.8, pad=0.2)

        fig, ax = plt.subplots(2, 2, figsize=(8, 8), subplot_kw={'projection': '3d'}, dpi=200)
        ax = ax.flatten()

        for zmc in [0, 1, 2, 3]:
            X, Y = np.meshgrid(case[zmc]['theta'], case[zmc]['phi'])
            surf = ax[zmc].plot_surface(X, Y, AFnormal[zmc], cmap='coolwarm')
            ax[zmc].contour(X, Y, AFnormal[zmc], zdir='z', offset=20, cmap="coolwarm")
            ax[zmc].set_title(title[zmc])
            ax[zmc].set_xlabel('Theta(degree)')
            ax[zmc].set_ylabel('Phi(degree)')
            ax[zmc].set_zlabel('dB')
            fig.colorbar(surf, ax=ax[zmc], shrink=0.8, pad=0.2)

        plt.tight_layout()
        fig.suptitle('Departure Chebyshev Array')

        fig, ax = plt.subplots(2, 2, figsize=(8, 8), subplot_kw={'projection': '3d'}, dpi=200)
        ax = ax.flatten()

        for zmc in [0, 1, 2, 3]:
            X, Y = np.meshgrid(case[zmc]['theta'], case[zmc]['phi'])
            surf = ax[zmc].plot_surface(X, Y, AFnormal_undepart[zmc], cmap='coolwarm')
            ax[zmc].contour(X, Y, AFnormal_undepart[zmc], zdir='z', offset=20, cmap="coolwarm")
            ax[zmc].set_title(title2[zmc])
            ax[zmc].set_xlabel('Theta(degree)')
            ax[zmc].set_ylabel('Phi(degree)')
            ax[zmc].set_zlabel('dB')
            fig.colorbar(surf, ax=ax[zmc], shrink=0.8, pad=0.2)

        plt.tight_layout()
        fig.suptitle('Undeparture Chebyshev Array')

        plt.show()


if __name__ == '__main__':
    unittest.main()
