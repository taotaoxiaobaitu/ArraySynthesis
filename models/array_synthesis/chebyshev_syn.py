# -*- coding:utf-8 -*-
# Author : ZhengMX
# Data : 2021/4/17 16:37
# Project : ArrarySynthesis
# FileName : Core
# Cooperation : 265

import math
from enum import Enum

import numpy as np


class Source(Enum):
    Equiv = 1
    Chebyshev = 2
    Taylor = 3


class ChebyshevPlaneSyn:

    def __init__(self,
                 sidelobe: float,
                 scan: np.ndarray,
                 omega: np.ndarray,
                 number=np.array([0, 0], dtype=int),
                 depart_flag=False):

        self.depart_flag = depart_flag
        self.sidelobe = np.abs(sidelobe)
        self.R0 = 10 ** (self.sidelobe / 20.0)

        self.scan = scan
        lobeFTemp = (2 / self.R0) * np.cosh(np.sqrt(np.arccosh(self.R0) * np.arccosh(self.R0) - np.pi * np.pi))
        self.lobeF = 1 + 0.636 * lobeFTemp * lobeFTemp
        self.gradlobeinterval = 1.0 / (1 + np.abs(np.sin(self.scan)))
        self.interval = self.gradlobeinterval

        self.omega = omega
        degree = self.omega * 180 / np.pi
        factor = 51 * self.lobeF / degree
        numberUV = np.array(2 * np.ceil(factor / self.gradlobeinterval / 2.0), dtype=int)

        self.__adjust_number__(numberUV, number)

    def __baberier__(self, x0, M, odd=True):
        """
            Babiere function to calculate the current of chebyshev in line
            :param x0: the zero point of main lobe
            :param M: the total number of elements 1-D array
            :return: the current of every element
        """

        if odd:
            current = np.zeros(M + 1, dtype=float)
            for zmc in range(M + 1):
                n = zmc + 1
                temp = 0.
                for q in np.arange(n, M + 2):
                    term = math.factorial(q + M - 2) / (
                            math.factorial(q - n) * math.factorial(q + n - 2) * math.factorial(M - q + 1))
                    coef = x0 ** (2 * q - 2) * (-1) ** (M - q + 1)
                    temp = temp + term * coef
                current[zmc] = (2 * M-1) * temp     #
            current[0] = current[0] / 2
        else:
            current = np.zeros(M, dtype=float)
            for zmc in range(M):
                n = zmc + 1
                temp = 0.
                for q in np.arange(n, M + 1):
                    term = math.factorial(q + M - 2) / (
                            math.factorial(q - n) * math.factorial(q + n - 1) * math.factorial(M - q))
                    coef = x0 ** (2 * q - 1) * (-1) ** (M - q)
                    temp = temp + term * coef
                current[zmc] = (2 * M - 1) * temp

        current = current / np.max(current)

        return current

    def __adjust_number__(self, preNum, compNum):

        self.numberUV[0] = np.maximum(preNum[0], compNum[0])
        self.numberUV[1] = np.maximum(preNum[1], compNum[1]) if self.depart_flag else self.numberUV[0]

        self.oddU = (np.mod(self.numberUV[0], 2) == 1)
        self.oddV = (np.mod(self.numberUV[1], 2) == 1)

        degree = self.omega * 180 / np.pi
        for zmc in np.array([0, 1], dtype=int):
            if preNum[zmc] > compNum[zmc]:
                self.interval[zmc] = 51 * self.lobeF / (degree[zmc] * self.numberUV[zmc])
            else:
                degree[zmc] = 51 * self.lobeF / (self.interval[zmc] * self.numberUV[zmc])

        self.omega = degree * np.pi / 180.

        self.x0 = np.cosh(np.arccosh(self.R0) / (self.numberUV - 1))
        direct = 2 * self.R0 * self.R0 / (1 + (self.R0 * self.R0 - 1) * degree / 51)
        self.direct = np.pi * direct[0] * direct[1]

    def synthesis(self):
        """
        synthesis of Chebyshev
        :param number: the size of new array
        """

        if self.depart_flag:
            Mu = int(self.numberUV[0] / 2.)
            Mv = int(self.numberUV[1] / 2.)

            self.current = np.einsum("m,n->mn",
                                     self.__baberier__(self.x0[0], Mu, odd=self.oddU),
                                     self.__baberier__(self.x0[1], Mv, odd=self.oddV))
        else:
            self.__undepart_synthesis__()

    def __undepart_synthesis__(self):
        """
        synthesis for undepart array
        :return:
        """

        n = self.numberUV[0]
        m = int(n / 2)
        w0 = np.cosh(np.arccosh(self.R0) / (n - 1))

        cheby_coef = np.zeros(n, dtype=float)
        cheby_coef[-1] = 1

        odd = self.oddU

        # odd or even
        val = 1 if odd else 0.5
        val2 = 2 if odd else 1
        val_current_coef = 4.0 if odd else 1
        pq = np.arange(1, m + val2, 1)

        cos_base = np.cos((pq - val) * np.pi / n)
        x0 = w0 * np.einsum("i,j->ij", cos_base, cos_base)
        cheby_value = np.polynomial.chebyshev.chebval(x0, cheby_coef)
        cheby_value[0, 0] = cheby_value[0, 0] / val_current_coef

        zmc = np.arange(1, m + val2, 1)
        cos_inner = np.cos(np.einsum("i,j->ij", (zmc - val), (pq - val)) * 2 * np.pi / n)
        current = (4 / n) ** 2 * np.einsum("mp,nq,pq->mn", cos_inner, cos_inner, cheby_value)

        self.current = current / np.max(current)

    def get_current(self):
        """
        :return: the first quadrant of current of array
        """
        return self.current

    def get_size(self):
        return self.numberUV

    def array_factor(self, theta_scope, phi_scope):
        """
        calculate the S
        :param theta_scope: the scan angle of theta
        :param phi_scope: the scan angle of phi
        :return: the array_factor S
        """
        if self.depart_flag:
            return self.__depart_array_factor__(theta_scope, phi_scope)
        else:
            return self.__undepart_array_factor__(theta_scope, phi_scope)

    def __depart_array_factor__(self, theta_scope, phi_scope):
        """
            calculate the S
            :param theta_scope: the scan angle of theta
            :param phi_scope: the scan angle of phi
            :return: the array_factor S
        """
        k = 2 * np.pi
        nu = int(self.numberUV[0] / 2.)
        nv = int(self.numberUV[1] / 2.)
        du = self.interval[0]
        dv = self.interval[1]

        u = 0.5 * k * du * np.cos(theta_scope)
        v = 0.5 * k * dv * np.einsum("p,t->pt", np.sin(phi_scope), np.sin(theta_scope))

        if self.oddU:
            mm = 2 * np.arange(1, nu + 2, 1) - 2
        else:
            mm = 2 * np.arange(1, nu + 1, 1) - 1

        cos_mu = np.cos(np.einsum("t,m->tm", u, mm))

        if self.oddV:
            nn = 2 * np.arange(1, nv + 2, 1) - 2
        else:
            nn = 2 * np.arange(1, nv + 1, 1) - 1

        cos_mv = np.cos(np.einsum("pt,k->ptk", v, nn))

        AF = np.einsum("ptn,tm,mn->pt", cos_mv, cos_mu, self.current)

        return 4 * AF

    def __undepart_array_factor__(self, theta_scope, phi_scope):
        """
        calculate the undepart array factor
        :param theta_scope:
        :param phi_scope:
        :return: array factor
        """

        k = 2 * np.pi
        du = self.interval[0]
        dv = self.interval[1]

        M = self.current.shape[0] - (1 if self.oddU else 0)

        zmc = 2 * np.arange(1, M + (2 if self.oddU else 1)) - 1
        local_current = np.copy(self.current)
        local_current[0, 0] = self.current[0, 0] / (4. if self.oddU else 1.)

        u = 0.5 * k * du * np.cos(theta_scope)
        v = 0.5 * k * dv * np.einsum("i,j->ij", np.sin(phi_scope), np.sin(theta_scope))

        cos_mu = np.cos(np.einsum("i,j->ij", u, zmc))
        cos_mv = np.cos(np.einsum("ij,k->ijk", v, zmc))
        AF = 4 * np.einsum("ptn,tm,mn->pt", cos_mv, cos_mu, local_current)

        return AF

    def show(self):
        Num = self.get_size()
        print("Department: " + str(self.depart_flag))
        print("ArraySize: " + str(Num[0]) + "*" + str(Num[1]))
        print("Sidelobe: " + str(self.sidelobe))
        degreeScan = self.scan * 180 / np.pi
        degreeOmega = self.omega * 180 / np.pi
        print("Scan Angle: " + "Theta: " + str(degreeScan[0]) + "\tPhi: " + str(degreeScan[1]))
        print("BWhalf : " + "Theta: " + str(degreeOmega[0]) + "\tPhi: " + str(degreeOmega[1]))
        print("Interval : " + "U: " + str(self.interval[0]) + "\tV: " + str(self.interval[1]))
        print("Direct: " + str(10 * np.log10(self.direct)) + "dB")

    sidelobe = 0
    scan = np.array([90, 90], dtype=float)
    omega = np.array([1, 1], dtype=float)
    interval = np.array([0.5, 0.5], dtype=float)
    x0 = np.array([1, 1], dtype=float)
    direct = 1
    current = None
    gradlobeinterval = np.array([0.5, 0.5], dtype=float)
    lobeF = 1
    R0 = 0
    numberUV = np.array([1, 1], dtype=int)
    oddU = True
    oddV = True
    depart_flag = False


if __name__ == '__main__':
    sidelobe = 30
    scan = np.array([np.pi / 3, np.pi / 3])
    omega = np.array([5, 5]) / 180 * np.pi
    number = np.array([70, 70], dtype=int)

    # test = ChebyshevPlaneSyn(sidelobe, scan, omega)
    # info = test.ParseIndex(number)
    # info.show()
