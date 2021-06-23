# -*- coding:utf-8 -*-
# Author : ZhengMX
# Data : 2021/4/19 15:55
# Project : ArrarySynthesis
# FileName : draw
# Cooperation : 265

import numpy as np
from matplotlib import pyplot as plt


def surface_3D(x_scope, y_scope, z_value, title="3D_surface"):
    """
    Draw 3D pattern with surface
    :param title: title of the 3D picture
    :param X:
    :param Y:
    :param Z:
    :return: handle of fig
    """

    fig = plt.figure(num=1, figsize=(8, 6), dpi=300)  # 参数为图片大小
    ax = plt.axes(projection='3d')  # get current axes，且坐标轴是3d的
    # ax.set_aspect('equal')  # 坐标轴间比例一致
    X, Y = np.meshgrid(x_scope, y_scope)

    surf = ax.plot_surface(X, Y, Z=z_value, cmap='coolwarm')
    ax.contour(X, Y, z_value, zdir='z', offset=20, cmap="coolwarm")  # 生成z方向投影，投到x-y平面

    ax.set_title(title)
    ax.set_xlabel('Theta(degree)')
    ax.set_ylabel('Phi(degree)')
    ax.set_zlabel('dB')

    # ax.set_zlim([-60, 0])

    fcb = fig.colorbar(surf, shrink=0.8, pad=0.1)

    plt.show()

    return fig


def IFT_MaskArray(radius, number, interval, ax):
    """
    Draw the circle aperture
    :param radius:
    :param number:
    :param interval:
    :param ax: handle of axe
    :return ax
    """

    phi = np.linspace(0, 2 * np.pi, 360 * 4)

    # array mask

    half = int(number / 2)
    mask_line = np.arange(-half, half + 1, 1) * interval
    temp_mask = np.einsum("i,j->ij", np.ones([2 * half + 1], dtype=float), mask_line * mask_line)
    dis_mask = temp_mask + temp_mask.T - radius * radius

    array_mask = np.array(np.where(dis_mask > 0, 0, 1), dtype=np.int8)

    idx, idy = np.where(array_mask == 1)

    # draw
    circle = ax.fill(radius * np.cos(phi), radius * np.sin(phi), color='lightblue', zorder=0)
    elements = ax.scatter(x=mask_line[idx], y=mask_line[idy], s=10, c='gray', zorder=1)

    # Arrow
    arrow_width = 0.03
    arrow_head_width = 0.3
    arrow_head_length = 0.5
    fontsize = 15

    ax.arrow(x=0, y=0, dx=radius - 0.5, dy=0,
             width=arrow_width, head_width=arrow_head_width, head_length=arrow_head_length, color='k')
    ax.text(x=radius / 2, y=0.3, s="r", size=fontsize, family='times new roman',
            style='italic')

    ax.set_aspect('equal', 'box')
    ax.set_axis_off()
    ax.grid(True)

    ax.set_title("Mask of Array with " + str(len(idx)) + " elements", fontsize=fontsize, family='times new roman')

    return ax


def IFT_3D_surface(theta, phi, AF, ax, title=''):
    r = np.linspace(-1, 1, len(theta))

    R, RHO = np.meshgrid(r, phi)

    X, Y = R * np.cos(RHO), R * np.sin(RHO)

    val = 20 * np.log10(AF / AF.max())

    surf = ax.plot_surface(X, Y, Z=val, cmap='coolwarm')
    ax.contour(X, Y, val, zdir='z', levels=8, offset=20, cmap="coolwarm")  # 生成z方向投影，投到x-y平面

    if title == '':
        title = "normalized 3D pattern"

    ax.set_title(title, family="times new roman", fontsize=15)
    # ax.set_xlabel('Theta(degree)')
    # ax.set_ylabel('Phi(degree)')
    ax.set_zlabel('/dB', family="times new roman")
    ax.set_zlim([-60, 0])

    return surf


def IFT_line_plot(x, y, ax, title=''):
    y = 20 * np.log10(y / y.max())

    theta = x * 180 / np.pi

    ax.plot(theta, y)

    ax.set_title(title, family="times new roman", style='italic')

    ax.set_xticks([-90, -60, -30, -5, 5, 30, 60, 90])
    ax.set_xlabel(r'$\theta$ (degree)', family="times new roman", style='italic', fontsize=15)
    ax.set_xlim([-90, 90])

    ax.set_ylabel('normalized pattern/dB', family="times new roman", fontsize=15)
    ax.set_ylim([-80, 0])
    ax.set_yticks([-80, -70, -60, -50, -40, -30, -20, -10, -3, 0])

    ax.grid(True)

    return ax


if __name__ == '__main__':
    radius = 8
    interval = 0.5
    number = 33

    fig = plt.figure(num=1, figsize=(8, 8), dpi=300)

    IFT_MaskArray(radius=radius, number=number, interval=interval, ax=fig.add_subplot(221))
    # surf = IFT_3D_surface(theta=0, phi=0, AF=0, ax=fig.add_subplot(224, subplot_kw={'projection': '3d'}), title='')
    # fig.colorbar(surf, shrink=0.8, pad=0.1)

    plt.show()
