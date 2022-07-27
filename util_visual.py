import numpy as np
import matplotlib.pyplot as plt

def plot_surf(tgrid,xgrid,ugrid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(tgrid, xgrid, ugrid, cmap='viridis')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$u_\\theta(t,x)$')
    ax.view_init(35,35)


def plot_loss_history(hist):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(range(len(hist)), hist,'k-')
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    return fig

