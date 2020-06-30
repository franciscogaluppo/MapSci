import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def gaussian_kernel(ax, x, y, xlab, ylab, title, cmap='coolwarm'):
    """
    """
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10
    
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.plot([0, xmax], [0, xmax], "--", color="darkgrey")
    
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    ax.imshow(np.rot90(f), cmap=cmap, extent=[xmin, xmax, ymin, ymax], aspect='auto')
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    plt.title(title)
