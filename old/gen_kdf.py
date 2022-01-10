'''
Generates a set of points with an input of the line of best fit and sample
    points using kernel density estimation.
Author: Edward Zhou
'''

from sklearn.neighbors import KernelDensity
import numpy as np
import random

def kde2D(xs, ys, bandwidth, xbins=100j, ybins=100j):
    '''
    Calculates the kernel density of 2D points.
    Args:
        xs (arr) : an array of x values of the sample points.
        ys (arr) : an array of y values of the sample points.
        bandwidth (float) : the bandwidth for KernelDensity.
        xbins (complex) : the dimension to split the x axis.
        ybins (complex) : the dimension to split the y axis.
    Returns:
        zs (np.array) : array of arrays of probabilities for each x, y bin.
    '''
    gridX, gridY = np.mgrid[xs.min():xs.max():xbins, 
                      ys.min():ys.max():ybins]

    grid = np.vstack([gridX.ravel(), gridY.ravel()]).T
    points  = np.vstack([ys, xs]).T

    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(points)

    # returns the likelihood of the samples
    zs = np.exp(kde.score_samples(grid))
    zs = np.reshape(zs, gridX.shape)
    return zs

def generator(xs, ys, size = 100, **kwargs):
    '''
    Generates points of size "size".
    Args:
        xs (arr) : an array of x values of the sample points.
        ys (arr) : an array of y values of the sample points.
        size (int) : the number of points to be generated.
    Returns:
        xs (arr) : an array of generated x values.
        ys (arr) : an array of generated y values.
    '''
    xs = np.array(xs, dtype = float)
    ys = np.array(ys, dtype = float)
    xMin = xs.min()
    xMax = xs.max()
    yMin = ys.min()
    yMax = ys.max()

    zs = kde2D(xs, ys, 2.0)
    tot = 0
    prob = []
    xs = []
    ys = []
    for z in zs:
        tot += z.sum()
        prob.append(tot)
    for i in range(size):
        index = random.uniform(0, tot)
        idx1 = idx2 = 0

        #finds the proper x value of the point based on the cumulative probability function of xs.
        while index > prob[idx1]:
            idx1 += 1
        if idx1 != 0:
            index -= prob[idx1-1]

        #finds the proper y value of the point based on the cumulative probability function of ys given an x.
        runningSum = zs[idx1][idx2]
        while index > runningSum:
            idx2 += 1
            runningSum += zs[idx1][idx2]

        xs.append(idx1*(xMax-xMin)/100+xMin)
        ys.append(idx2*(yMax-yMin)/100+yMin)
    return xs, ys