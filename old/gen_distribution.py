'''
Generates a set of points with an input of the line of best fit and sample
    points using a gaussian distribution.
Author: Edward Zhou
'''

import random

def generator(xs, slope, intercept, errors, sd, size, **kwargs):
    '''
    Generates points of size "size".
    Args:
        xs (arr) : an array of x values of the sample points.
        slope (float) : the slope of the line of best fit.
        intercept (float) : the intercept of the line of best fit.
        errors (arr) : the residuals of the sample points from
            the line of best fit.
        size (int) : the number of points to be generated.
    Returns:
        xs (arr) : an array of generated x values.
        ys (arr) : an array of generated y values.
    '''
    #gets the errors of the points.
    errorByX = []
    length = len(xs)
    minError = maxError = errors[0]
    for i in range(length):
        if errors[i] < minError:
            minError = errors[i]
        if errors[i] > maxError:
            maxError = errors[i]
        errorByX.append((xs[i], errors[i]))

    #sorts the points and errors by their x value.
    errorByX = sorted(errorByX, key=lambda x:x[0], reverse=False)

    #point generation.
    xs = []
    ys = []
    xRange = (errorByX[0][0], errorByX[-1][0])
    errorSum = 0
    for i in range(size):
        x = random.uniform(xRange[0], xRange[1])
        error = random.gauss(0, sd)*2*x/(xRange[1]-xRange[0])
        errorSum += error
        y = x*slope + intercept + error
        xs.append(x)
        ys.append(y)

    #distributes the errors randomly across the points, with the further points being more likely.
    errorPiece = errorSum/size
    for i in range(size):
        randX = random.expovariate(1)
        if randX > 5:
            randX = 5
        index = int(size/2-size/10*(5-randX)*random.choice([-1, 1]))

        ys[index] -= errorPiece
    
    return xs, ys
