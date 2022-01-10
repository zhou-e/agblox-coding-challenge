'''
Calculates the line of best fit for a set of points in a csv using
    the least squares method.
Author: Edward Zhou
'''

import csv
import time
import random

import gen_partition as gen
#import gen_distribution as gen
#import gen_kdf as gen

def get_file():
    '''
    Gets the input file name and feeds the file to other functions.
    '''
    validFile = False
    while not validFile:
        filename = input("Enter the name of the file with points: ")
        try:
            if not ".csv" in filename:
                filename += ".csv"
            file = open(filename)
            file = csv.reader(file, delimiter = ",")
            validFile = True
        except:
            print("Could not find file in directory.")
            print("Please enter a valid filename.")

    xs, ys, slope, intercept = get_points(file)

    #prints the SSError of all the points for the final model
    print("\nTotal Squared Loss:")
    errors, sd = find_errors(xs, ys, slope, intercept)

    #uncomment to use resampling to generate points
    genx, geny = gen.generator(xs, slope, intercept, errors, len(xs))

    #uncomment to use a distribution to generate points
    #genx, geny  = gen.generator(xs, slope, intercept, errors, sd, len(xs))

    #uncomment to use kernel desnity to generate points
    #genx, geny = gen.generator(xs, ys, len(xs))

    print("\nSlope: %.5f\nIntercept: %.5f"%(slope, intercept))
    write_gen(filename, genx, geny)
    time.sleep(5)

def get_points(file):
    '''
    Reads the input file of points, and outputs an estimation using the
        least squares model for m and x in the linear model mx+b with
        it's losses.
    Args:
        file (file) : a file that contains the lines with points.
    Returns:
        slope (float) : the updated slope of the model (a in y = ax+b).
        intercept (float) : the updated intercept of the model (b in y = ax+b).
    '''
    seen = slope = intercept = 0

    xs = []
    ys = []
    xTot = yTot = 0
    for line in file:
        #checks if the first character in a line is a number (i.e. a point)
        if line[0][0] in "-0123456789":
            x = float(line[0])
            y = float(line[1])
            xs.append(x)
            ys.append(y)
            xTot += x
            yTot += y

    return calculator(xs, ys, xTot/len(xs), yTot/len(ys))
    

def calculator(xs, ys, xMean, yMean):
    '''
    Calculates the slope and intercept updated with each given point.
    Args:
        xs (arr) : an array of x values stored as floats.
        ys (arr) : an array of y values stored as floats.
        xMean (float) : the mean of all x values.
        yMean (float) : the mean of all y valeus.
    Returns:
        xs (arr) : an array of x values stored as floats.
        ys (arr) : an array of y values stored as floats.
        slope (float) : the slope of the model (a in y = ax+b).
        intercept (float) : the intercept of the model (b in y = ax+b).
    '''
    xyDiff = xSqDiff = slope = intercept = 0

    print("Printing the loss of every 10th point")
    for i in range(len(xs)):
        oldSlope, oldInt = slope, intercept
        xyDiff += (xs[i] - xMean)*(ys[i] - yMean)
        xSqDiff += (xs[i] - xMean)**2
        slope = xyDiff/xSqDiff
        intercept = yMean - xMean*slope

        #loss function
        if i%10 == 0:
            print("Loss (Residual) of observed - " +\
                  "predicted: %f"%(ys[i]-(slope*xs[i]+intercept)))
            print("Slope loss: %f"%(oldSlope - slope))
            print("Intercept loss: %f"%(oldInt - intercept))

    return xs, ys, slope, intercept

def find_errors(xs, ys, slope, intercept):
    '''
    Prints the sum of errors squared for all x,y pairs given a slope and intercept.
    Args:
        xs (arr) : an array of x values stored as floats.
        ys (arr) : an array of y values stored as floats.
        slope (float) : the slope of the current model (a in y = ax+b).
        intercept (float) : the y-intercept of the current model (b in y = ax+b).
    Returns:
        errors (arr) : an array the residuals from the line of best fit.
        sd (float) : the average standard deviation from the line of best fit.
    '''
    loss = tot = 0
    errors = []
    for i in range(len(xs)):
        residual = (ys[i] - (slope*xs[i]+intercept))
        loss += residual**2
        errors.append(residual)
    print(loss)
    return errors, (loss/len(xs))**0.5

def write_gen(filename, xs, ys):
    '''
    Writes out the generated points in a csv file.
    Args:
        filename (str) : the name of the input file.
        xs (arr) : an array of generated x values.
        ys (arr) : an array of generated y values.
    '''
    outFileName = 'gen_'+filename
    with open(outFileName, 'w', newline = '') as pointsOut:
        slopeWrite = csv.writer(pointsOut)
        slopeWrite.writerow(['x', 'y'])
        for i in range(len(xs)):
            slopeWrite.writerow(['%f'%xs[i], '%f'%ys[i]])
    print("Generated points located in %s"%outFileName)

if __name__ == '__main__':
    get_file()
