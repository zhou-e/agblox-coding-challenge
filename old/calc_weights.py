'''
Calculates the line of best fit for a set of points in a csv using
    sorted points and weights.
Author: Edward Zhou
'''

import csv
import time
import math

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

    xs, ys, slope, intercept = get_slope(file)

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

def get_slope(file):
    '''
    Reads the input file of points, and outputs an estimation for m and x in
        the linear model mx+b with it's losses.
    Args:
        filename (str) : a string that represents the csv file with the points.
    Returns:
        xs (arr) : an array of x values stored as floats.
        ys (arr) : an array of y values stored as floats.
        slope (float) : the updated slope of the model (a in y = ax+b).
        intercept (float) : the updated intercept of the model (b in y = ax+b).
    '''
    points = []
    xs = []
    ys = []
    xTot = yTot = 0
    for line in enumerate(file):
        if line[1][0][0] in "-0123456789":
            x = float(line[1][0])
            y = float(line[1][1])
            xs.append(x)
            ys.append(y)
            points.append([x, y])

    points, xMedian, yMedian = sorter(points)

    seen = slope = intercept = 0
    print("Printing the loss of every 10th point")
    for point in enumerate(points):
        #uses the first point to set xMin and xMax
        if point[0] == 0:
            xMin = xMax = point[1][0]

        #checks if the first character in a line is a number (i.e. a point)
        seen += 1

        if point[1][0] < xMin:
            xMin = point[1][0]
        if point[1][0] > xMax:
            xMax = point[1][0]

        oldSlope, oldInt = slope, intercept
        slope, intercept = calculator(seen, point[1][0], point[1][1], 
                                      xMin, xMax, xMedian, yMedian, \
                                      slope, intercept)

        #loss function
        if point[0]%10 == 0:
            print("Loss (Residual) of observed - " +\
                  "predicted: %f"%(point[1][1]-(slope*x+intercept)))
            print("Slope loss: %f"%(oldSlope - slope))
            print("Intercept loss: %f"%(oldInt - intercept))

    return xs, ys, slope, intercept

def sorter(points):
    '''
    Sorts the points by increasing distance from the median using euclidean distance.
    Args:
        points (arr) : an array of arrays of x, y pairs.
    Returns:
        points (arr) : the sorted array.
        xMedian (float) : the median of the x points.
        yMedian (float) : the median of the y points.
    '''
    temp = sorted(points, key=lambda x:x[0], reverse=False)
    xMedian = temp[len(points)//2][0]

    temp = sorted(points, key=lambda x:x[1], reverse=False)
    yMedian = temp[len(points)//2][1]

    for point in points:
        point.append(math.sqrt((point[0]-xMedian)**2+(point[1]-yMedian)**2))

    points = sorted(points, key=lambda x:x[2], reverse=False)
    return points, xMedian, yMedian

def calculator(seen, x, y, xMin, xMax, xMed, yMed, slope, intercept):
    '''
    Updates the original model's slope and intercept based on the new point given.
    Args:
        seen (int) : number of points already used in calculation.
        x (float) : the x value of the new point given.
        y (float) : the y value of the new point given.
        slope (float) : the slope of the current model (a in y = ax+b).
        intercept (float) : the y-intercept of the current model (b in y = ax+b).
    Returns:
        newSlope (float) : the updated slope of the model (a in y = ax+b).
        newIntercept (float) : the updated intercept of the model (b in y = ax+b).
    '''
    #check if the min and max aren't the same (else zero division error)
    if xMax-xMin != 0:
        intercept = ((seen-1)*intercept + ((y-yMed) * \
        abs((x-xMed)/(xMax-xMin)*2))) / seen # intercept of new point compared to the mean of ys, weighted on the seen points
        slope = ((seen-1)*slope + ((y-intercept)/x * \
        abs((x-xMed)/(xMax-xMin)*2))) / seen # slope of new point compared to the intercept, weighted on the seen points

    else:
        intercept = ((seen-1)*intercept+y) / seen
        slope = ((seen-1)*slope+(y-intercept)/x) / seen

    return slope, intercept

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
