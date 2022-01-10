'''
Calculates the line of best fit for a set of points in a csv using
    machine learning (1 layer).
Author: Edward Zhou
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import csv
import numpy as np
import tensorflow as tf
import time

import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

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
        except FileNotFoundError:
            print("Could not find file in directory.")
            print("Please enter a valid filename.")

    xs, ys, slope, intercept, errors, sd = get_points(file)

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
    Reads the input file of points, and outputs an estimation using
        machine learning.
    Args:
        file (file) : a file that contains the lines with points.
    Returns:
        xs (arr) : an array of x values stored as floats.
        ys (arr) : an array of y values stored as floats.
        slope (float) : the updated slope of the model (a in y = ax+b).
        intercept (float) : the updated intercept of the model (b in y = ax+b).
    '''
    xs = []
    ys = []
    for line in enumerate(file):
        if line[1][0][0] in "-0123456789":
            xs.append(float(line[1][0]))
            ys.append(float(line[1][1]))

    xs = np.array(xs,  dtype = float)
    ys = np.array(ys,  dtype = float)

    slope, intercept = make_train_model(xs, ys)

    #prints the SSError of all the points for the final model
    print("\nTotal Squared Loss:")
    errors, sd = find_errors(xs, ys, slope, intercept)

    return xs, ys, slope, intercept, errors, sd

def make_train_model(xs, ys):
    '''
    Trains the model using the given x and y values.
    Args:
        xs (arr) : an array of x values stored as floats.
        ys (arr) : an array of y values stored as floats.
    Returns:
        slope (float) : the slope of the line of best fit.
        intercept (float) : the intercept of the line of best fit.
    '''
    model = tf.keras.Sequential(tf.keras.layers.Dense(units=1, input_shape=[1]))

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

    #loss function
    model.fit(xs, ys, epochs=10, verbose=2, steps_per_epoch = int(xs.size/2))

    slope, intercept = model.get_weights()

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
    for i in range(xs.size):
        residual = (ys[i] - (slope*xs[i]+intercept))
        loss += residual**2
        errors.append(residual)
    loss = loss[0][0]
    print(loss)
    return errors, (loss/xs.size)**0.5

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
