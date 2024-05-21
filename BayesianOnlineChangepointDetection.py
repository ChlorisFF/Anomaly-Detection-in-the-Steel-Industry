import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

def changepoint(d):
    n = len(d)
    dbar = np.mean(d)
    dsbar = np.mean(np.multiply(d, d))
    fac = dsbar - np.square(dbar)
    summup = []
    summ = 0

    for z in range(n):
        summ+=d[z]
        summup.append(summ)

    y = []
    for m in range(n - 1):
        pos = m + 1
        mscale = 4 * (pos) * (n - pos)
        Q = summup[m] - (summ - summup[m])
        U = -np.square(dbar * (n - 2 * pos) + Q) / float(mscale) + fac
        y.append(-(n / float(2) - 1) * math.log(n*U/2) - 0.5*math.log((pos * (n - pos))))

    z, zz = np.max(y), np.argmax(y)
    mean1 = sum(d[:zz+1])/float(len(d[:zz+1]))
    mean2 = sum(d[(zz+1):n])/float(n-1-zz)

    return y, zz, mean1, mean2

def rawSensorData(str):

    # Read data from CSV file
    measurements = []
    time = []
    i=3

    with open(str, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            measurements.append(float(row[1]))
            time.append(i)
            i=i+1

    # Plot sensor data
    measurements_series = pd.Series(measurements, index=time)
    measurements_series.plot(title='Sensor data')

    # Anomaly detection with online Bayesian changepoint detection
    step_like = changepoint(measurements)
    step_series = pd.Series(step_like[0], index=time[1:])
    plt.figure()
    step_series.plot(title='Log likelihood of changepoint in raw sensor data')

    plt.show()

def withKurtosis(str):

    # Read data from CSV file
    measurements = []
    time = []
    i=3

    with open(str, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            measurements.append(float(row[1]))
            time.append(i)
            i=i+1

    plt.plot(measurements)
    plt.title("Sensor Data")
    plt.show()

    window = 4
    s = pd.Series(measurements)
    kur = s.rolling(window).kurt()

    k = []
    for x in range(len(kur)):
        if x > window - 2:
            k.append(kur[x])
        if x < window - 1:
            kur[x] = 0
            k.append(kur[x])

    k_series = pd.Series(k, index=time)
    k_series.plot(title='Kurtosis')

    step_like = changepoint(k)
    step_series = pd.Series(step_like[0], index=time[1:])
    plt.figure()
    step_series.plot(title='Log likelihood of changepoint in Kurtosis')

    plt.show()

    a, time, b, c = changepoint(k)
    anomaly_time = time
    print('Alert at time:', anomaly_time)

def analyze(str, window_size):

    # Read data from CSV file
    measurements = []
    time = []
    i=3

    with open(str, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            measurements.append(float(row[1]))
            time.append(i)
            i=i+1

    # Plot sensor data
    plt.subplot(2, 2, 1)
    measurements_series = pd.Series(measurements, index=time)
    measurements_series.plot(title='Sensor data')

    # Anomaly detection with online Bayesian changepoint detection
    plt.subplot(2, 2, 2)
    step_like = changepoint(measurements)
    step_series = pd.Series(step_like[0], index=time[1:])
    step_series.plot(title='Log likelihood of changepoint in raw sensor data')

    window = window_size
    s = pd.Series(measurements)
    kur = s.rolling(window).kurt()

    k = []
    for x in range(len(kur)):
        if x > window - 2:
            k.append(kur[x])
        if x < window - 1:
            kur[x] = 0
            k.append(kur[x])

    plt.subplot(2, 2, 3)
    k_series = pd.Series(k, index=time)
    k_series.plot(title='Kurtosis')

    plt.subplot(2, 2, 4)
    step_like = changepoint(k)
    step_series = pd.Series(step_like[0], index=time[1:])
    step_series.plot(title='Log likelihood of changepoint in Kurtosis')

    plt.tight_layout()
    plt.show()

    a, anomaly_time, b, c = changepoint(k)
    a, point_time, b, c = changepoint(measurements)
    print('Changepoint time:', point_time)
    print('Alert at time:', anomaly_time)
