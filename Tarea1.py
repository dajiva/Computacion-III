import random as rn
from sympy import *
import numpy as np
import math
import matplotlib.pyplot as plt
import time


x = Symbol('x')
# Funciones
f1 = 2* sin(x) -x**2/10
f2 = x**2
f3 = 3*x**3 -5*x**2 + x -6
f4 = -1.5*x**6 - 2*x**4 + 12*x
f5 = 2*x +3/x
functions = [f1, f2, f3, f4, f5]

## Métodos

# Búsqueda ingenua
def naiveSearch(f):
    R = np.arange(-10000,10000,0.1)
    maxVal = -np.inf
    minVal = np.inf
    for i in R:
        fx = f.subs(x, i)
        if fx < minVal:
            minVal = fx
            xMin = i
        if fx >  maxVal:
            maxVal = fx
            xMax = i
            
    maximum = (xMax, maxVal)
    minimum = (xMin, minVal)

    return maximum, minimum

# Método de Newton
def doNewton(tol, maxIter, func):
    points = []
    for i in range(10):
        x0 = rn.randint(-10000,10000)
        points.append(NewtonMethdod(tol, maxIter, x0, func))
                      
    xMax = max(points)
    yMax = func.subs(xMax)
    xMin = min(points)
    yMin = func.subs(xMin)
                      
    maximum = (xMax, yMax)
    minimum = (xMin, yMin)
                      
    return maximum, minimum
    
def NewtonMethod(tol, maxIter, func):
        der = diff(func, x)
        der2 = diff(der,x)
        if der2 == 0:
            print("No es posible optimizar esta función con el método actual")
            return np.nan
        
        error = np.inf
        numIter = 0
        
        while error > tol and numIter < maxIter:
            x1 = float(x0 - (der.subs(x, x0)/ der2.subs(x,x0)))
            numIter += 1
            
            if x1 != 0:
                error = abs((x1-x0)/x1)*100
            else:
                error = abs(x1-x0)
                
        if error > tol:
            maxVal = x1
        else:
            maxVal = x0
        
        return maxVal


# Gráfica de la función
X = np.arange(-10000,10000,10)
def plotFunction(f):
    Y = []
    for i in X:
        Y.append(f.subs(x,i))
    plt.plot(X,Y)
    
# Comparativa métodos
timesNewton = []
timesNaive = []
for f in functions:
    print("Función: ")
    plotFunction(f)
    print("Naive Search:\n")
    t = time.time()
    maxNaive, minNaive = naiveSearch(f)
    timesNaive.append(time.time()-t)
    print("Max point: ", maxNaive)
    print("Min point: ", minNaive)
    print("Newton Method:\n")
    t = time.time()
    maxNewton, minNewton = NewtonMethod(0.0001, 1000000, f)
    timesNewton.append(time.time()-t)
    print("Max point: ", maxNewton)
    print("Min point: ", minNewton)
avgNewton = mean(timesNewton)
avgNaive = mean(timesNaive)



# Gráfica métodos
plt.bar(["Newton Method", "Naive Search"], [avgNewton, avgNaive] )