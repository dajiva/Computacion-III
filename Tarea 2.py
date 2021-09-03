#!/usr/bin/env python
# coding: utf-8

# # Optimización multidimensional sin restricciones

# ## Introducción

# El proceso de optimización es altamente utilizado en diversas aplicaciones, pues nos permite encontrar el máximo o mínimo de una función. Sin embargo, el proceso se complica cuando nos encontramos con funciones de más de una variable. Al requerir una gran cantidad de cálculos, los métodos numéricos son los más convenientes para este tipo de funciones. Uno de ellos se conoce como método del gradiente, el cual determina un punto óptimo calculando derivadas direccionales, es decir, el gradiente de la función respecto a un vector inicial. Este gradiente permite determinar un camino por el cual la función se guía para acercarse cada vez más a un punto máximo o mínimo, a través de ciertas iteraciones. A continuación se presenta una implementación de dicho método.

# ## Desarrollo

# Primero se cargaron las librerías necesarias para armar los arreglos, realizar los cálculos con variables simbólicas y generar las gráficas necesarias, así como la librería de time para calcular los tiempos de ejecución.

# In[146]:


import random as rn
from sympy import *
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Circle
import time


# Posteriormente se definieron 5 variables simbólicas y 4 funciones arbitrarias, de 1, 2, 3 y 4 variables cada una.

# In[147]:


x, y, z, w, h  = symbols('x, y, z, w, h')

# Funciones
f1 = 3*x**3 + 4*x**2 -x + 5
#f1 = 2* sin(x) -x**2/10
f1 = 3*sin(x)
f2 = 2*x*y + 2*x - x**2 - 2*y**2
f3 = x**2 + z**2 + x + 2*y + 3*z + 10
f4 = x**2+y**2+2*z-2*w+x+2*y+3*z-6*w-8
functions = [f1, f2, f3, f4]


# Se utilizó el método de Newton implementado en la tarea anterior como subrutina del método general.

# In[148]:


# Método de Newton
def NewtonMethod(tol, maxIter, func):
        x0 = rn.randint(-100,100)
        der = diff(func, h)
        der2 = diff(der,h)
        if der2 == 0:
            print("No es posible optimizar esta función con el método actual")
            return np.nan
        
        error = np.inf
        numIter = 0
        
        while error > tol and numIter < maxIter:
            x1 = float(x0 - (der.subs(h, x0)/ der2.subs(h,x0)))
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


# Posteriormente se implementó el método del gradiente, con varias subfunciones

# In[149]:


# Calcular gradiente de la función
def gradient(func, variables, vals):
    grad = list()
    for var in variables:
        der = diff(func, var)
        for i in range(len(vals)):
            der = der.subs(variables[i], vals[i])
        grad.append(der)
    return grad

# Reducir la función de varias variables a una variable usando el gradiente
def getOneDimFunc(func, variables, vals):
    grad = gradient(func, variables, vals)
    for i in range(len(variables)):
        func = func.subs(variables[i], vals[i] + grad[i]*h)
    oneDimFunc = simplify(func)
    return oneDimFunc

# Definir nuevos valores iniciales para la siguiente iteración
def newCoords(func, variables, vals, hVal):
    grad = gradient(func, variables, vals)
    for i in range(len(variables)):
        vals[i] = vals[i] + grad[i] * hVal
        
    return vals

# Función principal, método del gradiente
def multiDimOpt(func, variables, initPoints, numIters):
    for i in range(numIters):
        oneDimFunc = getOneDimFunc(func, variables, initPoints)
        hVal = NewtonMethod(0.001, 100000, oneDimFunc)
        newCoords(func, variables, initPoints, hVal)
        points = initPoints
    return points 


# Asimismo, se definieron dos funciones para graficar las funciones de una y dos variables, junto con sus respectivos puntos óptimos.

# In[150]:


# Graficar funciones
def plotFunction(f, point):
    X = np.arange(-150,150,1)
    Y = []
    for i in X:
        Y.append(f.subs(x,i))
    plt.plot(X,Y)
    # Punto óptimo
    plt.plot(point, f.subs(x,point[0]) , "o")
    
def plot3DFunction(f,points):
    X = np.arange(-10,10,0.01)
    Y = np.arange(-10,10,0.01)
    X, Y = np.meshgrid(X,Y)
    Z = 2*X*Y + 2*X - X**2 - 2*Y**2
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis_r', linewidth=0)
    # Punto óptimo
    p = Circle((points[0], points[1]), 1,  ec='k', fc="black")
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=4, zdir="z")


# ## Resultados

# Ya con todas las funciones necesarias, se pudo ejecutar el método para las funciones que se definieron. Se calculó el tiempo de cada una y se graficaron las dos primeras. Los puntos obtenidos y las gráficas se pueden ver en la impresión de la consola. 

# In[151]:


# Ejecución
times = [] 
vars = [x,y,z,w] #Variables
initPoints = [0]*5 #Puntos iniciales
for i in range(len(functions)):
    f = functions[i] #Elegir función 
    print("Función ", i+1, "variables:\n ")
    display(f) #Mostrar función 
    variables = vars[0:i+1] #Seleccionar variables
    t = time.time() # Set clock
    points = multiDimOpt(f, variables ,initPoints[0:i+1],10) # Ejecución del método
    times.append(time.time()-t) #Cálculo del tiempo
    #Gráfica funciones
    if i == 0:
        plotFunction(f, points)
    if i == 1:
        plot3DFunction(f, points)
    print("Puntos óptimos encontrados: ", points) #Imprimir puntos 


# A continuación se puede ver una gráfica que muestra la diferencia de tiempos según la cantidad de variables.

# In[158]:


# Graficar tiempos de ejecución
Variables = [1,2,3,4]
plt.plot(Variables, Tiempo)


# ## Conclusiones

# Como se puede observar, realmente no existe una relación clara entre el tiempo de ejecución y la cantidad de variables de la función. De hecho, la función más tardada fue la de solamente una variable. Se podría inferir entonces que la complejidad de la función influye más en este método que la cantidad de variables que tenga. Esta implementación del método del gradiente resultó efectiva para llegar a un punto máximo o mínimo de las funciones. Sin embargo, los tiempos de ejecución fueron mayores a un minuto, además de que no hubo certeza si el punto obtenido se trataba de un máximo o de un mínimo. 
