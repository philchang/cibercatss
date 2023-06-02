import numpy as np
import matplotlib.pyplot as pl
import math
import rk2
import newton_raphson as nr

N = 100
L = 1

def derivatives(x, y) :
  Phi = y[0]
  dPhidx = y[1]
  dydx = np.zeros(y.size)
  dydx[0] = dPhidx
  dydx[1] = math.sin(x)
  return dydx  

def g(alpha, output=False) : 
  x = 0
  dx = L/N
  y = np.zeros(2)
  y[1] = alpha
  yout = []
  xout = np.arange(0,1,dx)
  for x in xout:
    y = rk2.rk2(x, x+dx, dx, y, derivatives)
    if( output) : 
      yout.append(y[0])
  if( output) : 
    return xout, yout
  return y[0]


if __name__ == "__main__" : 
  alpha, error, iterations = nr.newton_raphson(g, 1.)
  x, y = g(alpha, output=True)
  pl.plot(x,y, lw=2)
  pl.show()