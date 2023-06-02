import numpy as np
import matplotlib.pyplot as pl
import math

N = 100
L = 1
TINY = 1e-10
MAXERR = 1e-7

def Jacobi(Phi, f) : 
  dx = L/(Phi.size-1)
  Phi[1:-1] = 0.5*(Phi[:-2] + Phi[2:] - dx*dx*f[1:-1])
  return Phi

def redBlackGaussSeidel(Phi, f) : 
  dx = L/(Phi.size-1)
  Phi[2:-1:2] = 0.5*(Phi[1:-2:2] + Phi[3::2] - dx*dx*f[2:-1:2])
  Phi[1:-1:2] = 0.5*(Phi[:-2:2] + Phi[2::2] - dx*dx*f[1:-1:2])
  return Phi

def error(Phi1, Phi2):
  return np.abs((Phi2-Phi1)[1:-1]/(0.5*np.abs(Phi1+Phi2)+TINY)[1:-1]).sum()

def init() : 
  Phi = np.zeros(N+1)
  dx = L/(Phi.size-1)
  x = np.arange(Phi.size)*dx
  f = np.sin(x)
  return x, Phi, f

if __name__ == "__main__" : 
  x, Phi, f = init()

  err = 1
  iterations = 0
  while err > MAXERR :
    iterations += 1
    Phi2 = Phi.copy()
    #Phi2 = Jacobi(Phi2, f)
    Phi2 = redBlackGaussSeidel(Phi2, f)

    #print(Phi2)
    err = error(Phi,Phi2)
    if( iterations % 1000 == 0) :
      print("Iteration: {0} {1:.3e}".format(iterations, err))
    Phi = Phi2

  print("Total iteration: {0} {1:.3e}".format(iterations, err))
  pl.scatter(x, Phi,s=1,label="numerical")
  pl.plot(x,-np.sin(x)+x*np.sin(1), label="analytic")
  pl.legend(loc="best")
  pl.savefig("relax.pdf")
