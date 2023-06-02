import numpy as np
import matplotlib.pyplot as pl
import math

N = 1000
xmax = 10
TINY = 1e-10
MAXERR = 1e-3

def applyBC( Phi) : 
  Phi[0] = Phi[-2]
  Phi[-1] = Phi[1]

def evolve(Phi, t1, t2, dt, dx) : 
  t = t1
  while(t < t2) : 
    dt = min(dt, t2-t)
    newPhi = Phi.copy()
    dx2 = dx*dx
    newPhi[1:-1] += dt/dx2*(Phi[0:-2] - 2*Phi[1:-1] + Phi[2:])
    applyBC(newPhi)
    Phi = newPhi
    t += dt

  return Phi

def init(N=N,t0=1e-2,Phi0=0.1) : 
  Phi = np.zeros(N+2)
  dx = 2*xmax/N
  x = np.arange(-xmax+dx/2, xmax, dx)
  Phi[1:-1] = np.exp(-np.minimum(x*x/(4*t0),100))/t0**0.5 + Phi0
  applyBC(Phi)
  return x, dx, Phi

if __name__ == "__main__" : 
  x, dx, Phi = init()

  iframe = 0
  cfl = 0.8
  dt = cfl*dx**2*0.5
  tstep = 0.1
  t = 0
  tend = 10
  while( t < tend) : 
    Phi2 = evolve(Phi, t, t+tstep, dt, dx)
    Phi = Phi2
    t += tstep
    print("iframe: {0} t={1:.3f}".format(iframe,t))
    pl.clf()
    pl.plot(x, Phi[1:-1],lw=2)
    pl.xlim(-xmax, xmax)
    pl.ylim(0,4.)
    pl.savefig("movie/frame{0:04d}.png".format(iframe))
    iframe += 1
