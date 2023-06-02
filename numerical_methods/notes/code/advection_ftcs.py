import numpy as np
import matplotlib.pyplot as pl
import math

cfl = 0.8
v = 1
L = 1

FTCS = 0
UPWIND = 1

TOPHAT = 0
GAUSSIAN = 1
PROFILE = GAUSSIAN

MODE = UPWIND
MOVIE = False

def ftcs(f, dt, dx, cfl=cfl) :
  newf = f 
  newf[1:-1] -= v*(f[2:] - f[0:-2])/(2*dx)*dt
  newf[0] -= v*(f[1] - f[-1])/(2*dx)*dt
  newf[-1] -= v*(f[0] - f[-2])/(2*dx)*dt
  return newf

def upwind(f, dt, dx, cfl=cfl) :
  newf = f 
  newf[1:] -= v*(f[1:] - f[0:-1])/dx*dt
  newf[0] -= v*(f[0] - f[-1])/dx*dt
  return newf

def initial_conditions(x) : 
  N = x.size
  
  f = np.ones(N)*0.5
  
  # top hat
  if( PROFILE == TOPHAT) :
    f[int(0.25*N):int(0.75*N)] = 1 # top hat profile
  elif( PROFILE == GAUSSIAN) : 
    # gaussian profile
    sigma = 0.125*L
    f += np.exp(-(x-x.mean())**2/sigma**2)

  return f

def error(f, ftrue) : 
  return np.sqrt(np.average((f - ftrue)**2))

def run_model(MODE, N=100) :

  dx = L/N
 
  tend = 1
  t = 0
  x = np.arange(0,L,dx)

  f = initial_conditions(x)
  i = 0

  while( t < tend) : 
    dt = min(tend-t,math.fabs(cfl*dx/v))
    if( MODE == FTCS) : 
      f = ftcs(f, dt, dx, cfl=cfl) 
    else : 
      f = upwind(f, dt, dx, cfl=cfl)
    t += dt

    if MOVIE : 
      i += 1
      pl.clf()
      pl.plot( x, f)
      pl.ylim(0.,2.0)
      print("Writing frame {0}".format(i))
      pl.savefig("movie/frame{0:04d}.png".format(i))

  if MOVIE: 
    pl.clf()
    pl.plot( x, f)
    pl.ylim(-0.5,1.5)
    pl.savefig("advect.pdf")
  return f

if __name__ == "__main__" : 
  import argparse
  parser = argparse.ArgumentParser(description='Run advection')
  parser.add_argument('--run_error', action='store_true', default=False,
                    help='make a graph of E vs T')

  parser.add_argument('--ftcs', action='store_true', default=False,
                    help='make a graph of E vs T')
  parser.add_argument('--movie', action='store_true', default=False,
                    help='make a graph of E vs T')

  args = parser.parse_args() 

  if args.ftcs :
    MODE = FTCS
  else : 
    MODE = UPWIND

  if not args.run_error : 
    MOVIE = args.movie
    run_model(MODE)
  else : 
    lgNs = np.arange(1.3,3.3,0.25)
    errors = []
    for lgN in lgNs :
      N = int(1e1**lgN) 
      dx = L/N

      x = np.arange(0,L,dx)

      ftrue = initial_conditions(x)
      f = run_model(MODE,N=N)
      errors.append( error(f,ftrue))

    pl.clf()
    pl.loglog(1e1**lgNs,errors)
    pl.savefig("advect_conv.pdf")
