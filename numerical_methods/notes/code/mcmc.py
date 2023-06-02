import numpy as np
import matplotlib.pyplot as pl
import math


def mcmc(func, t, y, x, burn_in=1000, MAX_CHAIN=500000) :
  xchain = [] 
  i = 0

  for j in range(MAX_CHAIN) : 
    x, accepted_move = move(func, t, y, x)
    if( accepted_move) :
      i += 1 
      if( i > burn_in) : 
        xchain.append(x)
  return np.array(xchain)

def move(func, t, y, x, h = 0.1, TINY=1e-30) :
  dx = (2*np.random.rand(x.size)-1.)*h

  xnew = x + dx
  # compute the priors
  p1 = priors(x) 
  p2 = priors(xnew)

  if( p2 <= 0) : 
    return x, False
  
  p = p2/(p1+TINY)

  # compute the likelihoods
  r1 = loglikelihood( func, t, y, x)
  r2 = loglikelihood( func, t, y, xnew)

  r = math.exp(min(r2-r1,0.))

  # accept the move 
  if( np.random.rand() < r*p) :
    return xnew, True

  return x, False 

def priors( x) : 
  if(x[0] < 1 or x[0] > 10) : 
    return 0
  if(x[1] < 0 or x[1] > 2*np.pi) : 
    return 0
  return 1

def loglikelihood( func, t, y, x, sigma=1) : 
  return -(((y-func(t,x))/sigma)**2).sum()

def test_function( t, x) : 
  return x[0]*np.sin(t + x[1]) 

def generate_data() :
  A = 5
  phi = np.pi/4
  sigma = 1.0

  theta = np.arange(-2*np.pi, 2*np.pi, 0.025*np.pi)
  y = A*np.sin(theta + phi) + np.random.normal(0, sigma, theta.size)

  return theta, y, np.array([A, phi])

if __name__ == "__main__" : 
  t, y, x0 = generate_data()
  guess = np.zeros(2)
  guess[0] = 2
  guess[1] = 0
  xchain = mcmc(test_function, t, y, guess, MAX_CHAIN=100000)
  print( "chain length = {2} MCMC mean = {0}, actual = {1}".format(str(np.average(xchain, axis=0)), str(x0), xchain.shape[0]))
  pl.hist2d(xchain[:,0],xchain[:,1],bins=100)
  pl.show()
  pl.clf()
  pl.hist(xchain[:,0],bins=100)
  pl.savefig("mcmc_1.pdf")
