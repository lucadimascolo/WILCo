import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.constants as const
import astropy.units as u

global    Tcmb; Tcmb    = 2.7255e+00
global  kboltz; kboltz  = const.k_B.value
global hplanck; hplanck = const.h.value

import numpy as np

import sys
import os

from mpi4py import MPI


######################################################################
# SZ spectrum
######################################################################
# Adimenional frequency
# --------------------------------------------------------------------
def getx(freq):
  return hplanck*freq/(kboltz*Tcmb)

# Compton y to Jy/pixel
# --------------------------------------------------------------------
def comptonToKcmb(freq):
  x = getx(freq)
  return (-4.00+x/np.tanh(0.50*x))


######################################################################
# Generic tools
######################################################################
# Normal message
# --------------------------------------------------------------------
def printInfo(message):
  if MPI.COMM_WORLD.Get_rank()==0:
    sys.stdout.write('{0}\n'.format(message))
    sys.stdout.flush()

# Error message
# --------------------------------------------------------------------
def printError(message):
  printInfo(message)
  raise ValueError(42)

# Updating message
# --------------------------------------------------------------------
def printUpdate(message):
  message = message.replace('\n','')

######################################################################
# Fits tools
######################################################################
# Read image fits file
# --------------------------------------------------------------------
def reduceImageFITS(hdu=None,filename=None,ihdu=0):
  if hdu is None: hdu = fits.open(filename)[ihdu]
  endianSYS = sys.byteorder
  endianHDU = interpretEndian(hdu.data.dtype.byteorder)
  if endianHDU not in [endianSYS,'same']:
    hdu.data = hdu.data.byteswap().newbyteorder()
  return hdu.data.astype(np.float64), hdu.header

# Interpret endianity string
# --------------------------------------------------------------------
def interpretEndian(endian):
  if   endian in ['>']: return 'big'
  elif endian in ['=']: return 'same'
  elif endian in ['<']: return 'little'