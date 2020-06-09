import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

import sys
import os

from mpi4py import MPI

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