import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import units as u
from astropy import constants as const

from astropy.convolution import convolve as convolve
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel

import numpy as np

import sys
import os

# ================================================
# Fits tools
# ================================================
# Read image fits file
# ------------------------------------------------
def reduceImageFITS(hdu=None,filename=None,ihdu=0):
  if hdu is None: hdu = fits.open(filename)[ihdu]
  endianSYS = sys.byteorder
  endianHDU = interpretEndian(hdu.data.dtype.byteorder)
  if endianHDU not in [endianSYS,'same']:
    hdu.data = hdu.data.byteswap().newbyteorder()
  return hdu.data.astype(np.float64), hdu.header

# Interpret endianity string
# ------------------------------------------------
def interpretEndian(endian):
  if   endian in ['>']: return 'big'
  elif endian in ['=']: return 'same'
  elif endian in ['<']: return 'little'

# ================================================
# Conversion functions
# ================================================
Tcmb = 2.7255*u.Kelvin

# Adimenional frequency
# ------------------------------------------------
def getx(freq):
  factor = const.h*(freq.to(u.Hz))/(const.k_B*Tcmb)
  return factor.to(u.dimensionless_unscaled).value

# Compton y to Jy/pixel
# ------------------------------------------------
def comptonToKcmb(freq):
  x = getx(freq)
  return (-4.00+x/np.tanh(0.50*x))
