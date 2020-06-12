from .utils import *

from scipy.ndimage import binary_erosion
from scipy.ndimage import gaussian_filter

######################################################################
class getdata:
  def __init__(self,imglist,cuts=1,idxdata=(0,...),**kwargs):
    
    if type(cuts)==int: numcuts = cuts
    else: numcuts = cuts.shape[0]+1

    frqdata = []
    fftdata = []
    for i, img in enumerate(imglist):
      imgdata, imghead = reduceImageFITS(None,img.get('name'),img.get('ihdu',0))
      if imgdata.ndim>2:imgdata = imgdata[idxdata]
      imgdata = apodize(imgdata/1.00E+06,**kwargs)

      frqdata.append(img.get('freq'))
      fftdata.append(np.fft.fft2(imgdata))
    frqdata = np.asarray(frqdata)
    fftdata = np.asarray(fftdata)
    fftdata = np.broadcast_to(fftdata[np.newaxis,...],(numcuts,*fftdata.shape)).copy()

    if numcuts>1: flatcut(fftdata,cuts)

    self.maps = np.fft.ifft2(fftdata,axes=(-2,-1)).real
    self.freq = frqdata

# Top-hat wavelet
# --------------------------------------------------------------------
def flatcut(data,cuts):
  freq = np.hypot(np.outer(np.ones(data.shape[-2]),np.fft.fftfreq(data.shape[-1])),
                  np.outer(np.fft.fftfreq(data.shape[-2]),np.ones(data.shape[-1])))
  edge = np.abs(np.fft.fftfreq(data.shape[-1])).max()

  if type(cuts)==int:
    cuts = np.array([1.00*c/cuts for c in range(1,cuts)])
  
  data[ 0,:,freq>cuts[ 0]*edge] = 0.00+0.00j
  data[-1,:,freq<cuts[-1]*edge] = 0.00+0.00j
  for c, cut in enumerate(cuts[:-1]):
    mask = np.logical_and(freq>=cuts[c]*edge,freq<cuts[c+1]*edge)
    data[c,:,~mask] = 0.00+0.00j

# Simple Gaussian apodization
# --------------------------------------------------------------------
def apodize(inpdata,pad=0,apo=2,apofact=1):
  outdata = np.ones_like(inpdata)
  if apo:
    outdata = binary_erosion(np.ones_like(inpdata),iterations=int(apofact*apo))
    outdata = gaussian_filter(outdata.astype(inpdata.dtype),apo,mode='constant',cval=0.00) 
  if pad: outdata = np.pad(outdata,pad)
  return outdata*inpdata
