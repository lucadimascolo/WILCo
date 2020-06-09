from .utils import *

from scipy.ndimage import binary_erosion
from scipy.ndimage import gaussian_filter

######################################################################
class getdata:
  def __init__(self,imglist,numkern=1,**kwargs):
    
    fftdata = []
    for i, img in enumerate(imglist):
      data, head = reduceImageFITS(None,img['name'],img['ihdu'])
      data = apodize(data,**kwargs)
      fftdata.append(np.fft.fft2(data))

    fftkern = np.zeros((numkern,*fftdata.shape))
    fftfreq = np.hypot(np.outer(np.ones(fftdata.shape[-2]),np.fft.fftfreq(fftdata.shape[-1])),
                       np.outer(np.fft.fftfreq(fftdata.shape[-2]),np.ones(fftdata.shape[-1])))
    fftedge = np.abs(np.fft.fftfreq(fftdata.shape[-1])).max()/numkern

    for nk in range(numkern):
      fftindx = np.logical_and(fftfreq>nk*fftedge,fftfreq<(nk+1)*fftedge)
      fftkern[nk,:,fftindx] = 1.00

    fftdata = np.asarray(fftdata)
    fftdata = np.broadcast_to(fftdata[np.newaxis,...],fftkern.shape)
    fftdata = fftdata*fftkern

    self.maps = np.fft.ifft2(fftdata,axes=(-2,-1)).real

        
# Simple Gaussian apodization
# --------------------------------------------------------------------
def apodize(inpdata,pad=0,apo=2,apofact=5):
  outdata = np.ones_like(inpdata)
  if apo:
    outdata = binary_erosion(np.ones_like(inpdata),iterations=apofact*apo)
    outdata = gaussian_filter(outdata.astype(inpdata.dtype),apo,mode='constant',cval=0.00) 
  if pad: outdata = np.pad(outdata,pad)
  return outdata*inpdata