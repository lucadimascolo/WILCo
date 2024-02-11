from .utils import *

import scipy.linalg

# ================================================
# Wavelets
# ================================================
class wavelet:
  def __init__(self,nd,data,kernel='cosine'):
    if nd>1:
      freq = np.hypot(np.fft.fftfreq(data.shape[0])[:,None],
                      np.fft.fftfreq(data.shape[1])[None,:])

      if kernel=='cosine':
        npeaks = np.maximum(np.abs(np.fft.fftfreq(data.shape[0]).max()),
                            np.abs(np.fft.fftfreq(data.shape[1]).max()))
        npeaks = npeaks*np.linspace(0,1,nd)#**0.50
        
      # self.scales = np.minimum(data.shape[0],data.shape[1])/(npeaks[1]-npeaks[0])
      # self.scales = np.append(self.scales,0.50*np.minimum(data.shape[0],data.shape[1])/npeaks[1:])
      # self.scales = self.scales*0.25

        self.scales = np.minimum(data.shape[0],data.shape[1])*(0.50-npeaks)

        self.base = [np.cos(0.50*np.pi*freq/(npeaks[1]-npeaks[0]))]
        self.base[0][freq>npeaks[1]] = 0.00
      
        for pi in range(1,len(npeaks)-1):
          base = np.zeros(freq.shape)
          cut1 = np.logical_and(freq>=npeaks[pi-1],freq<npeaks[pi])
          cut2 = np.logical_and(freq>=npeaks[pi],freq<npeaks[pi+1])

          base[cut1] = np.cos(0.50*np.pi*(npeaks[pi]-freq[cut1])/(npeaks[pi]-npeaks[pi-1]))
          base[cut2] = np.cos(0.50*np.pi*(freq[cut2]-npeaks[pi])/(npeaks[pi+1]-npeaks[pi]))

          self.base.append(base)

        cut2 = np.logical_and(freq>=npeaks[-2],freq<npeaks[-1])
        base = np.zeros(freq.shape)
        base[cut2] = np.cos(0.50*np.pi*(npeaks[-1]-freq[cut2])/(npeaks[-1]-npeaks[-2]))
        base[freq>=npeaks[-1]] = 1.00

        self.base.append(base)
        self.base = np.asarray(self.base)

      self.kernel = np.array([np.exp(-0.50*(si*freq)**2) for si in range(self.scales)])

# Apply wavelet filters
# ------------------------------------------------
  def fft(self,data):
    factor = np.fft.fft2(np.fft.fftshift(data))
    factor = np.array([factor*base for base in self.base])
    return np.fft.ifftshift(np.fft.ifft2(factor,axes=(-2,-1)),axes=(-2,-1)).real

# ================================================
# WILCo
# ================================================
# Main structure
# ------------------------------------------------
class build:
  def __init__(self,imglist=None,maps=None,needlet=0):
    if   (imglist is None) and (maps is not None):
      self.maps = maps
    elif (imglist is not None) and (maps is None):
      self.maps = getdata(imglist)

    self.needlet = wavelet(needlet,self.maps[0].data)

    maplist = np.array([self.needlet.fft(m.data) for m in self.maps])
    maplist = np.swapaxes(maplist,0,1)
    
    fftlist = np.fft.fft2(maplist,axes=(-2,-1))
    fftkern = self.needlet.kernel.copy() 
    
    for nd in range(maplist.shape[0]):
      for ni in range(maplist.shape[1]):
        fftlist[nd,ni] = fftlist[nd,ni]*fftkern[nd]

    fftlist = np.fft.ifft2(fftlist,axes=(-2,-1)).real

    covlist = np.zeros((maplist.shape[0],maplist.shape[-2],maplist.shape[-1],maplist.shape[1],maplist.shape[1],))
    for nd in range(maplist.shape[0]):
      for ni in range(maplist.shape[1]):
        for nj in range(maplist.shape[1]):
          covlist[nd,...,ni,nj] = (maplist[nd,ni]-fftlist[nd,ni])*(maplist[nd,nj]-fftlist[nd,nj])
          covlist[nd,...,nj,ni] = covlist[nd,...,ni,nj].copy()
      
      covkern = np.fft.fft2(covlist[nd],axes=(1,2))
      covkern = covkern*fftkern[nd][...,None,None]
      covlist[nd] = np.fft.ifft2(covkern,axes=(1,2)).real

    self.covlist = np.linalg.inv(covlist)
    self.maplist = maplist.copy()

# Obtained WILCo weights
# ------------------------------------------------
  def getw(self,bounds=[{'type': 'cmb', 'value': 1.00}],):
    cond = np.zeros(len(bounds))
    spec = np.zeros((self.covlist.shape[-1],len(bounds)))
    for b, bound in enumerate(bounds):
      if   bound['type']=='cmb': spec[:,b] = 1.00
      elif bound['type']=='tsz': 
        freq = np.array([m.freq.to(u.Hz).value for m in self.maps])*u.Hz
        spec[:,b] = comptonToKcmb(freq)

      cond[b] = bound['value']
    
    wilc = np.zeros(self.covlist[...,0].shape)
    for nd in range(self.covlist.shape[0]):
      cv = np.einsum('ij,bcjj->bcij',spec.T,self.covlist[nd])
      cw = np.einsum('ij,bcjj,jk->bcik',spec.T,self.covlist[nd],spec)
      cw = np.linalg.inv(cw)

      cw = np.einsum('bcij,bcjk->bcik',cw,cv)
      wilc[nd] = np.einsum('i,bcij->bcj',cond,cw)
    
    wilc = np.moveaxis(wilc,-1,1)
    return wilc

# Recombine maps
# ------------------------------------------------
  def combine(self,wilc):
    milc = np.sum(self.maplist*wilc,axis=0)
    return milc
     
# ================================================
# Support functions
# ================================================
# Build data structure
# ------------------------------------------------
def getdata(imglist):
  maps = []

  for i, img in enumerate(imglist):
    imgdata, imghead = reduceImageFITS(filename=img['name'],ihdu=img.get('ihdu',0))

    if imgdata.ndim>2: imgdata = imgdata[img.get('index',0)]

    imgfreq = img['freq'].to(u.Hz)        

    maps.append(type('field',(object,),{'data': imgdata, 
                                      'header': imghead, 
                                        'freq': imgfreq}))
  return maps

