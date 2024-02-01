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
  def __init__(self,imglist=None,maps=None,patches=None,needlet=0):
    if   (imglist is None) and (maps is not None):
      self.maps = maps
    elif (imglist is not None) and (maps is None):
      self.maps = getdata(imglist)

    self.needlet = wavelet(needlet,self.maps[0].data)

    maplist = np.array([self.needlet.fft(m.data) for m in self.maps])
    maplist = np.swapaxes(maplist,0,1)
    
    if patches==1 or patches is None:
      maplist = [[maplist]]
    elif len(patches)==2:

      maplist = np.array_split(maplist,patches[0],axis=-2)
      maplist = [np.array_split(m,patches[1],axis=-1) for m in maplist]

    # px, py, fft, freq, x, y
    self.icov = [[[np.cov(ni.reshape(ni.shape[0],-1)) for ni in mi] for mi in mj] for mj in maplist]
    self.icov = np.asarray(self.icov)

    print(self.icov[0][0][0])
    
# Obtained WILCo weights
# ------------------------------------------------
  def getw(self,bounds=[{'type': 'cmb', 'value': 1.00}],):
    cond = np.zeros(len(bounds))
    spec = np.zeros((self.icov.shape[-1],len(bounds)))
    for b, bound in enumerate(bounds):
      if   bound['type']=='cmb': spec[:,b] = 1.00
      elif bound['type']=='tsz': 
        freq = np.array([m.freq.to(u.Hz).value for m in self.maps])*u.Hz
        spec[:,b] = comptonToKcmb(freq)

      cond[b] = bound['value']

    wilc = np.zeros(self.icov[...,0].shape)
    for ci in range(self.icov.shape[0]):
      for cj in range(self.icov.shape[1]):
        for ni in range(self.icov.shape[2]):
          cv = np.matmul(spec.T,self.icov[ci,cj,ni].T)
          cw = np.matmul(spec.T,np.matmul(self.icov[ci,cj,ni],spec))
          wilc[ci,cj,ni] = np.matmul(cond,np.matmul(np.linalg.inv(cw),cv))
    return wilc

# Recombine maps
# ------------------------------------------------
  def combine(self,wilc,mode='tile'):
    maplist = np.array([self.needlet.fft(m.data) for m in self.maps])
    maplist = np.swapaxes(maplist,0,1)

    mapwilc = np.zeros(maplist.shape)

    mapwilc = np.array_split(mapwilc,self.icov.shape[0],axis=-2)
    mapwilc = [np.array_split(m,self.icov.shape[1],axis=-1) for m in mapwilc]

    for ci in range(self.icov.shape[0]):
      for cj in range(self.icov.shape[1]):
        for ni in range(self.icov.shape[2]):
          mapwilc[ci][cj][ni] = np.broadcast_to(wilc[ci,cj,ni][:,None,None],mapwilc[ci][cj][ni].shape).copy()

    if mode in ['tile','smooth']:
      parwilc = []
      for ni in range(self.icov.shape[2]):
        onewilc = np.array([np.concatenate([np.concatenate([mi[ni][ci] for mi in mj],axis=1) for mj in mapwilc],axis=0) for ci in range(mapwilc[0][0][ni].shape[0])])

        if mode=='smooth':
          mapkern = Gaussian2DKernel(np.minimum(onewilc.shape[-2]//self.icov.shape[0],
                                                onewilc.shape[-1]//self.icov.shape[1]))
          onewilc = [convolve(m,mapkern,boundary='extend') for m in onewilc]
      
        parwilc.append(onewilc)
      parwilc = np.asarray(parwilc)

    outwilc = np.sum(parwilc*maplist,axis=(1))

    return outwilc, parwilc

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

