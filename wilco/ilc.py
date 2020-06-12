from .utils import *

import scipy.linalg


class ilc:
  def __init__(self,data,bounds=[{'type': 'cmb', 'value': 1.00}]):
    mapsum = np.zeros(data.maps[:,0].shape)
    mapnum =data.maps.shape[1]

    for m, maps in enumerate(data.maps):
      mapcov = np.zeros((mapnum+len(bounds),mapnum+len(bounds)))
      mapvec = np.zeros((mapnum+len(bounds)))

      for imap in range(mapnum):
        for jmap in range(mapnum):
          mapcov[imap,jmap] = 2.00*np.sum(maps[imap]*maps[jmap])

      concov = np.zeros((mapnum,len(bounds)))
      for b, bound in enumerate(bounds):
        if bound['type']=='cmb':
          concov[:,b] = 1.00
          mapvec[mapnum+b] = bound['value']
        elif bound['type']=='sz':
          concov[:,b] = comptonToKcmb(data.freq)
          mapvec[mapnum+b] = bound['value']

      mapcov[:mapnum,mapnum:] = -concov.copy()
      mapcov[mapnum:,:mapnum] =  concov.copy().T

      wgtvec = scipy.linalg.lstsq(mapcov,mapvec)[0]

      mapsum[m] = np.sum(np.array([wgt*maps[w] for w, wgt in enumerate(wgtvec[:mapnum])]),axis=0)
      
      plt.imshow(mapsum[m])
      plt.show()
