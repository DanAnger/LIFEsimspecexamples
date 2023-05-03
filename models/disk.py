import numpy as np

class Disk(object):
    '''Disk class
    
    Contains methods to generate radial flux and surface brightness profiles
    for a simple power-law axisymmetric optically-thin disk. 
    
    Parameters
    ----------
    lstar : float
        Stellar luminosity in L_sun.
    dist : float
        Distance to star in pc.
    alpha : float
        Disk surface density power-law index.
    zodis : float
        Number of zodis in disk, simple multiplier.
    wav : float
        Wavelength of interest in m.
    rin : float
        Disk inner radius in AU.
    rout : float
        Disk outer radius in AU.
    r0 : float
        Disk reference radius in AU.
    nr : int
        Number of radial locations in disk.
        
    Attributes
    ----------
    r : array
        Calculates value at centre of each radial bin.
    rarcs : array
        r in arcseconds.
    tbb : array
        Black body temperature of each bin centre r.
    sigzodi : float
        1 zodi surface density by defintion after integrating Kelsall model
        g(xi) over column to get multiplier of 0.629991 for 1.13e-7 volume 
        density.
    bigsig : array
        Disk flux density. AU^2/AU^2 in each of the bins.
    aream : array
        Area in AU^2 of each bin.
    sig : array
        Total dust area in each of the r bins in AU^2.
    '''        
    
    
    
    def __init__(self,Ls=1,dist=10,alpha=0.34,zodis=1.0,
               wav=10e-6,rin=0.034422617777777775,rout=10.0,r0=1.0,nr=256,
               maps=True, star=None, options=None,
               maspp=None, norm1z=False):
        '''Initialise, default alpha=0.34, zodis=1.0, wav=10.0e-6, 
        rin=0.034422617777777775, rout=10.0, r0=1.0, nr=500
        '''
        '''
        if Ls == None:
            raise Exception('Must give lstar')
            
        if dist == None:
            raise Exception('Must give dist')
        '''
            
        if star is not None:
            self.Ls = star.Ls
            self.dist = star.Ds
            self.alpha = float(alpha)
            if norm1z == True:
                self.zodis = float(1)
            else:
                self.zodis = star.z
            self.wav = float(wav)
            
            self.rin = float(rin)*np.sqrt(self.Ls)
            self.rout = float(rout)*np.sqrt(self.Ls)
            self.r0 = float(r0)*np.sqrt(self.Ls)
        
        else:
            self.Ls = float(Ls)
            self.dist = float(dist)
            self.alpha = float(alpha)
            self.zodis = float(zodis)
            self.wav = float(wav)
            
            self.rin = float(rin)*np.sqrt(self.Ls)
            self.rout = float(rout)*np.sqrt(self.Ls)
            self.r0 = float(r0)*np.sqrt(self.Ls)

        # Maurice: this is my adaption to a 2D sky-image
        if maps==True:
            maspp = np.array([maspp])
            if maspp.shape[-1] >1:
                maspp = np.reshape(maspp, (maspp.shape[-1],1,1))
            
            self.au_pp = maspp / 1e3 * self.dist
    
            self.r_au = options.r_map * self.au_pp
            
            r_cond = ((self.r_au >= self.rin)
                      & (self.r_au <= options.image_size/2 * self.au_pp))
            
            self.tbb = np.where(r_cond,
                                278.3*(self.Ls**0.25)/np.sqrt(self.r_au), 0)
            
            self.sigzodi = 7.11889e-8 
            self.aream = self.au_pp ** 2
            '''
            self.bigsig = np.where(r_cond, self.sigzodi*self.zodis *
                                   (self.r_au/self.r0)**(-self.alpha), 0)
            self.sig = self.aream * self.bigsig
            '''
            self.sig = np.where(r_cond,
                                self.aream * self.sigzodi * self.zodis *
                                (self.r_au/self.r0)**(-self.alpha), 0)
    
    def bnu_Jy_sr(self,wav,t):
        '''A function that returns the black body spectral radiance in Jy/sr 
        for a given temperature and at one or more given wavelengths (in m).
        
        Parameters
        ----------
        wav : float
            Wavelength in m.
        t : array
            Temperature array in Kelvin.
            
        Returns
        -------
        ret : array
            Bnu in Jy/sr for given temperature and wavelength.
        '''
        

        k1 = 3.9728949e1     
        k2 = 1.438774e-2     
        t = np.array(t,dtype=float)
        fact1 = k1/(wav**3)
        with np.errstate(divide='ignore'):
            fact2 = k2/(t*wav)
            ret = np.array(fact1/(np.exp(fact2)-1.0))
        return ret
    
    
    def fnu_disk(self,wav):
        '''Returns the flux profile for a simple power-law axisymmetric 
        optically-thin disk.
        '''
        fnudisk = 2.95e-10 * self.bnu_Jy_sr(wav,self.tbb) * (
            0.25*self.sig/np.pi) / self.dist**2        
        return fnudisk
    
    
    #function by Maurice to convert Jy to photonss/m^2/s/micron
    def ph_flux(self,wav):
        '''Returns the photon number flux profile for a simple
        power-law axisymmetric optically-thin disk.
        '''
        wav = np.array([wav])
        if wav.shape[-1] >1:
            wav = np.reshape(wav,(wav.shape[-1],1,1))
        ph_flux = self.fnu_disk(wav) * 1.509e1 / wav
        return ph_flux
