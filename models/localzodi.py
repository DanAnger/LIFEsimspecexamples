import numpy as np

from modules import constants

class Localzodi:
    def __init__(self, model = "glasse"):
        self.model = model
        if model == "glasse":
            self.T_glasse = 270
            self.epsilon = 4.30e-8
            
        elif model == "darwinsim":
            self.R_sun_au = 0.00465047 #in AU
            self.tau = 4e-8
            self.Teff = 265 
            self.Tsun =5777
            self.A = 0.22
            
        else:
            print("model unspecified. Glasse is used")
            self.T = 270
            self.epsilon = 4.30e-8       

    def Bgamma(self,T, wav):
        
        '''
        A function that returns the black body function in photons
        for a given temperature and wavelength in m.
        
        Parameters
        ----------
        wav : float
            Wavelength in m.
        T : float
            Temperature in Kelvin.
            
        Returns
        -------
        ret : array
            Bgamma in photons/s/m^2/micron/sr for given wavelength.
        '''
        
        k1 = 2 * constants.c
        k2 = (constants.h * constants.c) / constants.k
        fact1 = k1/(wav**4)
        fact2 = k2/(T * wav)
        Bgamma  =  np.array(fact1/(np.exp(fact2)-1.0)) * 1e-6
        
        return Bgamma        
        
    def fgamma_sr(self,wav, long=3/4 * np.pi, lat=0):
        
        if self.model == "glasse":
            fgamma_sr = self.epsilon * self.Bgamma(self.T_glasse,wav)
            
        elif self.model == "darwinsim":
            Btot = self.Bgamma(self.Teff, wav) + self.A * self.Bgamma(self.Tsun, wav) * (self.R_sun_au/ 1.5)**2
            
            fgamma_sr = self.tau * Btot * np.sqrt(
                np.pi / np.arccos(np.cos(long) * np.cos(lat)) /
                (np.sin(lat)**2 + (0.6 * (wav/11e-6)**(-0.4) * np.cos(lat))**2)
                )
            
        return fgamma_sr
    
    
        
        
        
    

    