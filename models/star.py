import numpy as np

from modules import constants


class Star:
    @classmethod
    def __init__(self, planet=None):
        
        if planet is not None:
            self.Ds = planet.sdist               # distance to star in parsec
            self.Rs = planet.Rs                 # star radius R_sun
            self.Rs_m = planet.Rs_m             # star radius in m
            self.Ts = planet.Ts
            self.Ls = planet.Ls
            
            self.Sin = planet.s_eff_in          # in earth insulation
            self.Sout = planet.s_eff_out        # in earth insulation
            self.HZin = planet.HZ_in            # in AU
            self.HZout = planet.HZ_out
            self.HZcenter = planet.HZ_center
            self.z = planet.zodis
            
            #self.RA = planet.RA
            #self.Dec = planet.Dec
            self.lat = planet.lat
                      
        else:
            self.Ds = 10.
            self.Rs = 1.
            self.Rs_m = self.Rs * constants.R_sun
            self.Ts = 1. * constants.T_sun
            self.Ls = 1.
            
            self.Sin = 1.7665
            self.Sout =  0.324
            self.HZin = np.sqrt(self.Ls / self.Sin)
            self.HZout = np.sqrt(self.Ls / self.Sout)
            self.HZcenter = np.mean((self.HZin, self.HZout))
            self.z = 1
                 
            self.lat = 1./4. * np.pi


    

    def fgamma(self,wav):
        '''A function that returns the black body spectral flux in
        photons/s/m^2/micron at one or more given wavelengths (in m).
        
        Parameters
        ----------
        wav : float
            Wavelength in m.
            
        Returns
        -------
        fgamma : array
            Fgamma in photons/s/m^2/micron for given wavelength.
        '''
        
        k1 = 2 * constants.c
        k2 = (constants.h * constants.c) / constants.k    
        fact1 = k1/(wav**4)
  
        fact2 = k2/(self.Ts * wav)
        fgamma = np.array(fact1/(np.exp(fact2)-1.0)) * 1e-6 *np.pi * (
            (self.Rs * constants.R_sun) / (self.Ds * constants.m_per_pc))**2
        
        return fgamma
    
