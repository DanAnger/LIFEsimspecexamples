import numpy as np
import matplotlib.pyplot as plt
from modules import constants


class Planet:
    '''Planet class
    
    Used to define the planet parameters and contains methods to generate
    the emitted flux of the planet
    
    Parameters
    ----------

    sdist : float
        Distance to star in pc.
    Rs : float
        Radius of host star in stellar radius.
    Rs_m : float
        Radius of host star in meters.
    Ts : float
        Temperature of host star in Kelvin.
    Ls : float
        Luminosity of host star in solar luminosities
    Rp : float
        Radius of planet in earth radius.
    Rp_m : float
        Radius of planet in meters.
    Tp : float
        Temperature of planet in Kelvin.
    a : float
        Semi-major axis of planet in AU.
    ang_sep : float
        Angular separation of planet from host star in arcsec
    zodis : float
        zodi number (surface density of exozodiacal dust in HZ)
    
        
    Attributes
    ----------

    '''        
    
    @classmethod
    def init_earth_twin(cls, sdist=10, Rp=1, a=1, Tp=276, spectrum_type="blackbody"):
        self = cls()
        self.sdist = sdist                       # distance to star in parsec
        self.Rs = 1.                            # star radius R_sun
        self.Rs_m = self.Rs * constants.R_sun   # star radius in m
        self.Ts = 1. * constants.T_sun
        self.Ls = 1.
        self.planet_radius_px = 0.              # planet radius in pixel
        self.Rp = Rp                            # planet radius in R_earth
        self.Rp_m = self.Rp * constants.R_earth #planet radius in m
        self.Tp = Tp
        self.a = a
        self.ang_sep = self.a / self.sdist      # angular separation in arcsec
        self.s_eff_in = 1.7665                  # in earth insulation
        self.s_eff_out =  0.324                 # in earth insulation
        self.HZ_in = np.sqrt(self.Ls / self.s_eff_in)   #in AU
        self.HZ_out = np.sqrt(self.Ls / self.s_eff_out) #in AU
        self.HZ_center = np.mean((self.HZ_in, self.HZ_out))
        self.zodis = 3
        
        #self.RA = np.pi
        #self.Dec = 1 / 4 * np.pi 
        #ep = 23.4 / 180. * np.pi
        self.lat  = 1 / 4 * np.pi
                
        self.spectrum_type = spectrum_type
        
        return self
    
    @classmethod
    def init_planet(cls,
                    sdist=10,
                    Rs=1,
                    Ts=1,
                    Rp=1,
                    a=1,
                    Tp=276,
                    zodis=1,
                    spectrum_type="black_body",
                    path_to_spec=None,
                    spec=None):
        
        self = cls()
        self.sdist = float(sdist)                       # distance to star in parsec
        self.Rs = float(Rs)                           # star radius R_sun
        self.Rs_m = self.Rs * constants.R_sun   # star radius in m
        self.Ts = float(Ts)
        self.Ls = self.Rs**2 * (self.Ts / 5780) **4     # luminosity in L_sun
        self.Rp = float(Rp)                           # planet radius in R_earth
        self.Rp_m = self.Rp * constants.R_earth #planet radius in m
        self.Tp = float(Tp)
        self.a = float(a)
        
        self.ang_sep = self.a / self.sdist      # angular separation in arcsec
        self.zodis = float(zodis)
        
        self.ComputeHZ(Model = "MS")   

        self.lat  = 1 / 4 * np.pi
                
        self.spectrum_type = spectrum_type
        self.path_to_spec = path_to_spec
        self.spec = spec

        return self

    
    def fgamma(self, wl, wl_bin_edges=None):
        '''A function that returns the black body spectral flux
        in photons/s/m^2/micron at one or more given wavelengths (in m).
        
        Parameters
        ----------
        wl : float
            Wavelength in m.
            
        Returns
        -------
        fgamma : array
            Photon flux in photons/s/m^2/micron for given wavelength.
        '''
        
        if self.spectrum_type == "blackbody":
            k1 = 2 * constants.c
            k2 = (constants.h * constants.c) / constants.k    
            fact1 = k1/(wl**4)
      
            fact2 = k2/(self.Tp * wl)
            fgamma = np.array(fact1/(np.exp(fact2)-1.0)) * 1e-6 *np.pi * (
                (self.Rp * constants.R_earth) / (self.sdist * constants.m_per_pc)
                )**2


        if self.spectrum_type == "earth_spec":
            spec = np.loadtxt("/home/felix/Documents/MA/LIFEsim/lifesim/gui/Earth_Clear_R1000_10pc_Björn_Konrad.txt").T
            spec_wl = spec[0] * 1e-6 # per micron to per m
            spec_value = spec[1] / 3600.  # hours to seconds
            
            spec_value *= (10. / self.sdist)**2 # scale planet spectrum with distance
            
            fgamma = self.bin_spectrum(wl_bin_edges, spec_wl, spec_value) 
            
            
        if self.spectrum_type == "spec_wl":
            spec = self.spec
            spec_wl = spec[0] 
            spec_value = spec[1]
                        
            fgamma = self.bin_spectrum(wl_bin_edges, spec_wl, spec_value) 

        if self.spectrum_type == "contrast_to_star_wl_mum":
            k1 = 2 * constants.c
            k2 = (constants.h * constants.c) / constants.k    
            fact1 = k1/(wl**4)
      
            fact2 = k2/(self.Ts * wl)
            fgamma_star = np.array(fact1/(np.exp(fact2)-1.0)) * 1e-6 * np.pi * (
                (self.Rs * constants.R_sun) / (self.sdist * constants.m_per_pc)
                )**2

            spec = np.loadtxt(self.path_to_spec).T
            spec_wl = spec[0] * 1e-6         
            spec_contrast = spec[1]
            spec_contast_binned = self.bin_spectrum(wl_bin_edges, spec_wl, spec_contrast)
            
            fgamma = fgamma_star * spec_contast_binned
    
            
        if self.spectrum_type == "contrast_to_star_wn":
            k1 = 2 * constants.c
            k2 = (constants.h * constants.c) / constants.k    
            fact1 = k1/(wl**4)
      
            fact2 = k2/(self.Ts * wl)
            fgamma_star = np.array(fact1/(np.exp(fact2)-1.0)) * 1e-6 * np.pi * (
                (self.Rs * constants.R_sun) / (self.sdist * constants.m_per_pc)
                )**2

            spec = np.loadtxt(self.path_to_spec).T

            spec_wn = spec[0] * 1e2 # per cm to per m
            spec_wl = 1 / spec_wn
            
            spec_contrast = spec[1]
            
            spec_contast_binned = self.bin_spectrum(wl_bin_edges, spec_wl, spec_contrast)
            
            fgamma = fgamma_star * spec_contast_binned

        return fgamma
    
    
    
    def bin_spectrum(self, wl_bin_edges, spec_wl, spec_value):
        
        bins = np.digitize(spec_wl, wl_bin_edges)
        bins_mean = [spec_value[bins == i].mean() for i in range(1, len(wl_bin_edges))]
        bins_mean = np.array(bins_mean)
        
        return bins_mean


    def ComputeHZ(self, Model='MS'):
        """
        Parameters
        ----------
        Model: MS, POST-MS
            Model for computing the habitable zone (au).
        """
        
        # Compute the habitable zone (au).
        if (Model == 'MS'):
            S0in, S0out = 1.7665, 0.3240
            Ain, Aout = 1.3351E-4, 5.3221E-5
            Bin, Bout = 3.1515E-9, 1.4288E-9
            Cin, Cout = -3.3488E-12, -1.1049E-12

        elif (Model == 'POST-MS'):
            S0in, S0out = 1.1066, 0.3240
            Ain, Aout = 1.2181E-4, 5.3221E-5
            Bin, Bout = 1.5340E-8, 1.4288E-9
            Cin, Cout = -1.5018E-12, -1.1049E-12

        else:
            print('--> WARNING: '+str(Model)+' is an unknown model')
            Model = 'MS'
            S0in, S0out = 1.7665, 0.3240
            Ain, Aout = 1.3351E-4, 5.3221E-5
            Bin, Bout = 3.1515E-9, 1.4288E-9
            Cin, Cout = -3.3488E-12, -1.1049E-12

        T_star = self.Ts - 5780
        self.s_eff_in = S0in + Ain * T_star + Bin*T_star**2 + Cin*T_star**3 #in units of incoming flux, normalized to stellar flux on earth_0
        self.s_eff_out = S0out + Aout * T_star + Bout*T_star**2 + Cout*T_star**3
        self.Ls = self.Rs**2 * (self.Ts / 5780) **4     # luminosity in L_sun
        self.HZ_in = np.sqrt(self.Ls / self.s_eff_in)   # HZ inner boundery in AU
        self.HZ_out = np.sqrt(self.Ls / self.s_eff_out) # HZ outer boundery in AU
        self.HZ_center = (self.HZ_in + self.HZ_out)/2   # HZ center  in 

        pass





