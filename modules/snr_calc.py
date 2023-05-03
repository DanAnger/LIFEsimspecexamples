import numpy as np
import matplotlib.pyplot as plt

import os


from modules import transmission_generator as tm_gen
from modules.plotting import plotter

from models.planet import Planet
from models.instrument import Instrument
from models.disk import Disk
from models.localzodi import Localzodi
from models.star import Star

    



class SnrCalc:
    def __init__(self,
                 planet,
                 options,
                 instrument,
                 plot_snr_spectrum=False,
                 save_snr_spectrum=False):
        
        self.planet = planet
        self.options = options
        self.inst = instrument
     
        
        self.star = Star(planet)
        self.lz = Localzodi(model = options.lz_model)
        self.ez = Disk(star=self.star,
                       options=options,
                       maps=True,
                       maspp=self.inst.maspp)
                      
        self.inst.adjust_bl_to_HZ(self.star, options)
    
        self.inst.add_transmission_maps(options, map_selection="tm3_only")
   
    def predict_SNR(self,
                    plot=True):
        
        options = self.options

      
        N_sl = self.inst.get_stellar_leakage(self.star, options)
        N_lz = self.inst.get_localzodi_leakage(self.lz, self.star, options)
        N_ez = self.inst.get_exozodi_leakage(self.ez, options)
    
        trans_eff_p = tm_gen.transm_eff(self.inst.bl, self.inst.wl_bins, self.planet.ang_sep)
        Fp = self.planet.fgamma(wl=self.inst.wl_bins, wl_bin_edges=self.inst.wl_bin_edges)
        
        mult_factor = (options.t_tot * self.inst.telescope_area *
                       self.inst.wl_bin_widths *  self.inst.eff_tot)
        
        
        S_p = trans_eff_p * Fp * mult_factor    
        
        N_bg_norm = 2 * (N_sl + N_lz + N_ez)
        
        N_comb_norm = N_bg_norm + trans_eff_p * Fp
        
        N_comb = N_comb_norm * mult_factor
        
        snr = S_p/ np.sqrt(N_comb)
        snr_tot = np.sqrt((snr**2).sum())
        self.planet.snr_predicted = snr_tot
        print(f"--> Predicted SNR: {snr_tot:.2f}         \n")
    
        #if save_snr_spectrum:
        #    np.savetxt(f"data/spectral_snr_ET_D{self.inst.D*10:.0f}_R{options.R:.0f}const_t{options.t_tot/3600:.0f}h_bl10_z05.txt",
        #               np.array([self.inst.wl_bins, Fp, snr]).T)
        
        if plot: 
            print(f"--> Predicted SNR: {snr_tot:.2f}         \n")
            plotter.plot_fluxes_and_snr(self.planet, self.star, self.lz, self.ez, self.inst, S_p, N_comb,
                                        N_sl, N_lz, N_ez, self.options)
    
        return Fp, snr
            
     

