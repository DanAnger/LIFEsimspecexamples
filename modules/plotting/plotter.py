import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from matplotlib.colors import LogNorm


import seaborn as sns
import matplotlib as mpl
plt.style.use('seaborn-colorblind')

#sns.set_context("paper")
sns.set_style("white")
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['axes.grid'] = False


mpl.rcParams['xtick.major.size'] = 3.5
mpl.rcParams['ytick.major.size'] = 3.5
mpl.rcParams['axes.labelcolor'] = "0.0"
mpl.rcParams['axes.labelsize'] = "medium"
mpl.rcParams['figure.dpi'] = 300.
mpl.rcParams['axes.edgecolor'] = '.0'
mpl.rcParams['axes.linewidth'] = 0.5
pad=0.01



   
def plot_fluxes_and_snr(planet, star, localzodi, exozodi,
                        instrument, signal_p, N_comb,
                        stellar_leak, localzodi_leak, exozodi_leak, options):
    
    wl_bins = instrument.wl_bins
    Fp = planet.fgamma(wl_bins, wl_bin_edges=instrument.wl_bin_edges)
    Fs = star.fgamma(wl_bins)
    Fez = exozodi.ph_flux(
            wl_bins).sum(axis=(1,2))
    Flz = localzodi.fgamma_sr(wl_bins, lat=star.lat,
                                           ) * (np.pi * instrument.hfov**2)
    
    std = np.sqrt(N_comb)
    snr = signal_p / std
 
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(6.4,12.8),dpi=200)
    ax1.plot(wl_bins*1e6, Fp, color="r", label="Planet")
    ax1.plot(wl_bins*1e6, Fs, color="black", label="Star")
    #ez_label = f"Exozodi [{planet.zodis:.0f} z]"
    ez_label = f"Exozodi"
    ax1.plot(wl_bins*1e6, Fez, color="gray",
             label=ez_label)
    ax1.plot(wl_bins*1e6, Flz, color="darkblue", label="Local zodi")
             
    ax1.plot(wl_bins*1e6, stellar_leak, color="black",linestyle=":")
    ax1.plot(wl_bins*1e6, exozodi_leak, color="gray",linestyle=":")
    ax1.plot(wl_bins*1e6, localzodi_leak, color="darkblue",linestyle=":")

    ax1.set_xlim(instrument.wl_min, instrument.wl_max)
    ax1.set_xlabel(r"$\lambda$ [$\mu$m]", fontsize=12)
    ax1.set_ylabel(r"Input signal [ph s$^{-1}$m$^{-2}\mu$m$^{-1}$]", fontsize=12)
    ax1.set_ylim(1e-1 * Fp.min(), 1e1 * Fs.max())
    ax1.legend(loc='upper right', framealpha=1)
    ax1.set_yscale('log')
    ax1.grid()

    ax2.step(wl_bins*1e6, signal_p / options.t_tot,
             where ="mid", color="r", label=f"Planet")
    ax2.step(wl_bins*1e6 , std  / options.t_tot,
             where ="mid", color="black", label=f"Shot noise")
    ax2.set_xlabel(r"$\lambda$ [$\mu$m]", fontsize=12)
    ax2.set_ylabel(r"Detected signal per bin [e$^-$ s$^{-1}$ bin$^{-1}$]", fontsize=12)

    ax2.set_xlim(instrument.wl_min, instrument.wl_max)
    ax2.set_yscale('log')
    ax2.grid()

    
    ax2a =ax2.twinx()
    ax2a.set_yscale('log')
    #snr_label = f"SNR per bin \nTotal: {np.sqrt((snr**2).sum()):.2f}
    snr_label = f"SNR per bin"
    
    ax2a.step(wl_bins*1e6 , snr,
              where ="mid", color="darkblue", linestyle="--", label=snr_label)
    ax2a.set_ylabel('SNR per bin')
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2a.get_legend_handles_labels()
    ax2a.legend(lines + lines2, labels + labels2, framealpha=1)
    ax2a.grid(False)
    
    #TODO: ghetto fix
    ax2.set_ylim(top=(std  / options.t_tot).max() *10)
    #ax2a.set_ylim(1e-5, 1e0)
    
    

    plt.savefig("plots/input_signal_and_snr_prediction.pdf", bbox_inches='tight')
    plt.show()

   
def plot_spec_with_wo(Fp_with, Fp_wo, snr_with, inst, species=None, title=None):
    
    f_s=17
    
    fig, axes = plt.subplots(2,1, figsize=(8,10), constrained_layout=True)
    
    rand_noise = np.random.normal(0,Fp_with/snr_with,size = Fp_with.shape)
    
    ax=axes[0]
    ax.errorbar(inst.wl_bins*1e6, Fp_with, capsize=3, marker=".",
                    color = "cornflowerblue", ecolor="cornflowerblue", linestyle="-",
                    elinewidth = 1, alpha=1, label="with "+species)

    ax.fill_between(inst.wl_bins*1e6, Fp_with-(Fp_with/snr_with), Fp_with+(Fp_with/snr_with), alpha=0.15, label="1 $\sigma$")

    ax.errorbar(inst.wl_bins*1e6, Fp_with+rand_noise,yerr=Fp_with/snr_with, capsize=3, marker=".",
                    color = "grey", ecolor="grey", linestyle="none",
                    elinewidth = 1, alpha=1, label="sim. obs.")



# 0 is the mean of the normal distribution you are choosing from
# 1 is the standard deviation of the normal distribution
# 100 is the number of elements you get in array noise                

            
    
    ax.plot(inst.wl_bins*1e6, Fp_wo, color = "firebrick", linestyle="-", label="wo "+species)
                    
    ax.set_xlabel(r"$\lambda$ [$\mu$m]", fontsize=f_s)
    ax.set_ylabel(r"Planet flux $F_\lambda$ [ph $\mathrm{s}^{-1}$m$^{-2}\mu \mathrm{m}^{-1}$]", fontsize=f_s)
    ax.set_xlim(inst.wl_min, inst.wl_max)    
    #ax.set_xlim(8.,11.5)
    ax.legend(fontsize=(f_s-4))
    ax.tick_params(axis="x", labelsize=f_s-4)
    ax.tick_params(axis="y", labelsize=f_s-4)
    ax.grid()
    
    ax=axes[1]
    ax.step(inst.wl_bins*1e6, np.abs((Fp_wo-Fp_with)/(Fp_with/snr_with)),
                    where="mid", label="Diff "+species, color="forestgreen")
    
    ax.set_xlabel(r"$\lambda$ [$\mu$m]", fontsize=f_s)
    ax.set_ylabel(r"($F_{wo}-F_{with})/\sigma$", fontsize=f_s)
    ax.set_xlim(inst.wl_min, inst.wl_max) 
    #ax.set_xlim(8.,11.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=(f_s-4))
    ax.tick_params(axis="x", labelsize=f_s-4)
    ax.tick_params(axis="y", labelsize=f_s-4)
    ax.grid()
    
    fig.suptitle(title,fontsize=f_s)
    
        
    

def plot_planet_SED_and_SNR(wl_bins, Fp, Fp_est,
                            var, options, Fp_BB=None, snr_photon_stat=None):
      
    fig, ax = plt.subplots()
    
    sigma = np.sqrt(var)
    '''
    SNR= Fp_est / sigma
    ax2=ax.twinx()
    
    ax2.step(wl_bins*1e6, SNR, 
             label="SNR per bin", where="mid", color = "black", linestyle=":")
    ax2.set_ylabel(r"SNR per bin")
    ax2.set_ylim(bottom=0)
    '''
    
    ax.plot(wl_bins * 1e6, Fp, color="mediumblue", linestyle="-", label="True spectrum")
    
    if snr_photon_stat is not None:
        ax.fill_between(wl_bins * 1e6, Fp * (1-1/snr_photon_stat), Fp * (1+1/snr_photon_stat),
                        color="mediumblue", alpha=0.1, label=r"Photon noise $\sigma$")
    
    if Fp_BB is not None:
        ax.plot(wl_bins * 1e6, Fp_BB, color="green", linestyle="-", label="Fit BB Spectrum")
        
    ax.plot(wl_bins * 1e6, Fp_est, color="red", marker=".", linestyle="")
    
    with np.errstate(divide='ignore'):
        ax.errorbar(wl_bins * 1e6, Fp_est, yerr = sigma, capsize=3, marker=".",
                color = "red", ecolor="red", linestyle="none",
                elinewidth = 1, alpha=0.4,
                label="Estimated spectrum")
             
    ax.set_xlabel(r"$\lambda$ [$\mu$m]", fontsize=12)
    ax.set_ylabel(r"Planet flux $F_\lambda$ [ph $\mathrm{s}^{-1}$m$^{-2}\mu \mathrm{m}^{-1}$]", fontsize=12)
    ax.set_xlim(options.wl_min, options.wl_max)    
    ax.set_ylim(0, 1.6 * np.max(Fp))
    ax.grid()

    #lines, labels = ax.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    #ax.legend(lines + lines2, labels + labels2)
    ax.legend(fontsize=10)
    plt.savefig("plots/analysis/flux_extraction.pdf", bbox_inches='tight')
    plt.show()


