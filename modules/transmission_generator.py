import numpy as np


""" 
what follows is a simplified version to calculate the transmission
"""

def tms_fast(image_size, hfov_mas, small_baseline, wl, map_selection=None):
    """
    Parameters
    ----------
    image_size: float
        side length of image in pixels.
    hfov_mas: float
        half-field-of-view in milliarcsec.
    small_baseline: float
        small base line of X-telescope array in m.
    wl: float
        wavelength in microns.
    
    Returns
    -------
    
    tm1, tm2, tm3, tm4: np.array, 2D
        transmission map modes 1 to 4
    chopped_tm: np.array, 2D
        chopped transmission map
        
        
    """
    wl = np.array([wl])  # wavelength in m 
    
    if wl.shape[-1] >1:
        wl = np.reshape(wl,(wl.shape[-1],1,1)) 
        
    hfov_mas = np.array([hfov_mas])  # wavelength in m 
    
    if hfov_mas.shape[-1] >1:
        hfov_mas = np.reshape(hfov_mas,(hfov_mas.shape[-1],1,1))
    
    hfov = hfov_mas/(1000*3600*180)*np.pi #hfov in radians
    angle = np.linspace(-1, 1, image_size) #generare 1D array that spans field of view
    alpha = np.tile(angle,(image_size,1)) # angle matrix in x-direction ("alpha")
    beta = alpha.T # angle matrix in y-direction ("beta")
    alpha = alpha * hfov
    beta = beta * hfov
    L= small_baseline / 2 # smaller distance of telecscopes from center line
    
    
    if map_selection=="tm3_only": #if only interested in tm3 for SNR scan
        tm3 = np.sin(2*np.pi*L*alpha/wl)**2 * np.cos(12*np.pi*L*beta/wl - np.pi/4)**2 # transmission map of mode 3
        return None, None, tm3, None, None
    
    else:
        tm1 = np.cos(2*np.pi*L*alpha/wl)**2 * np.cos(12*np.pi*L*beta/wl - np.pi/4)**2 # transmission map of mode 1
        tm2 = np.cos(2*np.pi*L*alpha/wl)**2 * np.cos(12*np.pi*L*beta/wl + np.pi/4)**2 # transmission map of mode 2
        tm3 = np.sin(2*np.pi*L*alpha/wl)**2 * np.cos(12*np.pi*L*beta/wl - np.pi/4)**2 # transmission map of mode 3
        tm4 = np.sin(2*np.pi*L*alpha/wl)**2 * np.cos(12*np.pi*L*beta/wl + np.pi/4)**2 # transmission map of mode 4
        tm_chop = tm3-tm4 # difference of transmission maps 3 and 4 = "chopped transmission"
        return tm1, tm2, tm3, tm4, tm_chop


def tms_chop_pol(image_size, hfov_mas, small_baseline, wl, phi_n = 360):
    """
    Parameters
    ----------
    image_size: float
        side length of image in pixels.
    hfov_mas: float
        half-field-of-view in milliarcsec.
    small_baseline: float
        small base line of X-telescope array in m.
    wl: float
        wavelength in microns.
    phi_n: float
        number of azimuthal slized, default=360
    
    Returns
    -------
    trans_map_polar: np.array, 2D
        chopped transmission map in polar coordinates
    """
    
    
    wl = np.array([wl]) # wavelength in m 
    
    if wl.shape[-1] >1:
        wl = np.reshape(wl, (wl.shape[-1],1,1))
        
    hfov_mas = np.array([hfov_mas]) # wavelength in m 
    
    if hfov_mas.shape[-1] > 1:
        hfov_mas = np.reshape(hfov_mas, (hfov_mas.shape[-1],1,1))
    
    hfov = hfov_mas / (1000*3600*180) * np.pi
    radial_ang_px = int(image_size/2)
    
    phi_lin = np.linspace(0, 2*np.pi, phi_n, endpoint=False) # 1D array with azimuthal coordinates
    phi_mat = np.tile(phi_lin, (radial_ang_px,1)) # 2D map with azimuthal coordinates

    
    theta_lin = np.linspace(0, 1, radial_ang_px, endpoint=False) # 1D array with radial separation coord [radians]
    theta_lin += 1 * 0.5 / radial_ang_px # 1D array with radial separation coordinates [radians]
    theta_mat = np.tile(theta_lin, (phi_n,1)).T # 2D array with radial separation coordinates [radians]
    theta_mat = theta_mat * hfov
    
    L= small_baseline / 2
    
    tm_chop_pol = np.sin(2*np.pi*L / wl * theta_mat * np.cos(phi_mat))**2 * np.sin(
        24*np.pi*L / wl * theta_mat * np.sin(phi_mat))
    
    return tm_chop_pol



def transm_curve(bl, wl, ang_sep_as, phi_n = 360):
    
    wl = np.array([wl])# wavelength in m 
    if wl.shape[-1] >1:
        wl = np.reshape(wl,(wl.shape[-1],1))
        
    ang_sep_rad = ang_sep_as/(3600*180)*np.pi
    phi_lin = np.linspace(0, 2*np.pi, phi_n, endpoint=False) # 1D array with azimuthal coordinates
    
    L= bl / 2
    
    transm_curve = np.sin(2*np.pi*L/wl * ang_sep_rad * np.cos(phi_lin))**2 * np.sin(
        24*np.pi*L / wl * ang_sep_rad * np.sin(phi_lin))
        
    return transm_curve



def transm_eff(bl, wl, ang_sep_as):
    
    tc = transm_curve(bl, wl, ang_sep_as)
    transm_eff = np.sqrt((tc**2).mean(axis=-1))
    
    return transm_eff
