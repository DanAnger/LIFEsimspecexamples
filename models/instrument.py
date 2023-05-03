import numpy as np

from modules import transmission_generator as tm_gen


class Instrument:
    def __init__(self,
                 options,
                 D=2,
                 quantum_eff=0.7,
                 throughput=0.05,
                 wl_min=3.,
                 wl_max=20.,
                 R=20,
                 bl=20,
                 bl_min=10,
                 bl_max=100):
        
        self.D = float(D)
        self.quantum_eff = float(quantum_eff)
        self.throughput = float(throughput)
        self.wl_min = float(wl_min)
        self.wl_max = float(wl_max)
        self.R = float(R)
        self.bl = float(bl)
        self.bl_min = float(bl_min)
        self.bl_max = float(bl_max)
        
        self.telescope_area = np.pi * (self.D/2.)**2 * 4.
        self.eff_tot = self.quantum_eff *  self.throughput
        
        self.wl_bins, self.wl_bin_widths, self.wl_bin_edges = self.get_wl_bins_const_R()
        
        self.hfov =  self.wl_bins / (2. * self.D) # fov = wl / D -> hfov=wl/(2*D)
        self.hfov_mas =  self.hfov * (3600000. * 180.) / np.pi
        self.rpp = (2 * self.hfov) / options.image_size  # Radians per pixel
        self.maspp = (2 * self.hfov_mas) / options.image_size  # mas per pixel

        # apertures defines the telescope positions (and *relative* radius)
        self.apertures = np.array([[-self.bl/2, -6*self.bl/2., 1.],
                                   [ self.bl/2, -6*self.bl/2., 1.],
                                   [ self.bl/2,  6*self.bl/2., 1.],
                                   [-self.bl/2,  6*self.bl/2., 1.]])
        

    def get_wl_bins_const_R(self):
        wl_edge = self.wl_min
        wl_bins = []
        wl_bin_widths = []
        wl_bin_edges = [wl_edge]
        
        while wl_edge < self.wl_max:
            wl_bin_width = wl_edge / self.R / (1-1/self.R/2)
            
            if wl_edge + wl_bin_width > self.wl_max:
                wl_bin_width = self.wl_max - wl_edge

            wl_center = wl_edge + wl_bin_width/2
            wl_edge += wl_bin_width
            
            wl_bins.append(wl_center)
            wl_bin_widths.append(wl_bin_width)
            wl_bin_edges.append(wl_edge)
    
        wl_bins = np.array(wl_bins) * 1e-6 #in m
        wl_bin_widths = np.array(wl_bin_widths) #in microns
        wl_bin_edges = np.array(wl_bin_edges) * 1e-6 #in m
        
        return wl_bins, wl_bin_widths, wl_bin_edges
        

    def adjust_bl_to_HZ(self,
                        star,
                        options):
        
        HZcenter_rad = star.HZcenter / star.Ds / (3600 * 180) * np.pi # in rad
        
        # put first transmission peak of optimal wl on center of HZ
        self.bl = 0.589645 / HZcenter_rad * options.wl_optimal 
        
        self.bl = np.maximum(self.bl, self.bl_min)
        self.bl = np.minimum(self.bl, self.bl_max)

        self.apertures = np.array([[-self.bl/2, -6*self.bl/2., 1.],
                                   [ self.bl/2, -6*self.bl/2., 1.],
                                   [ self.bl/2,  6*self.bl/2., 1.],
                                   [-self.bl/2,  6*self.bl/2., 1.]])

        
    def add_transmission_maps(self, options, map_selection=None):
        self.tms1, self.tms2, self.tms3, self.tms4, self.tms_chop = tm_gen.tms_fast(
                                                                        options.image_size,
                                                                        self.hfov_mas,
                                                                        self.bl,
                                                                        self.wl_bins,
                                                                        map_selection)
   
    
    def get_stellar_leakage(self, star, options, image_size=50, tm="3"):
        
        Rs_au = 0.00465047 * star.Rs
        Rs_mas = Rs_au/ star.Ds * 1000
        Rs_mas = float(Rs_mas)
        
        if tm == "3":
            tm_star = tm_gen.tms_fast(image_size, Rs_mas, self.bl,
                                         self.wl_bins)[2]
        
        elif tm == "4":
            tm_star = tm_gen.tms_fast(image_size, Rs_mas, self.bl,
                                         self.wl_bins)[3]
        
        x_map = np.tile(np.array(range(0, image_size)),(image_size,1))
        y_map = x_map.T
        r_square_map = (x_map - (image_size-1) / 2)**2+ (y_map - (image_size-1) / 2)**2
        
        star_px = np.where(r_square_map<(image_size/2)**2,1,0)

        sl = (star_px* tm_star).sum(axis=(-2,-1)) / star_px.sum() * star.fgamma(self.wl_bins)
        
        return sl
    
    
    def get_exozodi_leakage(self, exozodi, options, tm="3"):
        
        if tm == "3":
            tm = self.tms3
        elif tm =="4":
            tm = self.tms4
            
        ap = np.where(options.r_map <= options.image_size/2, 1, 0)
        ez_flux =  exozodi.ph_flux(self.wl_bins)
        ez_leak = (ez_flux * tm * ap).sum(axis=(-2,-1))
        
        return ez_leak

    def get_localzodi_leakage(self, localzodi, star, options, tm="3"):
        
        long = 3/4 * np.pi  
        lat  = star.lat

        if tm == "3":
            tm = self.tms3
        elif tm =="4":
            tm = self.tms4
            
        ap = np.where(options.r_map <= options.image_size/2, 1, 0)
        lz_flux_sr = localzodi.fgamma_sr(self.wl_bins,
                                         long=long,
                                         lat=lat)
        
        lz_flux = lz_flux_sr * (np.pi * self.hfov**2)
        
        lz_leak = (ap * tm).sum(axis=(-2,-1)) / ap.sum() *lz_flux
        return lz_leak
    

    