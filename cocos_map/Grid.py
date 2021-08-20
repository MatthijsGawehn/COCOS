# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:33:48 2020

@author: gawehn
"""

import numpy as np
import time
from joblib      import Parallel, delayed

class Grid:
    
    def __init__(self, Video, opts):     
        print('initialize grid...')
        # intialize grid for given Video domain   
        if opts.grid_dx is not None: # if grid spacing (in [m]) is imposed use that value
            gc_shift = int(round(opts.grid_dx/Video.dx))
        else:
            if opts.CPU_speed == 'fast': # if no grid spacing is given, base the spacing on computational speed
                discretization = 20
            elif opts.CPU_speed == 'normal':
                discretization = 30
            elif opts.CPU_speed == 'slow':
                discretization = 45   
            elif opts.CPU_speed == 'accurate':
                discretization = 60    
            elif opts.CPU_speed == 'exact':
                discretization = 500 
            self.Npx_min    = int(np.max([opts.Npx_fftmin, round(opts.Nm_fftmin/Video.dx) + round(opts.Nm_fftmin/Video.dx) % 2]))
            gc_shift        = round(np.mean([(Video.m - self.Npx_min)/discretization,(Video.n - self.Npx_min)/discretization]))#.astype(int)#number of pixles between cells
        if gc_shift < 2: 
            gc_shift = 2 
        self.dx         = gc_shift*Video.dx
        Rows_cor        = np.arange(0,(Video.m - self.Npx_min) + 1, gc_shift) 
        Cols_cor        = np.arange(0,(Video.n - self.Npx_min) + 1, gc_shift) 
        self.Rows_ctr   = (Rows_cor + self.Npx_min/2).astype(int)
        self.Cols_ctr   = (Cols_cor + self.Npx_min/2).astype(int)  
        x               = opts.Origin[0] + self.Cols_ctr*Video.dx
        y               = opts.Origin[1] + self.Rows_ctr*Video.dx 
        [self.X,self.Y] = np.meshgrid(x,y)  
        
        print('   grid cell spacing = {} m'.format(self.dx))        
    
    def get_gcSizes(self, Video, opts, Omegas, mask):
        def get_gcSize_perOmega(self, opts, Omegas, dx, gc_OffWaveLengths = 2):
            print('   get subdomain size per mode frequency for', gc_OffWaveLengths, 'offshore wave length(s)...')
            gc_size_perOmega    = []
            for omega in Omegas:
                T   = 2*np.pi/omega
                L   = 1.56*T**2*gc_OffWaveLengths #determines gc_size
                Npx = int(L/dx) + int(L/dx) % 2
                if gc_OffWaveLengths*opts.minPpOffWaveLength > Npx:
                    Npx = 0
                elif Npx < self.Npx_min:                               
                    Npx = self.Npx_min
                gc_size_perOmega.append(int(Npx))
            
            return np.array(gc_size_perOmega)  
        
        def fix_boundaries(self, Video, gc_sz_w, gc_sz_w_min):
            print('   fix subdomain sizes near frame boundaries...', end =" ") 
            start = time.time()
            #prealloc
            # get matrices of vertical and horizontal gc-binsizes
            #vetical(rows)
            self.gc_sz_v    = np.ones([self.Numrows, self.Numcols, self.Nw], dtype = int) * np.array(gc_sz_w)[None,None,:]
            #horizontal (cols)
            self.gc_sz_h    = np.ones([self.Numrows, self.Numcols, self.Nw], dtype = int) * np.array(gc_sz_w)[None,None,:]     
                   
            for ii in range(self.Nw): 
                if gc_sz_w[ii] == 0:
                    break
                #vertical
                    #upper fix
                for gcr in range(self.Numrows):
                    if (self.Rows_ctr[gcr] - int(gc_sz_w[ii]/2)) < 0: #check if gc size of omega[ii] fits for x (e.g., 2) wavelengths
                        if (self.Rows_ctr[gcr] - int(gc_sz_w_min[ii]/2)) < 0: #check if gc size of omega[ii] fits for minimum of 1 wavelength
                            self.gc_sz_v[gcr,:,ii]   = 0
                        else:
                            self.gc_sz_v[gcr,:,ii]   = self.Rows_ctr[gcr]*2
                    #lower fix        
                    if (self.Rows_ctr[gcr] + int(gc_sz_w[ii]/2)) > Video.m: #check if gc size of omega[ii] fits for x (e.g., 2) wavelengths
                        if ((self.Rows_ctr[gcr] + int(gc_sz_w_min[ii]/2)) > Video.m) | np.all(self.gc_sz_v[gcr,:,ii]  == 0): #check if gc size of omega[ii] fits for minimum of 1 wavelength
                            self.gc_sz_v[gcr,:,ii]   = 0
                        else:
                            #in case cell size is big compared to domain size
                            #if cell size has been adjusted for left boundary, but right boundary requires even smaller cell size
                            #(if cell is too small it would already be set to 0 in if statement above)
                            if np.any((Video.m-self.Rows_ctr[gcr])*2 <= self.gc_sz_v[gcr,:,ii]): 
                                self.gc_sz_v[gcr,:,ii]   = (Video.m-self.Rows_ctr[gcr])*2
                            else:  
                                # otherwise keep left boundary adjustment 
                                continue
                for gcc in range(self.Numcols):            
                    if (self.Cols_ctr[gcc] - int(gc_sz_w[ii]/2)) < 0: #check if gc size of omega[ii] fits for x (e.g., 2) wavelengths
                        if (self.Cols_ctr[gcc] - int(gc_sz_w_min[ii]/2)) < 0: #check if gc size of omega[ii] fits for minimum of 1 wavelength
                            self.gc_sz_h[:,gcc,ii]   = 0
                        else:
                            self.gc_sz_h[:,gcc,ii]   = self.Cols_ctr[gcc]*2     
                    #right fix        
                    if (self.Cols_ctr[gcc] + int(gc_sz_w[ii]/2)) > Video.n: #check if gc size of omega[ii] fits for x (e.g., 2) wavelengths
                        if ((self.Cols_ctr[gcc] + int(gc_sz_w_min[ii]/2)) > Video.n) | np.all(self.gc_sz_h[:,gcc,ii]  == 0): #check if gc size of omega[ii] fits for minimum of 1 wavelength
                            self.gc_sz_h[:,gcc,ii]   = 0
                        else:
                            if np.any((Video.n-self.Cols_ctr[gcc])*2 <= self.gc_sz_h[:,gcc,ii]):    
                                self.gc_sz_h[:,gcc,ii]   = (Video.n-self.Cols_ctr[gcc])*2   
                            else:
                                continue                        
            
            zeros_IX = (self.gc_sz_v == 0) | (self.gc_sz_h == 0)
            self.gc_sz_v[zeros_IX] = 0
            self.gc_sz_h[zeros_IX] = 0        
            end = time.time()
            print('CPU time: {} s'.format(np.round((end-start)*100)/100))
            
        def fix_mask_and_sampling(self, Video, opts, Omegas, mask, gc_sz_w_min):  
            print('   fix subdomain sizes near (masked) areas with no data and \n   determine optimal subdomain resolution...', end =" ") 
            start = time.time()
            #intitialize
            self.smpstep        = np.zeros([self.Numrows, self.Numcols, self.Nw], dtype = 'float16')  
            self.OffWaveLengths = np.zeros([self.Numrows, self.Numcols, self.Nw], dtype = float)  
            
            Ngc = self.Numrows*self.Numcols                         
            for gc in range(Ngc): 
                gcc = (np.floor(gc/self.Numrows)).astype(int)
                gcr = (gc-(gcc*self.Numrows)).astype(int)  
                self.gc_sz_v[gcr,gcc,:], self.gc_sz_h[gcr,gcc,:], self.smpstep[gcr,gcc,:], self.OffWaveLengths[gcr,gcc,:] = self.gc_walk_fix_mask_and_sampling(gc, Video, opts, Omegas, mask, gc_sz_w_min)
            end = time.time()
            print('CPU time: {} s'.format(np.round((end-start)*100)/100))    
            
        print('get subdomain size and resolution per mode...') 
        start = time.time()        
        gc_sz_w     = get_gcSize_perOmega(self, opts, Omegas, Video.dx, gc_OffWaveLengths = opts.gc_OffWaveLengths) #important for accuracy!! --> bad for speed
        T           = np.round(1/(Omegas/(2*np.pi))*10)/10 
        print('      desired gc sizes for wave-periods {}-{} s: {}'.format(np.min(T),np.max(T),np.flipud(gc_sz_w)))
        gc_sz_w_min = get_gcSize_perOmega(self, opts, Omegas, Video.dx, gc_OffWaveLengths = 1)        

        self.Nw         = len(Omegas)
        self.Numrows    = len(self.Rows_ctr)
        self.Numcols    = len(self.Cols_ctr)
        
        fix_boundaries(self, Video, gc_sz_w, gc_sz_w_min) 
        fix_mask_and_sampling(self, Video, opts, Omegas, mask, gc_sz_w_min)
        self.mask       = np.all(self.gc_sz_h == 0, axis = -1) | np.all(self.gc_sz_v == 0, axis = -1)
        end = time.time()
        print('CPU time: {} s'.format(np.round((end-start)*100)/100))
        
    def get_gc_OffWaveLengths(self,omega, dx, Npx):        
        T   = 2*np.pi/omega
        L   = 1.56*T**2
        gc_OffWaveLengths = Npx*dx/L #+ = int(L/dx) + int(L/dx) % 2        
        return gc_OffWaveLengths    
                 
    def gc_walk_fix_mask_and_sampling(self, gc, Video, opts, Omegas, mask, gc_sz_w_min):
        gcc         = (np.floor(gc/self.Numrows)).astype(int)
        gcr         = (gc-(gcc*self.Numrows)).astype(int)              
        gc_sz_h_gc          = self.gc_sz_h[gcr,gcc,:]
        gc_sz_v_gc          = self.gc_sz_v[gcr,gcc,:]
        smpstep_gc          = self.smpstep[gcr,gcc,:]
        gc_OffWaveLengths   = self.OffWaveLengths[gcr,gcc,:]
        for ii in range(self.Nw):
            # --------
            # fix mask
            # --------
            # stop in case of boundary 0s
            if (gc_sz_h_gc[ii] == 0) | (gc_sz_v_gc[ii] == 0):
                continue
            # else get submask
            submask     = mask[self.Rows_ctr[gcr]-int(gc_sz_v_gc[ii]/2):self.Rows_ctr[gcr]+int(gc_sz_v_gc[ii]/2), 
                               self.Cols_ctr[gcc]-int(gc_sz_h_gc[ii]/2):self.Cols_ctr[gcc]+int(gc_sz_h_gc[ii]/2)]
            # if entire submask is True (bad area) set cell size 0
            if np.all(submask): 
                gc_sz_v_gc[ii:] = 0 #when triggered also true for next ii --> so break                        
                gc_sz_h_gc[ii:] = 0 #when triggered also true for next ii --> so break
                break
            # else check in increments how to adjust the cell size in order to have less than "nodatathr" of the cell filled with 0s
            nodata      = np.sum(submask)  
            nodatathr   = 0.25 #25% of cells are bad (e.g. due to landmask)
            while (nodata > nodatathr*gc_sz_h_gc[ii]*gc_sz_v_gc[ii]): #| np.isnan(dm_gc.ravel())| (np.nanstd(np.real(dm_gc.ravel())) < 0.01):                            
                # shrink cell size
                tmp1        = int(gc_sz_h_gc[ii]*(1-nodatathr))
                tmp2        = int(gc_sz_v_gc[ii]*(1-nodatathr))
                # make cell size even
                tmp1        = tmp1 - tmp1 % 2
                tmp2        = tmp2 - tmp2 % 2
                # get submasks
                submask1    = mask[self.Rows_ctr[gcr]-int(gc_sz_v_gc[ii]/2):self.Rows_ctr[gcr]+int(gc_sz_v_gc[ii]/2), 
                               self.Cols_ctr[gcc]-int(tmp1/2):self.Cols_ctr[gcc]+int(tmp1/2)]
                submask2    = mask[self.Rows_ctr[gcr]-int(tmp2/2):self.Rows_ctr[gcr]+int(tmp2/2), 
                               self.Cols_ctr[gcc]-int(gc_sz_h_gc[ii]/2):self.Cols_ctr[gcc]+int(gc_sz_h_gc[ii]/2)]
              
                nodata1     = np.sum(submask1)
                nodata2     = np.sum(submask2)               
                if nodata1 < nodata2:
                    gc_sz_h_gc[ii] = tmp1
                    nodata  = nodata1
                elif nodata2 < nodata1:
                    gc_sz_v_gc[ii] = tmp2
                    nodata  = nodata2
                else:
                    # shrink cell size
                    gc_sz_h_gc[ii]  = int(gc_sz_h_gc[ii]*(1-nodatathr/2)) 
                    gc_sz_v_gc[ii]  = int(gc_sz_v_gc[ii]*(1-nodatathr/2))
                    # make cell size even
                    gc_sz_h_gc[ii]  = gc_sz_h_gc[ii] - gc_sz_h_gc[ii] % 2
                    gc_sz_v_gc[ii]  = gc_sz_v_gc[ii] - gc_sz_v_gc[ii] % 2
                    # get cell mask                            
                    submask     = mask[self.Rows_ctr[gcr]-int(gc_sz_v_gc[ii]/2):self.Rows_ctr[gcr]+int(gc_sz_v_gc[ii]/2), 
                               self.Cols_ctr[gcc]-int(gc_sz_h_gc[ii]/2):self.Cols_ctr[gcc]+int(gc_sz_h_gc[ii]/2)]
                    # update how many pixels are bad
                    nodata          = np.sum(submask)
                # if cell needs to shrink beyond the mimimum cell size, set to 0    
                if (gc_sz_h_gc[ii] < gc_sz_w_min[ii]) | (gc_sz_v_gc[ii] < gc_sz_w_min[ii]):
                    gc_sz_h_gc[ii]  = 0
                    gc_sz_v_gc[ii]  = 0
                    break    
            # stop in case of too small grid cell 0s
            if (gc_sz_h_gc[ii] == 0) | (gc_sz_v_gc[ii] == 0):
                continue          
            # ------------------------------------
            # fix sampling resolution in subdomain
            # ------------------------------------          
            gc_sz_wii           = np.array([gc_sz_v_gc[ii], gc_sz_h_gc[ii]]) 
            gc_sz_wii_min       = min(gc_sz_wii)
            # calc how many offshore wavelengths fit into subdomain of omega
            gc_OffWaveLengths_prior   = self.get_gc_OffWaveLengths(Omegas[ii], Video.dx, gc_sz_wii_min)
            if gc_OffWaveLengths_prior*opts.minPpOffWaveLength > gc_sz_wii_min:
                # if less than opts.minPpOffWaveLength (default 8) poits per offshore wave length, set cell size to 0 (i.e. guarranteed too coarse resolution for omega. conservative, since wavelength is mostly smaller than offshore wavelength)
                gc_sz_v_gc[ii:]         = 0
                gc_sz_h_gc[ii:]         = 0
                gc_OffWaveLengths[ii:]  = 0 #redundant
                break
            else:                
                #calculate minimum number of points needed to represent offshore waves in cell
                minsamples  = int(gc_OffWaveLengths_prior*opts.minPpOffWaveLength)
                minsamples  = minsamples + minsamples % 2 
                #if smaller than threshold for minimum # of samples (opts.Npx_fftmin) --> set to opts.Npx_fftmin
                if minsamples < opts.Npx_fftmin:
                    minsamples = opts.Npx_fftmin
                mult_ms     = gc_sz_wii_min/minsamples     
                if mult_ms > 1: 
                    # if more than needed sample points --> subsample
                    smpstep_gc[ii]  = mult_ms 
                    bigger_IX       = gc_sz_wii/mult_ms % 2 != 0
                    while np.any(bigger_IX):
                        gc_sz_wii[bigger_IX]    = gc_sz_wii[bigger_IX]-2
                        bigger_IX               = gc_sz_wii/mult_ms % 2 != 0                        
                else:  
                    smpstep_gc[ii]  = 1
                    gc_sz_wii[:]    = gc_sz_wii_min
                    
                gc_sz_v_gc[ii]  = gc_sz_wii[0]  
                gc_sz_h_gc[ii]  = gc_sz_wii[1] 
                #or force h and v same size
# =============================================================================
#                 gc_sz_v_gc[ii] = gc_sz_wii_min
#                 gc_sz_h_gc[ii] = gc_sz_wii_min
# =============================================================================
            # calc how many offshore wavelengths fit into cell of omega
            gc_OffWaveLengths[ii]  = self.get_gc_OffWaveLengths(Omegas[ii], Video.dx, gc_sz_wii_min)  
        return gc_sz_v_gc, gc_sz_h_gc, smpstep_gc, gc_OffWaveLengths
    