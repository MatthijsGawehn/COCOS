# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:36:59 2020

@author: gawehn
"""
#external 
import numpy as np
from itertools import compress
# =============================================================================
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
# =============================================================================
import time
from joblib      import Parallel, delayed
from collections import namedtuple
from scipy.optimize import least_squares, minimize, Bounds #,curve_fit
from scipy import stats
from copy import copy
from scipy.signal import correlate #,unit_impulse, detrend, medfilt2d, convolve2d, gaussian
#from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale#,resize

#%%

class Inversion():
    
    def __init__(self):
        print('initialize inversion...')
        self.PIVcnt = 0
        self.FFTcnt = 0
        pass
    
    @staticmethod
    def get_storage(grid):
        print('initialize spectral storage...')
        class SSPC_storage():
            def __init__(self, grid): #, U_flag, 
                Ngc     = len(grid.Rows_ctr)*len(grid.Cols_ctr)
                
                self.kx_store       = [[] for _ in np.zeros(Ngc)]
                self.ky_store       = [[] for _ in np.zeros(Ngc)]
                self.omega_store    = [[] for _ in np.zeros(Ngc)]
                self.pWeights_store = [[] for _ in np.zeros(Ngc)] 
                self.time_store     = []

        storage = SSPC_storage(grid)   
        return storage
    
    def get_gridKernel(self, grid, rad = 100):

        def disc2D(dx, rad):
            discret     = int(round(rad*2/dx))
            discret     = discret + discret % 2 + 1 #make odd            
            rad         = ((discret-1)*dx)/2
            # make a disc mask 
            xy          = np.linspace(-rad,rad,discret)
            X,Y         = np.meshgrid(xy,xy)
            discmask    = np.ones((discret,discret)).astype(int)
            discmask[np.sqrt(X**2+Y**2) > rad] = 0        
            midpoint    = int(discret/2)
            discmask[midpoint,midpoint] = 0
            return discmask
        
        print('initialize sampling kernel...')
        gc_kernel       = disc2D(grid.dx, rad = rad)
# =============================================================================
#         p12             = np.hanning(gc_kernel.shape[0])    
#         p               = p12[:,None] @ p12[None,:]
# =============================================================================
        self.gc_kernel  = gc_kernel#*p
    
    def get_maps(self, Video, opts, dmd, grid, gc_walk_unwrap_self, storage, t):
        
        def get_storage_vals(gc, Ngc, Numrows, Numcols, kernel, grid_nanmask, kx_store, ky_store, pWeights_store, omegas_store):                
            gc_surround = []
            gc_kernel   = []
            gc_IDs      = np.where(kernel)
            sampnum     = len(gc_IDs[0])
            kernel_sz   = kernel.shape[0]
            #fix kernels at domain boundaries: check if disc kernel points are symetrically distributed within video domain boundaries
            if sampnum > 0:
                midpoint        = int((kernel_sz-1)/2)
                base_gcs_col    = (gc+(gc_IDs[1]-midpoint)*Numrows).astype(int)
                dev_rows        = (gc_IDs[0]-midpoint).astype(int)
                gcck            = (np.floor(base_gcs_col/Numrows)).astype(int)
                gcrk            = (base_gcs_col-(gcck*Numrows)).astype(int) + dev_rows 
                bad             = (gcck < 0) | (gcck >= Numcols) | (gcrk < 0) | (gcrk >= Numrows)
                for ii in range(sampnum):
                    if not bad[ii]:
                        bad[ii] = grid_nanmask[gcrk[ii],gcck[ii]]
                bad             = bad | bad[::-1] #erase points to make point symmetrical again
                
                if not np.all(bad):    
                    gcs         = base_gcs_col[~bad] + dev_rows[~bad]
                    good_IDs    = np.where(~bad)[0]
                    for ii in range(gcs.size):
                        gc_surround.append(gcs[ii])
                        gc_kernel.append(kernel[gc_IDs[0][good_IDs[ii]]][gc_IDs[1][good_IDs[ii]]])
            
            surround_val_kx         = [kx_store[number][ii] for number in gc_surround for ii in range(len(kx_store[number]))]
            midpoint_val_kx         = [kx_store[gc][ii] for ii in range(len(kx_store[gc]))]
            surround_val_ky         = [ky_store[number][ii] for number in gc_surround for ii in range(len(ky_store[number]))]
            midpoint_val_ky         = [ky_store[gc][ii] for ii in range(len(ky_store[gc]))]
            surround_val_pWeights   = [pWeights_store[number][ii] for number in gc_surround for ii in range(len(pWeights_store[number]))]
            midpoint_val_pWeights   = [pWeights_store[gc][ii] for ii in range(len(pWeights_store[gc]))]
            surround_val_omegas     = [omegas_store[number][ii] for number in gc_surround for ii in range(len(omegas_store[number]))]
            midpoint_val_omegas     = [omegas_store[gc][ii] for ii in range(len(omegas_store[gc]))]
            
            return [surround_val_kx, midpoint_val_kx],[surround_val_ky, midpoint_val_ky],[surround_val_pWeights, midpoint_val_pWeights],[surround_val_omegas, midpoint_val_omegas]   
        
        def randsampleKernel(kernel, samplenum):
            kernel_samp = np.zeros(kernel.shape)#.astype(int)
            gc_IDs      = np.where(kernel)
            gc_num      = len(gc_IDs[0])
            if samplenum > gc_num: # in case number of kernel points is less than desired number of sample points
                kernel_samp = kernel
            else:
                # pick random IDs from first quadrant and then point-symmetrically copy to the other quadrants
                sidelen     = kernel.shape[0]
                quadrow_sz  = int(np.ceil(sidelen/2))
                quadcol_sz  = int(np.floor(sidelen/2))
                gc_IDs      = np.where(kernel[0:quadrow_sz,0:quadcol_sz])
                gc_num      = len(gc_IDs[0])
                sample_IDs  = np.random.choice(gc_num, int(samplenum/4), replace=False)
                sidelen_ID  = sidelen - 1
                for samp in sample_IDs:
                    row_ID = gc_IDs[0][samp]
                    col_ID = gc_IDs[1][samp]
                    kernel_samp[0+row_ID,0+col_ID]                      = kernel[0+row_ID,0+col_ID] 
                    kernel_samp[sidelen_ID-col_ID,0+row_ID]             = kernel[sidelen_ID-col_ID,0+row_ID]
                    kernel_samp[sidelen_ID-row_ID,sidelen_ID-col_ID]    = kernel[sidelen_ID-row_ID,sidelen_ID-col_ID]
                    kernel_samp[0+col_ID,sidelen_ID-row_ID]             = kernel[0+col_ID,sidelen_ID-row_ID]
                    
            return kernel_samp
        
        # START THE ENTIRE THING
        # ----------------------
        print('invert d,u,v,cx,cy...') 
        starti = time.time()
        # get grid params
        Nw          = len(dmd.omega)
        Numrows     = len(grid.Rows_ctr)
        Numcols     = len(grid.Cols_ctr)
        Ngc         = Numrows*Numcols
        gc          = 0    
        # randsample grid cell kernel
        self.kernel_samp = randsampleKernel(self.gc_kernel, opts.gc_kernel_sampnum)
        # make cxy array based on opts.cxy_deltaT
        Tmin        = 1/max(opts.freqlims)
        Tmax        = 1/min(opts.freqlims)
        cxy_omegas  = 2*np.pi/np.linspace(Tmax, Tmin, int((Tmax-Tmin+1)/opts.cxy_deltaT))
        # prepare inversion
        if opts.fitter == 'SLSQP':
            # define inequality constraints for SLSQP
            ineq_cons   = {'type': 'ineq',
                         'fun' : lambda duv: np.array([opts.Ulim**2 - (duv[1]**2 + duv[2]**2)]),
                         'jac' : lambda duv: np.array([0,
                                                       -2*duv[1], 
                                                       -2*duv[2]])}           
            # define bounds for SLSQP
            bounds      = Bounds([min(opts.dlims), -opts.Ulim, -opts.Ulim ], [max(opts.dlims), opts.Ulim, opts.Ulim])            
            # check f_scale in relation to average difference between Dynamic Mode frequencies      
            domega  = np.mean(np.diff(dmd.omega))#*opts.domega_frac
            f_scale = opts.f_scale
            domega_penalty_thr      = np.sqrt(f_scale**2*10)
            print('   loss function: shoulder in cauchy derivative at about deriv = 1/10 and at about \n                  domega^2 = f_scale^2 * 10. For f_scale = {} this means \n                  strong penalties are expected for domega > {}, which is about 1/{} \n                  of average dw ({}) between Dyn. Modes'.format(f_scale,np.round(domega_penalty_thr*1000)/1000,np.round(domega/domega_penalty_thr*10)/10,np.round(domega*1000)/1000))
            # set fit methods with factory function
            fun, jac, conf_int_95 = self.set_fitfun(opts, f_scale)         
        else:
            ineq_cons   = None
            bounds      = None
            
        #preallocate
        duv         = np.empty((Ngc,3)).astype('float16')
        cxcy        = np.empty((Ngc,len(cxy_omegas),2)).astype('float16')
        kxky        = np.empty((Ngc,Nw,2)).astype('float32')
        omegas      = np.empty((Ngc,Nw)).astype('float16') 
        pWeights    = np.empty((Ngc,Nw)).astype('float16') 
        rmse        = np.empty(Ngc).astype('float16')        
        r2          = np.empty(Ngc).astype('float16')        
        conf95      = np.empty((Ngc,3)).astype('float16')  
        s2_w_cxcy   = np.empty((Ngc,len(cxy_omegas),2)).astype('float16')        
        
        if opts.parallel_flag:
            print('   invoke parallel computation for inversion...')#backend="loky"#require='sharedmem'#prefer = "threads"
            results  = Parallel(n_jobs= -1, backend="loky")\
                    (delayed(gc_walk_unwrap_self)((self, gc, Video.dx, opts, dmd, grid, Numrows, Numcols, Nw, ineq_cons, bounds, f_scale, fun, jac, conf_int_95, 
                                                get_storage_vals(gc, Ngc, Numrows, Numcols, self.kernel_samp, grid.mask, storage.kx_store, storage.ky_store, storage.pWeights_store, storage.omega_store),
                                                cxy_omegas))\
                                                for gc in range(Ngc))
      
            for gc in range(Ngc):
                duv[gc,:]           = results[gc][0]
                cxcy[gc,:,:]        = results[gc][1]
                kxky[gc,:,:]        = results[gc][2]
                omegas[gc,:]        = results[gc][3]   
                pWeights[gc,:]      = results[gc][4]
                rmse[gc]            = results[gc][5]
                r2[gc]              = results[gc][6]
                conf95[gc,:]        = results[gc][7]
                s2_w_cxcy[gc,:,:]   = results[gc][8]
        else:                        
            for gc in range(Ngc):  
                print('processing grid cells % {:}'.format(int(gc/Ngc*100)))
                duv[gc,:], cxcy[gc,:,:], kxky[gc,:,:], omegas[gc,:], pWeights[gc,:], rmse[gc], r2[gc], conf95[gc,:], s2_w_cxcy[gc,:,:] = self.gc_walk(gc, Video.dx, opts, dmd, grid, Numrows, Numcols, Nw, ineq_cons, bounds, f_scale, fun, jac, conf_int_95, 
                                                                                                                                                get_storage_vals(gc, Ngc, Numrows, Numcols, self.kernel_samp, grid.mask, storage.kx_store, storage.ky_store, storage.pWeights_store, storage.omega_store),
                                                                                                                                                cxy_omegas)#, self.ineq_cons, self.bounds)     
        endi  = time.time() 
        print('CPU time: {} s'.format(np.round((endi-starti)*100)/100))
        # store results
        print('gather results...') 
        ResultStruct    = namedtuple('struct', ['d', 'u', 'v', 'cx', 'cy', 'c_omega', 'omega', 'rmse', 'r2', 'conf95', 'var_cxcy'])
        Results         = ResultStruct(duv[:,0], -1*duv[:,1], -1*duv[:,2], -1*cxcy[:,:,0], -1*cxcy[:,:,1], cxy_omegas, omegas, rmse, r2, conf95, s2_w_cxcy)  
        # store sparse spectral point clouds (SSPCs) of current frame sequence for later updates
        print('store SSPCs of current sequence...', end =" ") 
        starti = time.time()
        isnan_IX        = np.isnan(omegas)
        storage.time_store.append(int(t))
        for gc in range(Ngc):            
            if ~np.all(isnan_IX[gc,:]):                
                storage.kx_store[gc].append((kxky[gc,~isnan_IX[gc,:],0]).astype('float32'))
                storage.ky_store[gc].append((kxky[gc,~isnan_IX[gc,:],1]).astype('float32'))
                storage.omega_store[gc].append((omegas[gc,~isnan_IX[gc,:]]).astype('float16'))
                storage.pWeights_store[gc].append((pWeights[gc,~isnan_IX[gc,:]]).astype('float16'))              
            else:
                storage.kx_store[gc].append([])
                storage.ky_store[gc].append([])
                storage.omega_store[gc].append([])
                storage.pWeights_store[gc].append([])
        endi = time.time()
        print('CPU time: {} s'.format(np.round((endi-starti)*100)/100))         
        # remove SSPCs in storage older than 'opts.stationary_time' seconds
        print('remove SSPCs older than stationary time = ',opts.stationary_time,'s from temporary storage...', end =" ") 
        starti = time.time()
        out_of_window_ID = [ii for ii,time in enumerate(storage.time_store) if ((storage.time_store[-1] - time) > opts.stationary_time)]
        if out_of_window_ID != []:
            for ii in sorted(out_of_window_ID, reverse=True):
                del(storage.time_store[ii])
                for gc in range(Ngc):                                     
                    del(storage.kx_store[gc][ii])
                    del(storage.ky_store[gc][ii])
                    del(storage.omega_store[gc][ii])
                    del(storage.pWeights_store[gc][ii])  
        endi = time.time()
        print('CPU time: {} s'.format(np.round((endi-starti)*100)/100))  
            
        return Results, storage
    
    def gc_walk(self, gc, dx, opts, dmd, grid, Numrows, Numcols, Nw, ineq_cons, bounds, f_scale, fun, jac, conf_int_95, store_vars_gc, cxy_omegas):
        
        #ger sparse spectral point clouds (SSPCs) from storage
        kx_store_gc         =  store_vars_gc[0]
        ky_store_gc         =  store_vars_gc[1]
        pWeights_store_gc   =  store_vars_gc[2]
        omegas_store_gc     =  store_vars_gc[3]
        
        #cast grid data into practical variables
        gcc         = (np.floor(gc/Numrows)).astype(int)
        gcr         = (gc-(gcc*Numrows)).astype(int)            
        y_px        = grid.Rows_ctr[gcr]
        x_px        = grid.Cols_ctr[gcc]
        bad         = False 
        gc_sz       = [grid.gc_sz_v[gcr,gcc,:],grid.gc_sz_h[gcr,gcc,:]]  
        gc_smpstep  = grid.smpstep[gcr,gcc,:]

        # preallocate result output
        duv_series      = np.empty(3).astype('float32');        duv_series[:]       = np.nan 
        kxky_series     = np.empty((Nw,2)).astype('float16');   kxky_series[:]      = np.nan 
        omega_series    = np.empty((Nw)).astype('float16');     omega_series[:]     = np.nan 
        pWeight_series  = np.empty((Nw)).astype('float16');     pWeight_series[:]   = np.nan 
        rmse_series     = np.nan 
        r2_series       = np.nan 
        conf95_series   = np.empty(3).astype('float16');        conf95_series[:]    = np.nan  
        
        cxcy_series         = np.empty((len(cxy_omegas),2)).astype('float16');   cxcy_series[:]         = np.nan 
        s2_w_cxcy_series    = np.empty((len(cxy_omegas),2)).astype('float16');   s2_w_cxcy_series[:]    = np.nan 
        
        for iii in range(1):# to be able to break if things go 'bad'
            
            if grid.mask[gcr,gcc]: 
                break
            
            #build pyramid cell
            gc_dm_Pyramid, gc_Tapers, pattern_Weights, gc_valid_w = self.gc_dm_buildPyramid(dmd, gc_sz, gc_smpstep, x_px, y_px, Nw, gc)    
            #check if at least one cell layer
            gc_Nw       = np.sum(gc_valid_w)                        
            bad =  gc_Nw == 0
            if bad: break
            #get frequencies of usable layers
            gc_omegas   = dmd.omega[gc_valid_w]
            #get dx (for lower resolution, bi-linearly interpolated layers)
            gc_dx       = grid.smpstep[gcr,gcc,gc_valid_w]*dx            
            # get wavenumbers prior to filtering and fitting
            Kx_prior, Ky_prior, Omega_prior, Weights_prior, omegas_prior, Nw, bad   = self.gc_dm_K_prior(opts, gc_dm_Pyramid = gc_dm_Pyramid, gc_Tapers = gc_Tapers, gc_omegas = gc_omegas, gc_dx = gc_dx, gc_Nw = gc_Nw, weights = pattern_Weights)             
            if bad: # all spectral layers are centered around k == 0 --> no directionality --> bad or too noisy data in cell
                break
            # filter 1 (current SSPC, i.e. without previous estimates)
            # --------
            cloudtype = opts.cloudtype_1filt
            reflect   = False
            Kx_filt1, Ky_filt1, Omega_filt1, Weights_filt1, kx_filt1, ky_filt1, _, _, weights_filt1, omegas_filt1, Nw_filt1, badcode = self.filter_cloud(opts, Kx_prior, Ky_prior, Omega_prior, Weights_prior, omegas_prior, Nw, cloudtype, reflect)
            if badcode == 1: 
                break 
            # store current SSPC
            # ------------------
            gc_dmdomega_IX  = np.isin(dmd.omega, omegas_filt1)
            kxky_series[gc_dmdomega_IX,0]   = kx_filt1
            kxky_series[gc_dmdomega_IX,1]   = ky_filt1
            omega_series[gc_dmdomega_IX]    = omegas_filt1  
            pWeight_series[gc_dmdomega_IX]  = weights_filt1                     
            # filter 2 (current SSPC + stored SSPCs, i.e. incl. previous estimates)
            # --------
            cloudtype = 'diff'
            reflect   = False
            if True:#OPTIONAL: don't fit SSPC of first update, but only DSPCs of later updates --> if (kx_store_gc[0] != []) | (kx_store_gc[1] != []):  
                #augment SSPC to DSPC (using temporarily stored SSPCs within Rad)
                Kx_combi        = self.get_inv_W(Kx_filt1,kx_store_gc)
                Ky_combi        = self.get_inv_W(Ky_filt1,ky_store_gc) 
                Weights_combi   = self.get_inv_W(Weights_filt1,pWeights_store_gc,1/4,1/4,1/2,True)
                Omega_combi     = self.get_inv_W(Omega_filt1,omegas_store_gc)#get_inv(Omega_filt1,[omegas_proj[np.concatenate(omega_ID_store_gc)]] if (omega_ID_store_gc != []) else [])

                omegas_combi    = np.unique(Omega_combi)  
                Nw_combi        = len(omegas_combi)
                #filter and fit DSPC if at least 3 layers in cell (we want direction filter to work, speeds up computation if check for 3 layers is done here)
                if (Nw_combi > 2):
                    Kx_filt, Ky_filt, Omega_filt, Weights_filt, _, _, cx_filt, cy_filt, weights_filt, omegas_filt, Nw_filt, badcode = self.filter_cloud(opts, Kx_combi, Ky_combi, Omega_combi, Weights_combi, omegas_combi, Nw_combi, cloudtype, reflect)
                    if badcode == 1: #no fitpoints, stop
                        break 
                    if badcode == 3: #less than 3 points to fit
                        break 
                    # layers are equally important, i.e. number of spectral points per layer shouldn't matter
                    ppOmga_Weights,_    = self.points_per_omega(Omega_filt, omegas_filt, Nw_filt)
                    Weights_filt_adj    = Weights_filt*ppOmga_Weights  
                    # -------------------------------------------------------------------------------
                    # Estimate phase velocities per T in manually defined array (see opts.cxy_deltaT)
                    # -------------------------------------------------------------------------------  
                    # save velocity vectors (for plotting)
                    ID = np.zeros(Nw_filt).astype(int)
                    #print(np.mean(np.diff(Omega_filt)))
                    for ii in range(Nw_filt):
                        ID[ii] = np.argmin(np.abs(omegas_filt[ii] - cxy_omegas))
                    ID_unique = np.unique(ID)   
                    for ii in ID_unique:                           
                        IX = ID == ii
                        V1          = np.sum(weights_filt[IX])
                        #calculate weighted confidence for cx and cy
                        if np.sum(IX) == 1:
                            s2_w_cxcy_series[ii,0]  = np.inf#0
                            s2_w_cxcy_series[ii,1]  = np.inf#0
                        else:                            
                            V2          = np.sum(weights_filt[IX]**2)    
                            sigma2_w_cx             = np.sum(weights_filt[IX]*(cx_filt[IX]-np.mean(cx_filt[IX]))**2)/V1 #biased weighted sample variance
                            s2_w_cxcy_series[ii,0]  = sigma2_w_cx/(1-(V2/V1**2)) #final unbiased estimate of sample variance
                            sigma2_w_cy             = np.sum(weights_filt[IX]*(cy_filt[IX]-np.mean(cy_filt[IX]))**2)/V1 #biased weighted sample variance
                            s2_w_cxcy_series[ii,1]  = sigma2_w_cy/(1-(V2/V1**2)) #final unbiased estimate of sample variance
                        #get weighted cx and cy
                        cxcy_series[ii,0]   = np.sum(cx_filt[IX]*weights_filt[IX])/V1
                        cxcy_series[ii,1]   = np.sum(cy_filt[IX]*weights_filt[IX])/V1
                    #checkplot
                    #fig = plt.figure(); ax = plt.axes(projection="3d");ax.set_xlim([-0.7,0.7]);ax.set_ylim([-0.7,0.7]); pl = ax.scatter3D(Kx_filt, Ky_filt, Omega_filt, c=Weights_filt, cmap='jet');fig.colorbar(pl);plt.pause(0.1);plt.pause(0.1)
                    
                    # ------------------------------------------------------
                    # Estimate depths and surface currents through inversion
                    # ------------------------------------------------------    
                    duv_series, rmse_series, r2_series, conf95_series   = self.coneFit(opts, Kx_filt, Ky_filt, Omega_filt, Weights_filt_adj, omegas_filt, ineq_cons, bounds, f_scale, fun, jac, conf_int_95)       
        
        return duv_series, cxcy_series, kxky_series, omega_series, pWeight_series, rmse_series, r2_series, conf95_series, s2_w_cxcy_series
    
    def coneFit(self, opts, Kx_cloud, Ky_cloud, Omega_cloud, Weights_cloud, omegas, ineq_cons, bounds, f_scale, fun, jac, conf_int_95):
        #initial depth estimate without fitting
        Kxy_cloud = np.sqrt(Kx_cloud**2+Ky_cloud**2)
        d0_arr  = 1/Kxy_cloud * np.arctanh(Omega_cloud**2/(9.81*Kxy_cloud))
        d0      = np.nansum(d0_arr*Weights_cloud)/np.nansum(Weights_cloud)
        #d0      = np.nanmedian(d0_arr)
        if (np.isnan(d0)) | (np.isinf(d0)) | (d0 < min(opts.dlims)) | (d0 > max(opts.dlims)):
            print('d0 set to d0_ini')
            d0 = opts.d0_ini
        # initialize d,u,v
        x0      = np.array([d0, 0.0, 0.0]) # set start estimate for d,u,v
        #x0      = np.array([opts.d0_ini, 0.0, 0.0]) # set start estimate for d,u,v   
        #------
        #Fit gc
        #------
        # via tweaked LM approach with less possibilities for regularization 
        if opts.fitter == 'LM+activation': 
            # fit standard LM-way + activation function
            if opts.loss == 'linear': # no cauchy loss
                fit_results_d   = least_squares(self.fun_d, x0[0], jac = self.jac_d, method = 'lm', loss = 'linear', args = (Kx_cloud, Ky_cloud, Omega_cloud, Weights_cloud), verbose=0) #, max_nfev = 10     #LM with loss = linear cannot use depth bounds                   
                fit_results     = least_squares(self.fun_softbound, [fit_results_d.x, 0, 0], jac = self.jac_softbound, loss = 'linear', args = (Kx_cloud, Ky_cloud, Omega_cloud, opts.Ulim, Weights_cloud), max_nfev = 5, verbose=0)#for max_nfev > 5 surface current magnitudes go towards maximum           
            elif opts.loss == 'cauchy':
                fit_results_d   = least_squares(self.fun_d, x0[0], bounds = (min(opts.dlims), max(opts.dlims)), jac = self.jac_d, loss = 'cauchy', f_scale = f_scale, args = (Kx_cloud, Ky_cloud, Omega_cloud, Weights_cloud), verbose=0) #, max_nfev = 10
                fit_results     = least_squares(self.fun_softbound, [fit_results_d.x, 0, 0], jac = self.jac_softbound, loss = 'cauchy', f_scale = f_scale, args = (Kx_cloud, Ky_cloud, Omega_cloud, opts.Ulim, Weights_cloud), max_nfev = 5, verbose=0)        
        # via SLSQP with f_scale and linear/cauchy loss
        elif opts.fitter == 'SLSQP':                            
            #f_scale    = 1  # fit SLSQP without f_scale effects
            # first depth estimate (without Doppler shift, i.e. near-surface currents U)
            # --------------------------------------------------------------------------
            #fit_results_d   = least_squares(self.fun_d, x0[0], jac = self.jac_d, method = 'lm', loss = 'linear', args = (Kx_cloud, Ky_cloud, Omega_cloud, Weights_cloud), verbose=0) # without depth bounds and cauchy
            fit_results_d   = least_squares(self.fun_d, x0[0], bounds = (min(opts.dlims), max(opts.dlims)), jac = self.jac_d, loss = 'cauchy', f_scale = f_scale, args = (Kx_cloud, Ky_cloud, Omega_cloud, Weights_cloud), verbose=0)
            # second full estimate of d,u,v    
            # -----------------------------
            fit_results     = minimize(fun, [fit_results_d.x, 0, 0], args = (f_scale, np.sqrt(Kx_cloud**2+Ky_cloud**2), Kx_cloud, Ky_cloud, Omega_cloud, Weights_cloud), method = 'SLSQP', 
                                       jac = jac, constraints = [ineq_cons], options = {'disp': False, 'ftol': 1e-08, 'maxiter':100},
                                       bounds = bounds) 
# =============================================================================
#             #OPTIONAL: without first depth estimate
#             fit_results     = minimize(fun, x0, args = (f_scale, np.sqrt(Kx_cloud**2+Ky_cloud**2), Kx_cloud, Ky_cloud, Omega_cloud, Weights_cloud), method = 'SLSQP', 
#                                        jac = jac, constraints = [ineq_cons], options = {'disp': False, 'ftol': 1e-08, 'maxiter':100},#, 'maxiter':30
#                                        bounds = bounds)  
# =============================================================================
            #print(fit_results.nfev)
        duv     = copy(fit_results.x)  
        # ----------------
        # get error of fit
        # ----------------
        if not (np.any(np.isnan(duv))):# | (duv[0] == d0)):
            try:
                if opts.fitter == 'LM+activation':
                    if (opts.loss == 'linear') | (opts.loss == 'cauchy'): #this line could be removed but is meant for understanding that for confidence intervals it here doesn't matter what loss function is chosen because function and jacobian values are already available, they don't have to be calculated as for SLSQP
                        conf95, rmse, rss   = self.non_linear_parameters_95_percent_confidence_interval(fit_results.fun, fit_results.jac) #weights are in this case already included in fun and jac                         
                elif opts.fitter == 'SLSQP':
                    if (opts.loss == 'linear') | (opts.loss == 'cauchy'):# also here only added for understanding, since factory function set_fitfun has built conf_int_95 based on the chosen loss function
                        conf95, rmse, rss   = conf_int_95(f_scale, fit_results.fun, fit_results.x, Kx_cloud, Ky_cloud, Omega_cloud, Weights_cloud)                        

                if (opts.loss == 'linear') | (opts.fitter == 'LM+activation'):
                    wss     = np.sum((Omega_cloud-np.mean(Omega_cloud))**2)
                elif (opts.loss == 'cauchy') & (opts.fitter == 'SLSQP'):
                    S       = (Omega_cloud-np.mean(Omega_cloud))/f_scale
                    wss     = np.sum(f_scale**2*Weights_cloud*np.log1p(S**2))
                r2      = 1-(rss/wss)
                rmse    = np.round(rmse/0.002)*0.002
                r2      = np.round(r2/0.002)*0.002
            except:
                rmse    = np.nan
                r2      = np.nan
                print('error estimation failed')         
            #filter out bad results
            bad = ((np.isinf(rmse)) | (r2 < 0.0) | (np.isnan(rmse))) # | (rmse > 0.2)
        else:
            bad = True
            
        if bad:
            duv     = [np.nan, np.nan, np.nan]
            rmse    = np.nan
            r2      = np.nan
            conf95  = [np.nan, np.nan, np.nan]   
        
        return duv, rmse, r2, conf95 
    
    def filter_cloud(self, opts, Kx_cloud, Ky_cloud, Omega_cloud, Weights, omegas, Nw, cloudtype, reflect = False):
        #Gamma filter using representative kx,ky-point per cell layer
        def filter_Gamma(Kx_cloud, Ky_cloud, Omega_cloud, kx_cloud, ky_cloud, omegas, Nw, kk0lims):                 
            k_norm          = np.sqrt(kx_cloud**2+ky_cloud**2)
            bad             = np.zeros(len(Kx_cloud)).astype('bool')
            for ii in range(Nw):
                omega_IX    = Omega_cloud == omegas[ii]
                k0Airy      = omegas[ii]**2/9.81                
                Gamma       = k0Airy/k_norm[ii]
                if ((Gamma > max(kk0lims)) | (Gamma < min(kk0lims))):
                    bad     = bad | omega_IX                  
            return bad
        #Gamma filter using all kx,ky-points per cell layer
# =============================================================================
#         def filter_Gamma(Kx_cloud, Ky_cloud, Omega_cloud, kx_cloud, ky_cloud, omegas, Nw, kk0lims):                 
#             #breakpoint()
#             K_norm          = np.sqrt(Kx_cloud**2+Ky_cloud**2)
#             bad             = np.zeros(len(Kx_cloud)).astype('bool')
#             for ii in range(Nw):
#                 omega_IX    = Omega_cloud == omegas[ii]
#                 k0Airy      = omegas[ii]**2/9.81                
#                 Gamma       = k0Airy/K_norm[omega_IX]
#                 bad_Gamma   = (Gamma > max(kk0lims)) | (Gamma < min(kk0lims))
#                 omega_IX[omega_IX] = bad_Gamma
#                 bad         = bad | omega_IX 
#             return bad
# =============================================================================

        def filter_PropagationDir(Kx_cloud, Ky_cloud, Omega_cloud):
            kxkykw          = np.concatenate(([Kx_cloud],[Ky_cloud],[Omega_cloud]), axis = 0)
            U,S,_           = np.linalg.svd(kxkykw.T,full_matrices=False)
            omega_rot_IX    = np.argmax(S)
            US_rot          = U[:,omega_rot_IX]*S[omega_rot_IX]     #Read: the data described in V-coordinates (or: projection of data onto new unit vectors V)
            Vcoord_shift    = np.abs(US_rot)-Omega_cloud            #take absolute value because V(:,1) can point down, resulating in negative US(:,1)
            bad             = Vcoord_shift < 0
            return bad      
         
        def get_kxky_cxcy(Kx_cloud, Ky_cloud, Omega_cloud, Weights_cloud, omegas, Nw, cloudtype = 'diff'):
            kx_cloud        = np.zeros(Nw)
            ky_cloud        = np.zeros(Nw)
            cx_cloud        = np.zeros(Nw)
            cy_cloud        = np.zeros(Nw)
            weights_cloud   = np.zeros(Nw)
            for ii in range(Nw):
                omega_IX    = Omega_cloud == omegas[ii]
                if cloudtype == 'same':
                    kx_cloud[ii]    = np.median(Kx_cloud[omega_IX])
                    ky_cloud[ii]    = np.median(Ky_cloud[omega_IX])
                    weights_cloud[ii]  = np.median(Weights_cloud[omega_IX])
                elif cloudtype == 'diff':
                    kx_cloud[ii]    = np.sum(Kx_cloud[omega_IX]*Weights_cloud[omega_IX])/np.sum(Weights_cloud[omega_IX])
                    ky_cloud[ii]    = np.sum(Ky_cloud[omega_IX]*Weights_cloud[omega_IX])/np.sum(Weights_cloud[omega_IX])         
                    weights_cloud[ii]  = np.sum(Weights_cloud[omega_IX]*Weights_cloud[omega_IX])/np.sum(Weights_cloud[omega_IX]) #np.mean(Weights_cloud[omega_IX])  
                elif cloudtype == 'best':
                    best            = np.argmax(Weights_cloud[omega_IX])                    
                    kx_cloud[ii]    = Kx_cloud[omega_IX][best]
                    ky_cloud[ii]    = Ky_cloud[omega_IX][best]        
                    weights_cloud[ii]  = Weights_cloud[omega_IX][best] 

                k_ii       = np.sqrt(kx_cloud[ii]**2+ky_cloud[ii]**2)
                if k_ii == 0:   
                   continue
                else:   
                   c_ii             = omegas[ii]/k_ii
                   if kx_cloud[ii] == 0:
                       divver       = np.inf
                   else:  
                       divver       = ((ky_cloud[ii]/kx_cloud[ii])**2+1)
                   cx_cloud[ii]     = np.sqrt(c_ii**2/divver)*np.sign(kx_cloud[ii])
                   cy_cloud[ii]     = np.sqrt(c_ii**2-cx_cloud[ii]**2)*np.sign(ky_cloud[ii])
            return kx_cloud, ky_cloud, cx_cloud, cy_cloud, weights_cloud  
         
        for iii in range(1):
            badcode         = 0
            cx_cloud        = np.nan
            cy_cloud        = np.nan
            kx_cloud        = np.nan
            ky_cloud        = np.nan
            weights_cloud   = np.nan
            
            if len(np.unique(Omega_cloud)) > 4: # --> assume at least 5 layers are needed to decide correct direction here
                # filter cloud for propagation direction             
                try:
                    bad            = filter_PropagationDir(Kx_cloud, Ky_cloud, Omega_cloud)
                except:
                    print('svd for propagation determination did not converge...')
                    badcode = 1
                    break
                if np.all(bad):
                    badcode = 1
                    break
                if reflect:
                    Kx_cloud[bad]  = -Kx_cloud[bad]  
                    Ky_cloud[bad]  = -Ky_cloud[bad]
                else:
                    Kx_cloud       = Kx_cloud[~bad]   
                    Ky_cloud       = Ky_cloud[~bad]
                    Omega_cloud    = Omega_cloud[~bad] 
                    Weights        = Weights[~bad]             
                             
                    omegas         = np.unique(Omega_cloud)
                    Nw             = len(omegas)    
            # get representative kx,ky-point per cell layer       
            kx_cloud, ky_cloud,_,_,weights_cloud = get_kxky_cxcy(Kx_cloud, Ky_cloud, Omega_cloud, Weights, omegas, Nw, cloudtype)
            # first Gamma filter: filter points for Gamma = [0, 1.5]
            bad         = filter_Gamma(Kx_cloud, Ky_cloud, Omega_cloud, kx_cloud, ky_cloud, omegas, Nw, [0, 1.5])              
            if np.all(bad):
                badcode = 1
                break
            Kx_cloud       = Kx_cloud[~bad]   
            Ky_cloud       = Ky_cloud[~bad]
            Omega_cloud    = Omega_cloud[~bad] 
            Weights        = Weights[~bad]   
            omegas         = np.unique(Omega_cloud)
            Nw             = len(omegas)
            # if less than three cell layers mark with badcode = 2 but continue
            if Nw < 3:
                badcode = 2    
            #checkplot
            #fig = plt.figure(); ax = plt.axes(projection="3d");ax.set_xlim([-0.7,0.7]);ax.set_ylim([-0.7,0.7]); ax.scatter3D(Kx_cloud, Ky_cloud, Omega_cloud, c=Weights, cmap='hsv');plt.pause(0.1);plt.pause(0.1)
            # if 3 cell layers or more, filter cloud for propagation direction 
            if badcode != 2:                      
                try:
                    bad            = filter_PropagationDir(Kx_cloud, Ky_cloud, Omega_cloud)
                except:
                    print('svd for propagation determination did not converge...')
                    badcode = 1
                    break
                if np.all(bad):
                    badcode = 1
                    break
                if reflect:
                    Kx_cloud[bad]  = -Kx_cloud[bad]  
                    Ky_cloud[bad]  = -Ky_cloud[bad]
                else:
                    Kx_cloud       = Kx_cloud[~bad]   
                    Ky_cloud       = Ky_cloud[~bad]
                    Omega_cloud    = Omega_cloud[~bad] 
                    Weights        = Weights[~bad]             
                    omegas         = np.unique(Omega_cloud)
                    Nw             = len(omegas)
            # get representative kx,ky-point per cell layer (after rough Gamma filter and direction filter have been applied)           
            kx_cloud, ky_cloud,_,_,weights_cloud = get_kxky_cxcy(Kx_cloud, Ky_cloud, Omega_cloud, Weights, omegas, Nw, cloudtype)
            
            # second Gamma filter: filter points for Gamma = opts.kk0lims
            bad    = filter_Gamma(Kx_cloud, Ky_cloud, Omega_cloud, kx_cloud, ky_cloud, omegas, Nw, opts.kk0lims)                
            if np.all(bad):
                badcode = 1
                break
            Kx_cloud       = Kx_cloud[~bad]   
            Ky_cloud       = Ky_cloud[~bad]
            Omega_cloud    = Omega_cloud[~bad] 
            Weights        = Weights[~bad]
            omegas         = np.unique(Omega_cloud)
            Nw             = len(omegas)
            
            bad = ((Kx_cloud == 0) & (Ky_cloud == 0))
            if np.any(bad):
                Kx_cloud       = Kx_cloud[~bad]   
                Ky_cloud       = Ky_cloud[~bad]
                Omega_cloud    = Omega_cloud[~bad] 
                Weights        = Weights[~bad]
                omegas         = np.unique(Omega_cloud)
                Nw             = len(omegas)
# =============================================================================
#             if Nw == 1:
#                 badcode = 1   
#                 break
# =============================================================================              
            if len(Kx_cloud) < 3:
                badcode = 3           
            # get representative kx,ky-point per cell layer (after rough Gamma filter, direction filter and fine Gamma filter have been applied)            
            omegas         = np.unique(Omega_cloud)
            Nw             = len(omegas)
            kx_cloud, ky_cloud, cx_cloud, cy_cloud, weights_cloud = get_kxky_cxcy(Kx_cloud, Ky_cloud, Omega_cloud, Weights, omegas, Nw, cloudtype)
        return Kx_cloud, Ky_cloud, Omega_cloud, Weights, kx_cloud, ky_cloud, cx_cloud, cy_cloud, weights_cloud, omegas, Nw, badcode
            
    def gc_dm_K_prior(self, opts, gc_dm_Pyramid = None, gc_omegas = None, gc_Nw = None, gc_Tapers = None, gc_dx = None, weights = None):         
        
        def gc_dm_K_prior_layer(opts, gc_dm = None, p = None, omega = None, dx = None):#, p = None, fft2_sz = None
            #c_prior denotes phase velocity estimates before the inversion
            bad = False
            for iii in range(1):# to be able to break if things go 'bad'
                #autocorrelate cell layer
                cr              = correlate(gc_dm,gc_dm,mode = 'same')
                zero_IX         = cr == 0 
                #normalize
                cr[~zero_IX]    = cr[~zero_IX]/np.abs(cr[~zero_IX])           
                #---------------------------------
                #calculate k and error from 2D-FFT
                #---------------------------------
                #spectra of tapered cross-correlation
                cr_spec     = np.fft.fft2(cr*p)  
                cr_spec_A   = np.abs(cr_spec) #amplitude                
                # separate k from noise floor
                # get index IX
                if opts.Kth == 'Eth':
                    k_IX    = (cr_spec_A-np.min(cr_spec_A))/(np.max(cr_spec_A)-np.min(cr_spec_A)) > 0.5
                elif opts.Kth == 'Max':
                    k_IX    = cr_spec_A == np.max(cr_spec_A)
                # get ID
                k_ID    = np.where(k_IX)
                # get pure k spectrum
                k_spec  = np.zeros(cr_spec.shape).astype('complex128');k_spec[k_IX] = cr_spec[k_IX]
                # get cross correlated cell layer and taper
                tmp1 = cr*p
                # get wave pattern of k and taper
                tmp2 = np.fft.ifft2(k_spec);tmp2 = tmp2/np.max(np.abs(tmp2));tmp2[cr.shape[0]:,:] = 0;tmp2[:,cr.shape[1]:] = 0; tmp2[:cr.shape[0],:cr.shape[1]] = tmp2[:cr.shape[0],:cr.shape[1]]*p
                # see how well k approximates actual wave pattern
                shift, error_direct, diffphase = phase_cross_correlation(tmp1,tmp2)
                
                kx1 = np.linspace(-2*np.pi/2, 2*np.pi/2, cr_spec.shape[1]+1); kx1 = kx1[:-1]#; kx1[0] = 0
                ky1 = np.linspace(-2*np.pi/2, 2*np.pi/2, cr_spec.shape[0]+1); ky1 = ky1[:-1]#; ky1[0] = 0
                kx_cr = (np.fft.fftshift(kx1)[k_ID[1]]/dx).tolist()
                ky_cr = (np.fft.fftshift(ky1)[k_ID[0]]/dx).tolist()
                #--------------------------------------------
                #calculate c (proxy for k) and error from PIV
                #--------------------------------------------
                #refine shift estimate  
                upsample    = 100
                #if not opts.standing_wave_flag:
                im1     = np.real(cr)*p; im1[im1<0] = 0
                im2     = np.imag(cr)*p; im2[im2<0] = 0
                shift, error, diffphase = phase_cross_correlation(im2,im1,upsample_factor = upsample)
                    
                dt025T      = 0.25/(omega/(2*np.pi))#90deg phase shift
                cx_prior    = shift[1]*dx/dt025T
                cy_prior    = shift[0]*dx/dt025T
                c_prior     = np.sqrt(cx_prior**2+cy_prior**2)
                
                #--------------------------------
                #combine 2D-FFT and PIV estimates
                #--------------------------------
                #bad if both estimates are == 0
                if (c_prior == 0) & np.all(np.concatenate(np.array([kx_cr])) == 0) & np.all(np.concatenate(np.array([ky_cr])) == 0):     
                    print('bad k estimates from PIV and 2D-FFT')
                    bad = True
                    break
                # if c from PIV is too small, only use k estimate from 2D-FFT
                if c_prior < np.sqrt(9.81*min(opts.dlims)):#< np.sqrt(9.81*opts.dlims):#== 0:        
                    print('bad k estimates from PIV')
                    Weight      = np.array([(1-error_direct)/np.size(kx_cr)]*np.size(kx_cr)) 
                    Kx_prior    = np.array([x for x in kx_cr]) #np.concatenate(np.array([kx_cr]))
                    Ky_prior    = np.array([x for x in ky_cr]) #np.concatenate(np.array([ky_cr]))
                    Omega_prior = np.array([omega]*np.size(kx_cr))  
                    #self.FFTcnt += 1 #not possible in parallel
                # else caclulate k from c for PIV and possibly combine with 2D-FFT   
                else: 
                    # caclulate k from c for PIV
                    k_prior     = omega/c_prior
                    if cx_prior == 0: 
                        divver  = np.inf
                    else:  
                        divver  = ((cy_prior/cx_prior)**2+1).astype('float64')
    
                    kx_prior    = np.sqrt(k_prior**2/divver)*np.sign(cx_prior)
                    ky_prior    = np.sqrt(k_prior**2-kx_prior**2)*np.sign(cy_prior)
                    # if k estimates from 2D-FFT are too small, only use k from PIV
                    if np.all(np.concatenate(np.array([kx_cr])) == 0) & np.all(np.concatenate(np.array([ky_cr])) == 0):
                        print('bad k estimates from 2D-FFT')
                        Weight      = np.array([1-error]) 
                        Kx_prior    = np.array([kx_prior]) 
                        Ky_prior    = np.array([ky_prior]) 
                        Omega_prior = np.array([omega])  
                        #self.PIVcnt += 1#not possible in parallel
                    # else use k from both 2D-FFT and PIV    
                    else:
                        Weight      = np.array([1-error] + [(1-error_direct)/np.size(kx_cr)]*np.size(kx_cr)) 
                        Kx_prior    = np.array([kx_prior] + [x for x in kx_cr]) 
                        Ky_prior    = np.array([ky_prior] + [x for x in ky_cr]) 
                        Omega_prior = np.array([omega] + [omega]*np.size(kx_cr))                          
                        #    self.PIVcnt += 1 #not possible in parallel                        
                        #    self.FFTcnt += 1 #not possible in parallel
# =============================================================================
#                 #OPTIONAL: discard points with low weights
#                 badpoints_IX = Weight < 0.4 #0.2
#                 if np.all(badpoints_IX):
#                     bad = True
#                     break
#                 else:
#                     Weight      = Weight[~badpoints_IX]
#                     Kx_prior    = Kx_prior[~badpoints_IX]
#                     Ky_prior    = Ky_prior[~badpoints_IX]
#                     Omega_prior = Omega_prior[~badpoints_IX]
# =============================================================================                                     
            if bad:
               Kx_prior     = np.nan
               Ky_prior     = np.nan
               Omega_prior  = np.nan
               omega        = np.nan
               Weight       = np.nan                    
            return Kx_prior, Ky_prior, Omega_prior, omega, Weight, bad
        
        # preallocate
        gc_PIV_weights  = [np.nan] * gc_Nw; gc_inp_weights  = [np.nan] * gc_Nw 
        gc_Omega_prior  = [np.nan] * gc_Nw; gc_omegas_prior = [np.nan] * gc_Nw  
        gc_Kxthr_prior  = [np.nan] * gc_Nw; gc_Kythr_prior  = [np.nan] * gc_Nw                   
        
        for ii in range(gc_Nw):
            gc_Kxthr_prior[ii], gc_Kythr_prior[ii], gc_Omega_prior[ii], gc_omegas_prior[ii], gc_PIV_weights[ii], bad = gc_dm_K_prior_layer(opts, gc_dm = gc_dm_Pyramid[ii], p = gc_Tapers[ii], omega = gc_omegas[ii], dx = gc_dx[ii])                                       
            if not bad: 
                gc_inp_weights[ii] = np.ones(gc_Kxthr_prior[ii].shape)*weights[ii]
                
        ii_bad  = np.isnan(gc_omegas_prior)
        if np.all(ii_bad):
            bad         = True
            Kx_prior    = np.nan
            Ky_prior    = np.nan
            Omega_prior = np.nan
            Weights     = np.nan
            omegas      = np.nan
            Nw          = np.nan 
        else:
            # Concatenate
            # -----------
            bad         = False
            Kx_prior    = np.hstack(list(compress(gc_Kxthr_prior,~ii_bad)))
            Ky_prior    = np.hstack(list(compress(gc_Kythr_prior,~ii_bad)))
            Omega_prior = np.hstack(list(compress(gc_Omega_prior,~ii_bad)))
            Weights_PIV = np.hstack(list(compress(gc_PIV_weights,~ii_bad)))
            #Weights_inp = np.hstack(list(compress(gc_inp_weights,~ii_bad)))                
            omegas      = np.hstack(list(compress(gc_omegas_prior,~ii_bad))) 
            Nw          = np.sum(~ii_bad)
            
            Weights     = Weights_PIV
            #Weights     = Weights_inp
        return Kx_prior, Ky_prior, Omega_prior, Weights, omegas, Nw, bad       
    
    def gc_dm_buildPyramid(self, dmd, gc_sz, gc_smpstep, x_px, y_px, Nw, gc):
        # build Pyramid Cell (gc_dm_Pyramid)
        # ----------------------------------                                      
        # get dmd info within (frequency-coupled) window extents
        # (lower frequencies --> larger windows) and stack to acquire 
        # pyramid shaped cell
        gc_dm_Pyramid   = []
        gc_Tapers       = []        
        Weights         = []      
        gc_valid_w      = np.zeros(Nw, dtype = bool)
        for ii in range(Nw):# loop through dmd frequencies
            gc_sz_wii = np.array([gc_sz[0][ii],gc_sz[1][ii]]) 
            if np.any(gc_sz_wii == 0):
                continue
            # get dynamic mode info in cell
            gc_dm   = dmd.phi[y_px-int(gc_sz_wii[0]/2):y_px+int(gc_sz_wii[0]/2), 
                              x_px-int(gc_sz_wii[1]/2):x_px+int(gc_sz_wii[1]/2), ii]                                 

            if gc_smpstep[ii] > 1:
                gc_dm1 = rescale(np.real(gc_dm),1/gc_smpstep[ii],order = 1)#mode = 'symmetric')
                gc_dm2 = rescale(np.imag(gc_dm),1/gc_smpstep[ii],order = 1)#mode = 'symmetric')
                gc_dm  = gc_dm1+1j*gc_dm2
                if (gc_dm.shape[0] % 2 == 1) | (gc_dm.shape[1] % 2 == 1):
                    breakpoint()
            # lowpass filter
# =============================================================================
#             tmp = np.array([[0,1,0],[1,4,1],[0,1,0]])
#             gc_dm = convolve2d(gc_dm, tmp, boundary='symm', mode='same')
# =============================================================================
            gc_dm = gc_dm/np.abs(gc_dm)
            gc_dm[np.isnan(gc_dm)] = 0    
            # demean
            gc_dm   = gc_dm-np.mean(gc_dm)   
            # get 2D taper                    
            p1      = np.hanning(gc_dm.shape[0])    
            p2      = np.hanning(gc_dm.shape[1])
            p       = p1[:,None] @ p2[None,:]  
            # taper cell
            gc_dm   = gc_dm*p  
            # estimate coherence in the pattern
# =============================================================================
#             U,S,V   = np.linalg.svd(gc_dm,full_matrices=False)
#             #gc_dm_pm    = np.outer(U[:,0], V[0,:]) # dominant signal in pattern
#             w       = S[0]**2/np.sum(S**2)
# =============================================================================
            w = 1 #simply set to 1 if no quality estimate of input data is done at this point
            gc_dm_Pyramid.append(gc_dm)
            gc_Tapers.append(p)                
            Weights.append(w)  
            gc_valid_w[ii] = True
        return gc_dm_Pyramid, gc_Tapers, Weights, gc_valid_w
    
    def get_inv_W(self,x,y,W_surr_store = 1,W_mid_store = 1,W_mid_now = 1,W_flag = False):
        if ((y[0] != []) & (y[1] != [])): #surrounding and midpoints present in storage
            store_surr  = np.concatenate(y[0])
            store_mid   = np.concatenate(y[1])
            now_mid     = x
            if W_flag:
                comb    = np.concatenate([store_surr/len(store_surr)*W_surr_store,store_mid/len(store_mid)*W_mid_store,now_mid/len(now_mid)*W_mid_now])
                comb    /= np.max(comb)
            else:
                comb    = np.concatenate([store_surr,store_mid,now_mid])              
        elif ((y[0] != []) & (y[1] == [])): #only surrounding points present in storage
            store_surr  = np.concatenate(y[0])
            if W_flag:
                comb    = np.concatenate([store_surr/len(store_surr)*W_surr_store,now_mid/len(now_mid)*W_mid_now])
                comb    /= np.max(comb)
            else:
                comb    = np.concatenate([store_surr,now_mid])
        elif ((y[0] == []) & (y[1] != [])): #only mid points present in storage
            store_mid   = np.concatenate(y[1])
            now_mid     = x
            if W_flag:
                comb    = np.concatenate([store_mid/len(store_mid)*W_mid_store,now_mid/len(now_mid)*W_mid_now])
                comb    /= np.max(comb)
            else:
                comb    = np.concatenate([now_mid,store_mid]) 
        else:
            comb    = x
            if W_flag:
                comb    /= np.max(comb)
        return comb
    
    def points_per_omega(self, Omega_cloud, omegas, Nw):
        ppOmga          = np.zeros(Nw)
        ppOmga_Weights  = np.zeros(len(Omega_cloud))
        for ii in range(Nw):
            omega_IX    = Omega_cloud == omegas[ii]
            ppOmga[ii]  = np.sum(omega_IX)
            ppOmga_Weights[omega_IX]    = 1/ppOmga[ii]
        ppOmga_Weights  = ppOmga_Weights/np.max(ppOmga_Weights)  #normalize to [..,1]  
        return ppOmga_Weights, ppOmga            
#%%   
    # OPTIMIZATION FUNCTIONS FOR LEAST-SQUARES FITTING
    # ------------------------------------------------        
    #@staticmethod  
    def fun_d(self, duv, kx_points, ky_points, kw_points, W):
        k_norm  = np.sqrt(kx_points**2+ky_points**2)
        M       = np.sqrt(9.81*k_norm*np.tanh(k_norm*duv[0]))
        return np.sqrt(W)*(M - kw_points) #(kw_points-M)
    
    def jac_d(self, duv, kx_points, ky_points, kw_points, W):
        k_norm  = np.sqrt(kx_points**2+ky_points**2)
        # get poartial derivatives
        Jd      = np.sqrt(W)*(np.sqrt(9.81*k_norm)*0.5*np.tanh(k_norm*duv[0])**np.float64(-1/2))*(1-np.tanh(k_norm*duv[0])**np.float64(2))*k_norm # incl. weights: sqrt(W) * (more info: for implementation of Weights see help nlinfit)
        if (np.any(np.isinf(Jd))) | (np.any(np.isnan(Jd))):
            breakpoint()
        return Jd[:,None]  
    
    def fun_softbound(self, duv, kx_points, ky_points, kw_points, uvlim, W):
        k_norm  = np.sqrt(kx_points**2+ky_points**2)
        M       = np.sqrt(9.81*k_norm*np.tanh(k_norm*duv[0]))+ duv[1]*kx_points + duv[2]*ky_points
        A       = np.max([1,np.sqrt(duv[1]**2+duv[2]**2)/uvlim]) #activation
        #A       = 1+np.log(1+np.exp(alpha*(np.sqrt(duv[1]**2+duv[2]**2)-uvlim)))
        M       = M*A
        return np.sqrt(W)*(M - kw_points) #(kw_points-M)
    
    def jac_softbound(self, duv, kx_points, ky_points, kw_points, uvlim, W):
        J       = np.empty((kx_points.size, duv.size))
        k_norm  = np.sqrt(kx_points**2+ky_points**2)
        
        M       = np.sqrt(9.81*k_norm*np.tanh(k_norm*duv[0]))+ duv[1]*kx_points + duv[2]*ky_points
        A       = np.max([1,np.sqrt(duv[1]**2+duv[2]**2)/uvlim])
        # get partial derivatives
        # M
        JMd     = (np.sqrt(9.81*k_norm)*0.5*np.tanh(k_norm*duv[0])**np.float64(-1/2))*(1-np.tanh(k_norm*duv[0])**np.float64(2))*k_norm # incl. weights: sqrt(W) * (more info: for implementation of Weights see help nlinfit)
        JMu     = kx_points           
        JMv     = ky_points
        # A
        JAd     = 0
        if np.sqrt(duv[1]**2+duv[2]**2) < uvlim:
            JAu = 0
            JAv = 0
        else:
            JAu = 0.5*(duv[1]**2+duv[2]**2)**-0.5 * 2*duv[1]
            JAv = 0.5*(duv[1]**2+duv[2]**2)**-0.5 * 2*duv[2]       
            
        Jd      = np.sqrt(W)*(JMd*A + M*JAd)
        Ju      = np.sqrt(W)*(JMu*A + M*JAu)
        Jv      = np.sqrt(W)*(JMv*A + M*JAv)
        # form Jacobian
        J[:,0]  = Jd
        J[:,1]  = Ju
        J[:,2]  = Jv
        return J
    
    def set_fitfun(self, opts, f_scale):#factory function
        if opts.loss == 'cauchy':
            def fun(duv, f_scale, k_norm, kx_points, ky_points, kw_points, W):#self, 
                M       = np.sqrt(9.81*k_norm*np.tanh(k_norm*duv[0]))+ duv[1]*kx_points + duv[2]*ky_points
                F       = (M-kw_points)/f_scale
                G2      = f_scale**2*W*np.log1p(F**2) #G**2
                cost    = 0.5*np.sum(G2)
                return cost
                
            def jac(duv, f_scale, k_norm, kx_points, ky_points, kw_points, W):#self, 
                J       = np.empty((duv.size,1))
                M       = np.sqrt(9.81*k_norm*np.tanh(k_norm*duv[0]))+ duv[1]*kx_points + duv[2]*ky_points
                F       = (M-kw_points)/f_scale
                # F deriv
                Jd      = (1/f_scale)*(np.sqrt(9.81*k_norm)*0.5*np.tanh(k_norm*duv[0])**np.float64(-1/2))*(1-np.tanh(k_norm*duv[0])**np.float64(2))*k_norm # incl. weights: sqrt(W) * (more info: for implementation of Weights see help nlinfit)
                Ju      = (1/f_scale)*kx_points           
                Jv      = (1/f_scale)*ky_points   
                # G2 deriv
                G2d     = f_scale**2*W*1/(1+F**2)*2*F*Jd
                G2u     = f_scale**2*W*1/(1+F**2)*2*F*Ju
                G2v     = f_scale**2*W*1/(1+F**2)*2*F*Jv
                # cost deriv (here J)
                J[0]    = 0.5*np.sum(G2d)
                J[1]    = 0.5*np.sum(G2u)
                J[2]    = 0.5*np.sum(G2v)
                return J
            
            def conf_int_95(f_scale, fnorm, duv, kx_points, ky_points, kw_points, W_points):
                n       = kx_points.size
                p       = duv.size
                J       = np.empty((kx_points.size, duv.size))
                k_norm  = np.sqrt(kx_points**2+ky_points**2)
                M       = np.sqrt(9.81*k_norm*np.tanh(k_norm*duv[0]))+ duv[1]*kx_points + duv[2]*ky_points
                F       = (M-kw_points)/f_scale
                
                G2      = f_scale**2*W_points*np.log1p(F**2)
                rss     = np.sum(G2)
                # F deriv
                Jd      = (1/f_scale)*(np.sqrt(9.81*k_norm)*0.5*np.tanh(k_norm*duv[0])**np.float64(-1/2))*(1-np.tanh(k_norm*duv[0])**np.float64(2))*k_norm # incl. weights: sqrt(W) * (more info: for implementation of Weights see help nlinfit)
                Ju      = (1/f_scale)*kx_points           
                Jv      = (1/f_scale)*ky_points
                # G deriv (--> here J is NOT cost deriv NOR G2 deriv...unlike in jac)
                J[:,0]  = 0.5*(f_scale**2*W_points*np.log1p(F**2))**(-0.5)*f_scale**2*W_points*1/(1+F**2)*2*F*Jd
                J[:,1]  = 0.5*(f_scale**2*W_points*np.log1p(F**2))**(-0.5)*f_scale**2*W_points*1/(1+F**2)*2*F*Ju
                J[:,2]  = 0.5*(f_scale**2*W_points*np.log1p(F**2))**(-0.5)*f_scale**2*W_points*1/(1+F**2)*2*F*Jv

                nmp     = n - p
                ssq     = rss / nmp
                _, s, VT    = np.linalg.svd(J, full_matrices=False) #scipy.linalg
                threshold   = np.finfo(float).eps * max(J.shape) * s[0]
                s       = s[s > threshold]
                VT      = VT[:s.size]
                c       = np.dot(VT.T / s**2, VT)
                pcov    = c * ssq
                err     = np.sqrt(np.diag(np.abs(pcov))) * stats.t.ppf(1-0.025, nmp) #--> use because sensitive to small sample sizes such as is the case here
                return err, np.sqrt(ssq), rss
            return fun, jac, conf_int_95
        
        elif opts.loss == 'linear':   
            def fun(duv, f_scale, k_norm, kx_points, ky_points, kw_points, W):#self, 
                M       = np.sqrt(9.81*k_norm*np.tanh(k_norm*duv[0]))+ duv[1]*kx_points + duv[2]*ky_points
                #F       = np.sqrt(W)*(M-kw_points) #same as below
                #return 0.5*np.sum(F**2)            #same as below
                F       = (M-kw_points)             #same as above
                G2      = W*F**2                    #same as above
                cost    = 0.5*np.sum(G2)            #same as above    
                return cost
                              
            def jac(duv, f_scale, k_norm, kx_points, ky_points, kw_points, W):#self, 
                J       = np.empty((duv.size,1))
                M       = np.sqrt(9.81*k_norm*np.tanh(k_norm*duv[0]))+ duv[1]*kx_points + duv[2]*ky_points
                # get partial derivatives
                F       = (M-kw_points)
                # F deriv
                Jd      = (np.sqrt(9.81*k_norm)*0.5*np.tanh(k_norm*duv[0])**np.float64(-1/2))*(1-np.tanh(k_norm*duv[0])**np.float64(2))*k_norm # incl. weights: sqrt(W) * (more info: for implementation of Weights see help nlinfit)
                Ju      = kx_points           
                Jv      = ky_points    
                # G2 deriv
                G2d     = W*2*F*Jd
                G2u     = W*2*F*Ju
                G2v     = W*2*F*Jv                
                # cost deriv (here J)
                J[0]    = 0.5*np.sum(G2d)
                J[1]    = 0.5*np.sum(G2u)
                J[2]    = 0.5*np.sum(G2v)               
                return J
            
            def conf_int_95(f_scale, fnorm, duv, kx_points, ky_points, kw_points, W_points):
                n       = kx_points.size
                p       = duv.size
                J       = np.empty((kx_points.size, duv.size))
                k_norm  = np.sqrt(kx_points**2+ky_points**2)
                M       = np.sqrt(9.81*k_norm*np.tanh(k_norm*duv[0]))+ duv[1]*kx_points + duv[2]*ky_points
               
                F       = (M-kw_points)
                G       = np.sqrt(W_points)*F# == np.sqrt(W_points*F**2) --> anologous formulation as for cauchy
                rss     = np.sum(G**2)
                #F deriv
                Jd      = (np.sqrt(9.81*k_norm)*0.5*np.tanh(k_norm*duv[0])**np.float64(-1/2))*(1-np.tanh(k_norm*duv[0])**np.float64(2))*k_norm # incl. weights: sqrt(W) * (more info: for implementation of Weights see help nlinfit)
                Ju      = kx_points           
                Jv      = ky_points
                #G deriv
                J[:,0]  = np.sqrt(W_points)*Jd
                J[:,1]  = np.sqrt(W_points)*Ju
                J[:,2]  = np.sqrt(W_points)*Jv
                
                nmp     = n - p
                ssq     = rss / nmp
                _, s, VT    = np.linalg.svd(J, full_matrices=False) #scipy.linalg
                threshold   = np.finfo(float).eps * max(J.shape) * s[0]
                s       = s[s > threshold]
                VT      = VT[:s.size]
                c       = np.dot(VT.T / s**2, VT)
                pcov    = c * ssq
                err     = np.sqrt(np.diag(np.abs(pcov))) * stats.t.ppf(1-0.025, nmp) #--> use because sensitive to small sample sizes such as is the case here
                return err, np.sqrt(ssq), rss

            return fun, jac, conf_int_95         
    
    def domega_maxcurv_from_cauchy_softmargin(self, f_scale):
        f_scale2    = f_scale**2 # softmargin squared
        #exactly
# =============================================================================
#         domega2     = np.linspace(0,1,100000) #delta omega squared
#         kappa       = np.abs(-1/f_scale2*(1+domega2/f_scale2)**-2)*(1+(1+domega2/f_scale2)**-2)**-3/2
#         opt_IX      = np.nanargmax(kappa)        
#         domega2_maxcurv     = domega2[opt_IX]       
#         return np.sqrt(domega2_maxcurv)
# ============================================================================= 
        #or via limit coefficient
# =============================================================================
#         f_scale2    = np.linspace(0,1,10000) 
#         f_scale2    = f_scale2[1:]
#         domega2_f_scale2_maxcurv    = np.zeros(len(f_scale2))
#         domega2     = np.linspace(0,1,100000) 
#         for ii in range(len(f_scale2)):
#             kappa           = np.abs(-1/f_scale2[ii]*(1+domega2/f_scale2[ii])**-2)*(1+(1+domega2/f_scale2[ii])**-2)**-3/2
#             opt_IX          = np.nanargmax(kappa)  
#             domega2_f_scale2_maxcurv[ii] = domega2[opt_IX]/f_scale2[ii]
# =============================================================================
        domega2_f_scale2_maxcurv = 0.4142
        return np.sqrt(domega2_f_scale2_maxcurv*f_scale2)
    
    def non_linear_parameters_95_percent_confidence_interval(self, fvec, jac):
        """Returns the 95% confidence interval on parameters from
        non-linear fit results."""
        # residual sum of squares
        rss         = np.sum(fvec**2) #--> Note: loss function weights in Jacobian and function-residual are included --> see trf: scale_for_robust_loss_function
        # number of data points and parameters
        n, p        = jac.shape
        # the statistical degrees of freedom
        nmp         = n - p
        # mean residual error
        ssq         = rss / nmp
        # Do Moore-Penrose inverse discarding zero singular values
        #start = time.time()
        _, s, VT    = np.linalg.svd(jac, full_matrices=False) #scipy.linalg
        threshold   = np.finfo(float).eps * max(jac.shape) * s[0]
        s           = s[s > threshold]
        VT          = VT[:s.size]
        c           = np.dot(VT.T / s**2, VT)
        #print('calc time {:}\n'.format(time.time()-start))
# =============================================================================
#         # MATLAB way
#         start = time.time()
#         _,R         = np.linalg.qr(jac) #mode='reduced' default
#         Rinv        = np.linalg.lstsq(R,np.eye(R.shape[0],R.shape[1]))[0]
#         diag_info   = np.sum(Rinv*Rinv,1)
#         print('calc time {:}\n'.format(time.time()-start))
#         err         = np.sqrt(np.abs(diag_info)*ssq) * 1.96
# =============================================================================
# =============================================================================
#         # Calc Inverse way
#         # the Jacobian
#         start = time.time()
#         J = np.matrix(jac)
#         # covariance matrix
#         c = np.linalg.inv(J.T*J)
#         print('calc time {:}\n'.format(time.time()-start))
# =============================================================================
        # variance-covariance matrix.
        pcov        = c * ssq
        # Diagonal terms provide error estimate based on uncorrelated parameters.
        # The sqrt convert from variance to std. dev. units.
        #err         = np.sqrt(np.diag(np.abs(pcov))) * 1.96  # std. dev. x 1.96 -> 95% conf
        err         = np.sqrt(np.diag(np.abs(pcov))) * stats.t.ppf(1-0.025, nmp) #--> use because sensitive to small sample sizes such as is the case here
        #print('inv students t {:}\n'.format(stats.t.ppf(1-0.025, nmp)))
        # Here err is the full 95% area under the normal distribution curve. This
        # means that the plus-minus error is half of this value
        return err, np.sqrt(ssq), rss 
    