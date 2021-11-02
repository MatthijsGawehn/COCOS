# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:26:08 2020

@author: gawehn
"""
# -----------------------------------------------------------------------------
# LOAD MODULES
# -----------------------------------------------------------------------------
#Python modules
import numpy             as np
import cv2               as cv #print(cv.__version__)
import time
import matplotlib.pyplot as plt
#COCOS modules
from Data      import Data
from Options   import Options
from DMD       import ExactDMD, OptDMD
from Inversion import Inversion
from Kalman    import Kalman
from Grid      import Grid
from Plot      import Plot

#%%
fieldsite   = 'duck'
# -----------------------------------------------------------------------------
# LOAD VIDEO DATA
# -----------------------------------------------------------------------------
# load video data
Video,PlotLims   = Data.get_Video(fieldsite)
#%% 
# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------
# set options
opts = Options(Video, CPU_speed = 'slow', parallel_flag = True, gc_kernel_sampnum = 80, f_scale = 0.012)
#%%
# -----------------------------------------------------------------------------
# INITIALIZE
# -----------------------------------------------------------------------------
# prepare parallel computation of grid cells
def gc_walk_unwrap_self(arg, **kwarg):      #__main__ function for inversion
    #print(__name__)
    return Inversion.gc_walk(*arg, **kwarg)
# initialize grid
grid = Grid(Video, opts)
# initialize inversion
InvObj  = Inversion()
# initialize spectral storage
InvStg  = Inversion.get_storage(grid)
# initialize SSPC sampling kernel
InvObj.get_gridKernel(grid,opts.gc_kernel_rad)
# initialize optimized DMD
dmd     = OptDMD(opts, alpha0 = None)
# initialize Kalman filter
KalObj  = Kalman(opts, grid)
# initialize plotting
plot    = Plot(opts, Video, grid, step = None)

# preallocation
# -------------
t_iter = [];                        # OPTIONAL: for saving results
Dk = [];    Uk = [];    Vk = [];    # OPTIONAL: for saving results
Cxk = [];   Cyk = [];               # OPTIONAL: for saving results

#%%
# -----------------------------------------------------------------------------
# PROCESS
# -----------------------------------------------------------------------------
frame_start     = 0
cnt             = 0
while frame_start+opts.Nt <= Video.ImgSequence.shape[2]: #(remove <= tt for unlimited analysis)        
    print('\n --------------- START Update #{:} ---------------\n'.format(cnt+1))    
    t_real_start    = time.time()
    t               = (frame_start + np.round(opts.Nt/2))*Video.dt
    # make timestamps of frame sequence
    dmd.get_times(Video, frame_start, frame_start+opts.Nt)
    # build video matrix
    dmd.get_Xvid(Video, frame_start, frame_start+opts.Nt)   
    try:
        # get Dynamic Modes
        dmd.get_dynamic_modes()
    except:
        print('previous local minimizer and re-initialized minimizer failed')
        print('try next image sequence?')
        t_real_end = time.time()
        t_shift = t_real_end - t_real_start
        print('iteration time {:} sec \n '.format(t_shift))
        cnt += 1
        if opts.frame_int == 'OnTheFly':
            frame_start = frame_start+int(t_shift/Video.dt)
        else:
            frame_start = frame_start+opts.frame_int
        continue    
    # frequency filter Dynamic Modes
    dmd.filter_frequencybounds()
    # delete weak Dynamic Modes
    dmd.del_weak_modes()
    # get Fourier compliant spectral amplitudes
    dmd.get_b_fourier()
    # convert Dynamic Modes to phase images
    dmd.transform_phi2phaseIm()
    # stack Dynamic Mode layers
    dmd.stack_phi() 
    # OPTIONAL: smoothe Dynamic Modes
    #dmd.clean_phi()    
    # get subdomain size and resolution per mode and location
    mask = np.reshape(dmd.badpix_IX,(Video.m,Video.n), order="F")# could be any mask
    grid.get_gcSizes(Video, opts, dmd.omega, mask) 
    # invert d,u,v,cx,cy 
    Results, InvStg     = InvObj.get_maps(Video, opts, dmd, grid, gc_walk_unwrap_self, InvStg, t)      
    # Kalman filter d,u,v,cx,cy    
    KalObj.Filter(opts, Results, t)
    # get current timestamp and shift to next image sequence
    t_real_end = time.time()
    t_shift = t_real_end - t_real_start
    cnt += 1
    if opts.frame_int == 'OnTheFly':
        frame_start = frame_start+int(t_shift/Video.dt)
    else:
        frame_start = frame_start+opts.frame_int
    # simple visualization of results
# =============================================================================
#     d_lims      = [0,6]
#     diff_lims   = [-1.5,1.5]
#     err_lims    = [0, 2]
# =============================================================================
    if cnt > 0:
        try:
            plot.results(opts, grid, Results, KalObj, InvStg, PlotLims.d_lims, PlotLims.diff_lims, PlotLims.err_lims, InvObj.kernel_samp, (dmd.A_fft, dmd.omegas_fft), (dmd.b_fourier,dmd.omega), t_shift)
        except:
            print('no plot. probably empty results')
#%%
# -----------------------------------------------------------------------------
# OPTIONAL: SAVE RESULTS
# -----------------------------------------------------------------------------            
    # save data from update for postprocessing       
    t_iter.append(t)    
    Dk.append(np.copy(KalObj.derrt_prev))
    Uk.append(np.copy(KalObj.uerrt_prev))
    Vk.append(np.copy(KalObj.verrt_prev))    
    Cxk.append(np.copy(KalObj.cxerrt_prev))
    Cyk.append(np.copy(KalObj.cyerrt_prev))
# save other     
Cxy_omega   = Results.c_omega
Dgt         = Data.get_GroundTruth(opts, Video, grid, step = None)

# =============================================================================
# np.savez('Results/' + fieldsite + '_CPU_speed_'+ opts.CPU_speed,    t_iter = t_iter, Dk = Dk, Uk = Uk, Vk = Vk, Cxk = Cxk, Cyk = Cyk, Cxy_omega = Cxy_omega, Dgt = Dgt, grid_dx = grid.dx, grid_X = grid.X, grid_Y = grid.Y, grid_Rows_ctr = grid.Rows_ctr, grid_Cols_ctr = grid.Cols_ctr) 
# =============================================================================
