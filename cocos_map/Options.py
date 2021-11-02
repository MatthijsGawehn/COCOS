# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:22:11 2020

@author: gawehn
"""
import numpy             as np

class Options():
    def __init__(self, Video,
                 parallel_flag      = False,            # run parallel?
                 analytic_flag      = True,             # build and use analytic video matrix
                 calcdmd            = 'standard',       # standard or robust (standard is faster, robust is slower but more accurate)
                 iniDMD_exact       = 'nonequidistant', # initialize either with exact-like DMD for 'nonequidistant' framespacing or 'equidistant' frame spacing (default for video). 'nonequidistant' always works, wether frames are spaced in a fixed interval or not, but it can be less accurate if the threshold for the number of modes 'r_DMD' is much larger than the actual number of modes in the data
                 Nt                 = 64,               # [#] frame bin size  (number of frames needed to aquire near-linearly dependend columns in operator A)    
                 excl_nan_flag      = True,             # increase computational speed --> ignore nan pixel timeseries. Setting areas in video to nan acts as a mask (e.g. solid ground). This increases computational speed.               
                 coast_detect       = False,            # detection of dry areas (not sophisticated, should be improved for future release)
                 standing_wave_flag = True,             # enable standing wave detection
                 freqlims           = [1/3, 1/15],      # [1/s] frequency limits for inversion
                 cxy_deltaT         = 1,                # [s] discretisation of wave periods on which to project phase velocities
                 dlims              = [0.1, 50],        # [m] depth limits for inversion
                 Ulim               = 0.75,             # [m/s] maximum allowed near-surface current magnitude
                 Npx_fftmin         = 24,               # [px] minimum size of 2d FFT tile (regardless of wave lengths/periods)
                 Nm_fftmin          = 10,               # [m] minimum size of 2d FFT tile (regardless of wave lengths/periods)
                 minPpOffWaveLength = 8,                # [#] minimum number of pixles to cover one offshore wave length
                 gc_OffWaveLengths  = 2,                # [#] preferred number of offshore wavelengths in tile
                 Kth                = 'Max',            # threshold type to seperate waves from noise floor. 'Eth' = relative energy threshold of 0.5; 'Max' Maximum energy point
                 cloudtype_1filt    = 'diff',           # handling of FFT and PIV estimate: 'best' of FFT and PIV. Or 'diff' weighted mean of FFT and PIV
                 CPU_speed          = 'normal',         # 'fast','normal','slow','exact' --> determines number of grid cells to be inverted
                 grid_dx            = None,             # [m] grid cell resolution can be hardcoded. CPU_speed is ignored in that case
                 gc_kernel_rad      = 75,               # [m] radius of kernel to be used around a grid cell to better capture wavedirections and their modulation by near-surface currents u,v
                 gc_kernel_sampnum  = 17,               # [#] number of grid cells that are sampled within gc kernel
                 reweight           = True,             # reweight spectral quality weights of SSPCs from surrounding gcs
                 fitter             = 'SLSQP',          # 'LM+activation','SLSQP' --> LM with loss = linear cannot use depth bounds
                 loss               = 'cauchy',         # 'linear','cauchy' --> 'cauchy' uses 'f_scale'  
                 f_scale            = 0.012,            # [-] only active if loss = 'cauchy' (depends on frames per second and typical frequency difference between Dynamic Modes) (can be automated in future release)
                 kk0lims            = [0.3, 1.0],       # [-] minimum and maximum k/k0 (i.e. "Gamma") for Gamma filter. higher kk0min means that waves longer than L should not be used to retrieve a local depth d (e.g. for T = 10s and d = 2.5m --> kk0min = 0.3 which equals L/d = 20. Hence for kk0min > 0.3 the wave length is more than 20 times larger than the depth, which indicates a loss of morphological detail)
                 d0_ini             = 5,                # [m] only used if automated seed fails: initialization depth for d,U-fit
                 frame_int          = 32,               # [#] frame shift --> give number of frames [#] e.g. frame_int = 8 (note: time shift = frame shift * dt) or let time shift depend on computation time frame_int = 'OnTheFly'
                 stationary_time    = 60,               # [s] time spectral points are stored for reuse in successive iterations
                 Q_d                = 0.0000,           # proces variance d
                 Q_U                = 0.0005,           # proces variance U
                 Q_C                = 0.0005,           # proces variance c
                 R_type             = 'fit_conf95',     # type of error estimation --> sensitivity of fit, i.e. 95% confidence of fit 'fit_conf95' or crudely based on previous estimate 'x_diff2'
                 **kwargs):
        
        print('set options...')
# =============================================================================
#         for k in kwargs:
#             exec('{KEY} = {VALUE}'.format(KEY = k, VALUE = repr(kwargs[k])))
# =============================================================================
        # processing
        # ----------       
        self.parallel_flag      = parallel_flag
        # Video data
        # ----------
        if 'Origin' in kwargs: ##########################################
            self.Origin         = kwargs['Origin']
        else:
            self.Origin         = [Video.X[0,0], Video.Y[0,0]]
        self.analytic_flag      = analytic_flag    
        # DMD
        # ---
        self.calcdmd            = calcdmd
        self.iniDMD_exact       = iniDMD_exact
        self.Nt                 = Nt
        self.excl_nan_flag      = excl_nan_flag 
        self.coast_detect       = coast_detect
        if 'r_DMD' in kwargs:
            self.r_DMD          = kwargs['r_DMD']
        else:
            if self.analytic_flag:
                self.r_DMD          = int(Nt/4)
            else:
                self.r_DMD          = int(Nt/2)
                self.r_DMD          = self.r_DMD + self.r_DMD % 2  
        self.standing_wave_flag = standing_wave_flag
        
        # Depth inversion
        # ---------------  
        if max(freqlims) <= 2/(Video.dt): #Nyquist criterion   
            self.freqlims       = freqlims
        else:
            freqlims[freqlims.index(max(freqlims))] = 1/(2*Video.dt)
            self.freqlims      = freqlims
            print('maximum frequency changed to Nyquist criterion max(freqlims) = {}'.format(max(freqlims)))
        
        self.cxy_deltaT         = cxy_deltaT                
        self.dlims              = dlims
        self.Ulim               = Ulim
        self.Npx_fftmin         = Npx_fftmin + Npx_fftmin % 2
        self.Nm_fftmin          = Nm_fftmin
        self.minPpOffWaveLength = minPpOffWaveLength
        self.gc_OffWaveLengths  = gc_OffWaveLengths
        self.Kth                = Kth
        self.cloudtype_1filt    = cloudtype_1filt
        self.CPU_speed          = CPU_speed
        self.grid_dx            = grid_dx
        self.gc_kernel_rad      = gc_kernel_rad
        self.gc_kernel_sampnum  = gc_kernel_sampnum
        self.reweight           = reweight
        self.fitter             = fitter   
        self.loss               = loss 
        self.f_scale            = f_scale 
        self.kk0lims            = kk0lims
        minNumPix_from_kk0lims  = int(2/min(self.kk0lims)) + int(2/min(self.kk0lims)) % 2 # read '2/':at least 2 pixels to recognize any wave length
        if minNumPix_from_kk0lims <= minPpOffWaveLength:
            self.minPpOffWaveLength = minPpOffWaveLength
        else:
            if minNumPix_from_kk0lims <= 10:
                self.minPpOffWaveLength = minNumPix_from_kk0lims
            else:
                self.minPpOffWaveLength = 10 #maximum number of pixels per wave length. This is a practical limit as for T = 15 s the wave is approx. 4.5 times the length in shallow water.
            print('minimum number of points per offshore wave length increased to {} according to minimum kk0 = {}'.format(self.minPpOffWaveLength,min(self.kk0lims)))
        self.d0_ini             = d0_ini
        # timestepping
        # ------------        
        self.frame_int          = frame_int
        self.stationary_time    = stationary_time
        # Kalman filter
        # -------------
        self.Q_d                = Q_d
        self.Q_U                = Q_U
        self.Q_C                = Q_C
        self.R_type             = R_type

        self.print_params()
        
    def print_params(self):
          print(self.__dict__)
# =============================================================================
#         print('Tmax: {}s'.format(np.round(1/self.freqlims[1])))
#         print('chosen Ulim: {}m/s'.format(self.Ulim))
#         print('frame interval: {}steps'.format(self.frame_int))
#         print('chosen Q_d: {}m**2'.format(self.Q_d))
#         print('minimum fft2 size: {}px'.format(self.Npx_fftmin)) 
# =============================================================================
        
