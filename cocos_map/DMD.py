# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:24:04 2020

@author: gawehn
"""

import numpy as np
import time
from scipy.linalg import svdvals, qr, lu, svd
from scipy.signal import detrend, hilbert, convolve2d
from variableProj import Varpro
from copy import copy
from optht import optht
from mpl_toolkits.mplot3d import axes3d
#from fbpca import pca as fbsvd
import matplotlib.pyplot as plt
#import dask.array as da
#from dask.array.linalg import svd_compressed as dasvd
#from dask.array import from_array
#from dask import visualize as davisualize

class DMD():

    __counter = 0    
    def __init__(self,opts):#(self,opts):       
        self.opts = opts
        self.Nw   = self.opts.r_DMD        
       # self.counter += 1 #makes counter visible
        type(self).__counter += 1     #use type(self) instead of DMD.counter because DMD has subclass OptDMD

    def __del__(self):
        type(self).__counter -= 1   #also works on OptDMD if we initialize superclass within subclass
        #does not work if name was overwritten by superclass 

    @classmethod
    def Instances(cls):
        return cls, cls.__counter
    
    def plot_modes(self,Video):
        breakpoint()
        fig     = plt.figure()
        ax      = fig.gca(projection='3d')
        t_ID    = 10
        offset  = -16
        colfac  = 0.5#2/3
        for t_ID in range(1):
            ax.cla()
            #ax.plot_surface(Video.X, Video.Y, Video.ImgSequence[:,:,t_ID] - offset/2, rstride=1, cstride=1, cmap='PuBu', shade = True)#, alpha=1
            ax.contourf(Video.X, Video.Y, Video.ImgSequence[:,:,t_ID], zdir='z', offset= -offset/2,  cmap='PuBu', vmin=-np.sum(self.b_orig)*colfac, vmax=np.sum(self.b_orig)*colfac) 
            for ii in range(self.Nw):
                #ax.plot_surface(Video.X, Video.Y, 0*np.real(self.phi[:,:,ii]*np.exp(1j*self.omega[ii]*self.t[t_ID])) + offset*(self.Nw-ii), rstride=1, cstride=1, cmap='Greys', shade = True)#, alpha=0.5
                #ax.contourf(Video.X, Video.Y, self.b_orig[ii]*np.real(self.phi[:,:,ii]*np.exp(1j*self.omega[ii]*self.t[t_ID])), zdir='z', offset=offset*(self.Nw-ii),  cmap='PuBu', vmin=-np.max(self.b_orig)*colfac, vmax=np.max(self.b_orig)*colfac) 
                aa = -0.2+0*np.real(self.phi[:,:,ii]);aa[:,0] = 1;aa[:,-1] = 1;aa[0,:] = 1;aa[-1,:] = 1
                ax.contourf(Video.X, Video.Y, aa, zdir='z', offset=offset*(self.Nw-ii),  cmap='Greys', vmin=-np.max(self.b_orig)*colfac, vmax=np.max(self.b_orig)*colfac) 
            ax.set_zlim(offset*self.Nw, -offset)
            ax.grid(False)   
            #plt.axis('off')
            ax.set_box_aspect((0.3, 0.3*Video.m/Video.n, 1.0))
            plt.pause(0.5);plt.pause(0.5)
            
    def get_times(self, Video, start, stop, stoptype = 'exclude'):
        print('create frame timestamps...')  
        #build an array of timestamps for the frame sequence
        #for equispaced video frames these timestamps are based on the framerate
        #(for non-equispaced frames, the timestamps should be imposed via Data.py in a furture release. Note that standing wave environments are not supported then, i.e. opts.standing_wave_flag = False)
        if stoptype == 'exclude':
            self.t  = np.linspace(start,stop-1,stop-start)*Video.dt
        elif stoptype == 'include':
            self.t  = np.linspace(start,stop,stop-start+1)*Video.dt 
        if self.opts.standing_wave_flag:
            self.t  = self.t[:-1]
            
    # get video matrix Xvid
    def get_Xvid(self, Video, start, stop):
        
        def minmax_norm(self,data):  
            print('   minmax normalize video...')
            normfac = np.nanmax(data)- np.nanmin(data)
            data    = (data-np.nanmin(data))/(normfac)  
            #data    = data - np.nanmean(data)
            return data,normfac
        
        def set_Xvid_nanzeroColumns2nans(self):
            print('   convert pixel zeros to NaN...')
            self.badpix_IX              = (np.sum((self.Xvid==0) | (np.isnan(self.Xvid)),1)  == self.Xvid.shape[1]) | (np.sum(np.isnan(self.Xvid),1) > 0)
            self.Xvid[self.badpix_IX,:] = np.nan 
            
        def detrend_Xvid(self): 
            print('   detrend frame sequence...')
            #self.Xvid = detrend(self.Xvid, axis = -1, type = 'constant')
            for ii in range(self.Xvid.shape[1]):
                self.Xvid[:,ii] = self.Xvid[:,ii] - np.nanmean(self.Xvid[:,ii])
        
        def handle_Xvid_nans(self):
            # optional: delete fully nan timeseries
            if self.opts.excl_nan_flag:
                print('   exclude fully NaN pixel timeseries...')
                self.Xvid   = self.Xvid[~self.badpix_IX,:]    
                self.Npx    = np.sum(~self.badpix_IX)
            # convert nans to 0s    
            self.Xvid[np.isnan(self.Xvid)] = 0
        
        def get_fourier(self):
            print('   get Fourier spectrum for comparison (not necessary)...', end =" ")
            start   = time.time()
            
            coeff_abs   = np.abs(np.fft.fft(self.Xvid,axis=-1))
            self.A_fft  = np.mean(coeff_abs,axis = 0)
            N           = len(self.A_fft)  # number of samples
            self.A_fft  = self.A_fft/N
            Fs          = 1/(self.t[1]-self.t[0])      # sampling frequency
            if self.opts.analytic_flag:
                self.omegas_fft = Fs*(np.linspace(0,int(N)-1,int(N)))/N*2*np.pi  
            else:    
                self.A_fft      = self.A_fft[:int(round(N/2))+1]*2
                self.omegas_fft = Fs*(np.linspace(0,int(round(N/2)),int(round(N/2))+1))/N*2*np.pi  
                coeff_abs   = coeff_abs[:,:int(round(N/2))+1]*2
            end     = time.time()    
            print('CPU time: {} s'.format(np.round((end-start)*100)/100))
            
            if self.opts.coast_detect:
                print('   try to detect dry areas...', end =" ")
                thr = 0.3
                #f_ocean_IX      = ((self.omegas_fft/(2*np.pi)) > min(self.opts.freqlims)) & ((self.omegas_fft/(2*np.pi)) < max(self.opts.freqlims))
                f_ocean_IX      = ((self.omegas_fft/(2*np.pi)) > 1/20) & ((self.omegas_fft/(2*np.pi)) < 1/3)
                EnoiseVsEwaves  = np.sum(coeff_abs[:,~f_ocean_IX], axis = -1)/np.sum(coeff_abs[:,f_ocean_IX], axis = -1)
                EnoiseVsEwaves[np.isnan(EnoiseVsEwaves) | np.isinf(EnoiseVsEwaves)] = 0
                EnoiseVsEwaves  = minmax_norm(self,EnoiseVsEwaves)[0]
                coastpix_IX                     = np.zeros((self.m*self.n),dtype = bool)#, dtype = 'float64')
                coastpix_IX[~self.badpix_IX]    = EnoiseVsEwaves > thr
                #plt.figure();plt.imshow(np.reshape(coastpix_IX,(self.m,self.n), order = 'F').astype(float), origin = 'lower');plt.colorbar();plt.pause(0.5);plt.pause(0.5)
                self.Xvid       = self.Xvid[EnoiseVsEwaves <= thr,:]                   
                self.badpix_IX  = coastpix_IX | self.badpix_IX
                self.Npx        = np.sum(~self.badpix_IX)
                end     = time.time()    
                print('CPU time: {} s'.format(np.round((end-start)*100)/100))
                
        def make_Xvid_analytic(self,npix_max = 25_000):
            if self.opts.analytic_flag: 
                print('   extend to analytic signal...', end =" ")
                start   = time.time()
                self.Xvid = self.Xvid.astype('complex64')
                iternum = int(np.ceil(self.Xvid.shape[0]/npix_max))
                if iternum > 1: #for large number of pixels, convert in badges
                    for ii in range(iternum-1):
                        self.Xvid[ii*npix_max:(ii+1)*npix_max,:] = hilbert(np.real(self.Xvid[ii*npix_max:(ii+1)*npix_max,:]), axis = -1).astype('complex64')
                        #print('{}% of video converted to analytic signal'.format(round((ii+1)/iternum*100)))
                    self.Xvid[(ii+1)*npix_max:,:] = hilbert(np.real(self.Xvid[(ii+1)*npix_max:,:]), axis = -1)   
                else:
                    self.Xvid = hilbert(np.real(self.Xvid), axis = -1)
                end     = time.time()    
                print('CPU time: {} s'.format(np.round((end-start)*100)/100))
                
        def standing_wave_detect(self):
            if self.opts.standing_wave_flag:
                print('   enable standing wave detection...')
                self.Xvid = np.vstack((self.Xvid[:,:-1],self.Xvid[:,1:]))
        
        print('build video matrix...')
        startt  = time.time()
        self.m      = Video.ImgSequence.shape[0]
        self.n      = Video.ImgSequence.shape[1]
        self.Npx    = self.m*self.n
        l           = stop-start
        
        # get video fragment
        self.Xvid = np.reshape(Video.ImgSequence[:,:,start:stop],(self.m*self.n,l), order="F").astype('float32')
        # fill pixle timeseries only containing 0s and nans with nans
        set_Xvid_nanzeroColumns2nans(self)
        # detrend             
        detrend_Xvid(self) 
        # normalize to [0 1]
        self.Xvid,self.normfac = minmax_norm(self,self.Xvid)
        # demean
# =============================================================================
#         self.Xvid -= np.nanmean(self.Xvid)        
# =============================================================================
        # convert nans to zeros and optionally delete fully nan timeseries
        handle_Xvid_nans(self)  
        # build analytic dignal
        make_Xvid_analytic(self, npix_max = 25_000)
        # get Fourier spectrum and pixels of coast
        get_fourier(self)        
        # enable standing wave detection
        standing_wave_detect(self)       
        # get number of DMD modes (optimal rank) according to Gavish and Donoho 2014
        #k = optht(self.Xvid, sv=svdvals(self.Xvid), sigma=None)
        endt    = time.time()    
        print('CPU time: {} s'.format(np.round((endt-startt)*100)/100))
        
    def get_u(self,rank): 
        def randsvd(A,rank,q):
            m,n     = A.shape
            nr      = min([max([2*rank,rank+5]),n])
            r       = np.random.randn(n,nr)
            QY      = A @ r #usually Y = ... but already name QY to overwrite and save memory
            QY      = qr(QY, overwrite_a = True, mode = 'economic')[0]      #usually qr(Y) #numpy qr is faster than scipy qr  #reduced              
            # perform subspace iteration            
            for j in range(q):
                QY      = A.conj().T @ QY #usually Y = ... 
                QY      = qr(QY, overwrite_a = True, mode = 'economic')[0] #usually qr(Y)    
                QY      = A @ QY #usually Y = ... 
                QY      = qr(QY, overwrite_a = True, mode = 'economic')[0] #usually qr(Y)    
            
            A       = QY.conj().T @ A #usually AY = ... but name A to overwrite and save memory
            U       = svd(A,full_matrices=False)[0] #,S,Vh #scipy:svd(A,full_matrices=False)[0]
            U       = QY @ U
            U       = U[:,:rank]
            #S       = np.diag(S[:rank])
            #V       = Vh.conj().T[:,:rank]
            return U
# =============================================================================
#         from sklearn.utils.extmath import randomized_svd
#         from sklearn.decomposition import TruncatedSVD
# =============================================================================
# =============================================================================
#         %time self.u  = randsvd(self.Xvid,rank,2)#self.Xvid[:self.Npx]
#         %time self.u  = np.linalg.svd(self.Xvid, full_matrices = False)
# =============================================================================
# =============================================================================
#         %time self.u  = svd(self.Xvid,rank)[0]
#         %time self.u  = da.linalg.svd(self.Xvid)[0]
# =============================================================================
        print('   get U basis of video matrix...', end =" ")
        start   = time.time()
        if self.Xvid.shape[0] > 1_000_000:#300_000:
# =============================================================================
#             print('Use dask')
#             breakpoint() # --> this requires another thoorough look. No consistent realization if repeated. Big effect on depth inversion
#             self.u,s,v  = dasvd(from_array(self.Xvid,chunks=(100_000,self.Xvid.shape[1])), rank)#[0]#da.linalg.
#             #davisualize(self.u,s,v)
#             self.u  = (self.u.compute()[:,:rank]).astype('complex64')
# =============================================================================
            self.u   = svd(self.Xvid,full_matrices=False)[0][:,0:rank]
        else:
            #%time self.u  = fbsvd(self.Xvid,rank)[0]#fbsvd
            self.u   = svd(self.Xvid,full_matrices=False)[0][:,0:rank]
        end     = time.time()
        print('CPU time: {} s'.format(np.round((end-start)*100)/100))
# =============================================================================
#         u,_,_   = svd(self.Xvid,full_matrices=False)
#         u       = u[:,0:rank]
#         self.u  = u                   
# =============================================================================
    def project_Xvid(self, npix_max = 100_000):
        print('   project video matrix on U basis...', end =" ")
        start   = time.time()
        iternum     = int(np.ceil(self.Xvid.shape[0]/npix_max))
        if iternum > 1: #for large number of pixels, convert in badges
            self.Xvid_uproj = 0j
            for ii in range(iternum-1):
                self.Xvid_uproj = self.Xvid_uproj + self.u.conj().T[:,ii*npix_max:(ii+1)*npix_max]  @ self.Xvid[ii*npix_max:(ii+1)*npix_max,:]                        
                #print('{}% of Xvid projected'.format(round((ii+1)/iternum*100)))
            self.Xvid_uproj = self.Xvid_uproj + self.u.conj().T[:,(ii+1)*npix_max:]  @ self.Xvid[(ii+1)*npix_max:,:]  
        else:
            self.Xvid_uproj = self.u.conj().T @ self.Xvid
        end     = time.time()    
        print('CPU time: {} s'.format(np.round((end-start)*100)/100))

    def filter_frequencybounds(self):
        
        def get_frequencies(self):
            self.omega  = np.abs(np.imag(self.alpha))
            self.f      = self.omega/(2*np.pi)
        def get_keep_phi(self):  
            keep_phi    = np.where((self.f > np.min(self.opts.freqlims)) & (self.f < np.max(self.opts.freqlims)))
            if self.opts.analytic_flag:
                keep_phi    = keep_phi[0]
            else:
                keep_phi    = keep_phi[0][range(0,len(keep_phi[0]),2)] #discard conjugate modes
            return keep_phi  
        
        print('frequency filter Dynamic Modes...', end =" ")
        start   = time.time()
        get_frequencies(self)
        keep_phi    = get_keep_phi(self)
        self.omega  = self.omega[keep_phi]
        self.f      = self.f[keep_phi]
        self.Nw     = len(keep_phi)
        self.phi    = self.phi[:,keep_phi]
        self.b      = self.b[keep_phi]    
        if not self.opts.analytic_flag:
            self.b  = self.b*2 #if good signal then conjugate has same energy. Otherwise conjugate pairs of b have to be added together instead.
        end     = time.time()    
        print('CPU time: {} s'.format(np.round((end-start)*100)/100))
    
    def del_weak_modes(self):
        del_IX  = self.b == 0 # see beta_tilde2phi_b for rules when b = 0
        if np.any(del_IX):
            print('delete {} weak Dynamic Mode(s)...'.format(np.sum(del_IX)))
            self.omega  = self.omega[~del_IX]
            self.f      = self.f[~del_IX]
            self.phi    = self.phi[:,~del_IX]
            self.b      = self.b[~del_IX]
            self.Nw     = len(self.b)
    
    def stack_phi(self):
        print('stack Dynamic Mode layers...')
        self.phi    = np.reshape(self.phi,(self.m,self.n,self.Nw), order="F")    
        
    def clean_phi(self):
        kernel = np.array([[0,1,0],[1,4,1],[0,1,0]])
        for ii in range(self.Nw):
            for rep in range(2):
                self.phi[:,:,ii] = convolve2d(self.phi[:,:,ii], kernel, boundary='symm', mode='same')
            zero_IX = self.phi[:,:,ii] == 0
            self.phi[:,:,ii][~zero_IX] = self.phi[:,:,ii][~zero_IX]/np.abs(self.phi[:,:,ii][~zero_IX])           
            
    def transform_phi2phaseIm(self):
        print('convert Dynamic Modes to phase images...')
        iszero_IX               = self.phi == 0
        self.phi[~iszero_IX]    = self.phi[~iszero_IX]/np.absolute(self.phi[~iszero_IX])
    
    def sort_dynamic_modes(self):
        print('   sort Dynamic Modes...')
        # sort according to frequency
        sortid      = np.argsort(np.absolute(np.imag(self.alpha)))
        self.phi    = self.phi[:,sortid]
        self.alpha  = self.alpha[sortid]
        self.b      = self.b[sortid]
        if not self.opts.analytic_flag:
            #pair up (assumes modes to be returned pairwise, which should be the case if good signal)
            alpha_s = copy(self.alpha) 
            b_s     = copy(self.b) 
            phi_s   = copy(self.phi)
            plusminus = (-1)**np.arange(self.opts.r_DMD-1) #[normal conjugate normal conjugate normal...]
            for jj in range(0,self.opts.r_DMD-1,2):
                plmn        = np.sign(np.imag(self.alpha[jj]))
                if plusminus[jj] == plmn:
                    alpha_s[jj] = self.alpha[jj];   alpha_s[jj+1]   = self.alpha[jj+1]
                    b_s[jj]     = self.b[jj];       b_s[jj+1]       = self.b[jj+1]
                    phi_s[:,jj] = self.phi[:,jj];   phi_s[:,jj+1]   = self.phi[:,jj+1]
                else:
                    alpha_s[jj] = self.alpha[jj+1]; alpha_s[jj+1]   = self.alpha[jj]
                    b_s[jj]     = self.b[jj+1];     b_s[jj+1]       = self.b[jj]
                    phi_s[:,jj] = self.phi[:,jj+1]; phi_s[:,jj+1]   = self.phi[:,jj]     
            self.phi    = phi_s 
            self.alpha  = alpha_s 
            self.b      = b_s
    
    def get_b_fourier(self):
        print('get Fourier compliant spectral amplitudes...')
        self.b_fourier  = self.b/np.sqrt(self.Npx) #To get onset (fourier) amplitude, divide by square root of number of pixels
        self.b_orig     = self.b_fourier*self.normfac #because Xvid was normalized, multiply by normalization factor.     
        
class ExactDMD(DMD):
    def __init__(self, opts):#(self,opts):    
        print('initialize exact DMD...')
        super(ExactDMD, self).__init__(opts)
    
    def __del__(self):
        super(ExactDMD, self).__del__()

    def get_dynamic_modes(self):
        XD1         = self.Xvid[:,:-1]
        XD2         = self.Xvid[:,1:]
        # step 1
        [U,S,Vh]    = svd(XD1,0)
        V           = Vh.conj().T
        # step 2
        Sinv        = np.diag(S[:self.opts.r_DMD]**-1)
        Atilde      = U[:,:self.opts.r_DMD].conj().T @ XD2 @ V[:,:self.opts.r_DMD] @ Sinv
        # step 3
        [D,W]       = np.linalg.eig(Atilde)
        # step 4
        self.phi    = XD2 @ V[:,:self.opts.r_DMD] @ Sinv @ W
        self.alpha  = np.log(D)/(self.t[1] - self.t[0])
        self.b      = np.linalg.lstsq(self.phi, self.Xvid[:,0],rcond=None)[0]
# treat optimized DMD via variable projections as subclass of DMD
# the set up enables the natural implementation/substitution of alternative DMD types
class OptDMD(DMD):

    def __init__(self, opts, alpha0 = None):#(self,opts):      
        print('initialize optimized DMD...')
        self.alpha0 = alpha0
        super(OptDMD, self).__init__(opts)
    
    def __del__(self):
        super(OptDMD, self).__del__()
    
    def __repr__(self):
        return "OptDMD(" + "opts" + ", " +  str(self.alpha0) +  ")" #call the instance (e.g. dmd) as str(dmd) or repr(dmd) and create new instance by new_dmd = eval(str/repr(dmd))

    def __str__(self):
        return "Options: " + "opts" + ", alpha0: " +  str(self.alpha0)     

    def get_alpha0(self,rank):
        # initialize alpha via exact-like DMD
        print('   find initial Dynamic Mode frequencies via', self.opts.iniDMD_exact, 'approach...', end =" ")
        start   = time.time()
        if self.alpha0 is None:
            # exact-like DMD algorithm of Askham and Kutz 2018 (can handle non-equidistant frame spacing)
            # -------------------------------------------------------------------------------------------
            if self.opts.iniDMD_exact == 'nonequidistant':
                order = 4
                print('      order = {}'.format(order))
                trunc = order - 2                      
# =============================================================================
#                 ux_start = copy(self.Xvid) #without projection on U also works but more expensive and not necessary
# =============================================================================
                if order > 1:
                    ux2       = self.Xvid_uproj[:,1+trunc:]
                if order > 1:
                    ux1       = self.Xvid_uproj[:,1+trunc-1:-1]
                if order > 3:  
                    ux0       = self.Xvid_uproj[:,1+trunc-2:-2] 
                if order > 4:  
                    ux1m      = self.Xvid_uproj[:,1+trunc-3:-3]    
                ux_start  = self.Xvid_uproj[:,:-(1+trunc)]
                
                t1  = self.t[trunc:-1]
                t2  = self.t[1+trunc:]
            
                dx  = (ux2-ux1) @ np.diag(1.0/(t2-t1))   
                if order == 2:
                    xin = (1/2*ux2 + 1/2*ux_start) #Crank-Nicholson
                elif order == 3:
                    xin = (5/12*ux2 + 2/3*ux1 - 1/12*ux_start)
                elif order == 4:  
                    xin = (9/24*ux2 + 19/24*ux1 - 5/24*ux0 + 1/24*ux_start)
                elif order == 5:  
                    xin = (251/720*ux2 + 646/720*ux1 - 264/720*ux0 + 106/720*ux1m - 19/720*ux_start)
                #xin = ux1/1.0 #implicit Euler
            
                [u1,s1,v1h]  = svd(xin,full_matrices=False)
            
                v1      = v1h.conj().T 
                u1      = u1[:,:rank]
                v1      = v1[:,:rank]
                s1inv   = np.diag(1.0/s1[:rank])
                
                atilde  = u1.conj().T @ (dx @ (v1 @ s1inv))
# =============================================================================
#                     self.alpha0  = (((order-2)/(order-1))*self.alpha0 + (1/(order-1))*np.linalg.eig(atilde)[0])
# =============================================================================
                self.alpha0  = np.linalg.eig(atilde)[0]
# =============================================================================
#             [D,W]       = np.linalg.eig(atilde)
#             phi         = dx @ (v1 @ (s1inv @ W))
#             b           = np.linalg.lstsq(phi, self.Xvid[:,0],rcond=None)[0]
# =============================================================================
            #self.s1 = s1[:rank]
            
            # exact DMD algorithm
            # -------------------
            if self.opts.iniDMD_exact == 'equidistant':
                XD1         = self.Xvid[:,:-1]
                XD2         = self.Xvid[:,1:]
                #forward DMD
                # step 1
                [U,S,Vh]    = svd(XD1, full_matrices = False)
                V           = Vh.conj().T
                # step 2
                sd          = S[:self.opts.r_DMD]
                Sinv        = np.diag(sd**-1)
                fAtilde     = U[:,:self.opts.r_DMD].conj().T @ XD2 @ V[:,:self.opts.r_DMD] @ Sinv         
                # step 3
                [D,W]       = np.linalg.eig(fAtilde)
                # step 4
                self.alpha0 = np.log(D)/(self.t[1] - self.t[0])
        end     = time.time()   
        print('CPU time: {} s'.format(np.round((end-start)*100)/100))

# =============================================================================
#             phi         = XD2 @ V[:,:self.opts.r_DMD] @ Sinv @ W
#             b           = np.linalg.lstsq(phi, self.Xvid[:,0],rcond=None)[0]
#             tolrank     = len(b)*np.finfo(float).eps
#             self.exrank = np.sum(sd>(tolrank*sd[0])) 
# =============================================================================
    def get_dynamic_modes(self):                
        def get_beta_tilde_alpha(self):  
            print('   get scaled Dynamic Modes and corresponding frequencies via variable projection...')   
            start   = time.time()      
             
            m       = self.opts.Nt     #simply via Xvid_tilde_T?!
            if self.opts.standing_wave_flag:
                m -= 1
            n       = self.opts.r_DMD
            ia      = self.opts.r_DMD
            iss     = self.opts.r_DMD
            
            #Xvid_tilde_T    = (self.u.conj().T @ self.Xvid).T
    
            tikh_flag = False
            if tikh_flag:
                gamma = 1.5 #get via L-curve
            else:
                gamma = 1
# =============================================================================
#             #get Tikhonov L-curve according to https://link.springer.com/article/10.1007/BF02149761
#             gammaarr   = np.array([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.25, 0.5, 1])#, 2, 3, 4, 5, 7.5, 10])
#             Err1 = []; Err2 = []
#             for gamma in gammaarr:
#                 varp    = Varpro(tikh_flag = True, gamma = gamma, fulljac_flag = False)            
#                 beta_tilde,omega,niter,err,imode,alphas,err_semi,err_side,res,irank  = varp.varpro_LM(Xvid_tilde_T, self.t, m, n, iss, ia, self.alpha0)#varpro2(y,t,phi,dphi,m,n,iss,ia,alpha0,opts,gamma,iftik)
#                 Err1.append(err_semi);Err2.append(err_side)
#             plt.figure();plt.scatter(np.log(Err1),np.log(Err2),5,gammaarr, cmap = 'jet');plt.colorbar();plt.pause(0.1);plt.pause(0.1)    
# =============================================================================
# =============================================================================
#             varp    = Varpro(tikh_flag = True, gamma = 0.04, fulljac_flag = False)                        
# =============================================================================
            varp    = Varpro(tikh_flag = tikh_flag, gamma = gamma, fulljac_flag = False, proxfun_flag = False)   #don't need to calculate full jacobian: costs more CPU and does not improve results         
            beta_tilde_T, self.alpha, self.niter, self.err, self.imode, self.alphas, self.err_semi, self.err_side, self.res, self.irank = varp.varpro_LM(self.Xvid_uproj.T, self.t, m, n, iss, ia, self.alpha0)
            self.beta_tilde    = beta_tilde_T.T
            
            if self.irank < self.opts.r_DMD:
                print('   rank reduced to {}'.format(self.irank))
            end     = time.time()   
            print('   CPU time: {} s'.format(np.round((end-start)*100)/100))
        
        def beta_tilde2phi_b(self):   
            print('   split scaled Dynamic Modes into Dynamic Modes with spectral weights...', end =" ")
            start   = time.time()
            # normalize
            b                       = np.sqrt(np.sum(np.absolute(self.beta_tilde)**2.0,0)).T
            inds_small              = b < 10*np.finfo(float).eps*np.max(b)
            b[inds_small]           = 1.0
            phi_tilde               = self.beta_tilde @ np.diag(1.0/b)
            phi_tilde[:,inds_small] = 0.0 #future implementation rather delete, but messes with current handling of dimensionalities, allocations and sorting (often based on self.opts.r_DMD): phi_tilde  = np.delete(phi_tilde, obj = np.where(inds_small)[0], axis = -1)
            b[inds_small]           = 0.0 #future implementation rather delete, but messes with current handling of dimensionalities, allocations and sorting (often based on self.opts.r_DMD): b          = np.delete(b, obj = np.where(inds_small)[0], axis = -1)
            if self.opts.excl_nan_flag:
                self.phi                    = np.zeros((self.m*self.n, self.opts.r_DMD), dtype = 'complex64')
                self.phi[~self.badpix_IX,:] = self.u[:self.Npx,:] @ phi_tilde #need to specify :np.sum(~self.badpix_IX) for case that opts.standing_wave_flag == True
            else:
                self.phi                    = self.u[:self.m*self.n,:] @ phi_tilde #need to specify :m for case that opts.standing_wave_flag == True
            
            if self.opts.standing_wave_flag:
                self.b      = b/np.sqrt(2)
            else:    
                self.b      = b                       
            end     = time.time()   
            print('CPU time: {} s'.format(np.round((end-start)*100)/100))
            
        print('get Dynamic Modes...')
        start   = time.time()    
        # get projection matrix: left singular vectors of Xvid
        self.get_u(self.opts.r_DMD)
        # convert video matrix to lower dimensional form                    
        self.project_Xvid()
        # initialize alpha: if none --> get via exact-like DMD   
        self.get_alpha0(self.opts.r_DMD)               
        success1 = False
        success2 = False
        try:
            get_beta_tilde_alpha(self)  # get projected weighted dynamic modes (beta_tilde) and corresponding frequencies(alpha, including real part representing increase/decrease)
            success1 = True
            if (self.opts.calcdmd == 'robust') and self.alpha0 is not None:
                print('   robust DMD calculation...')
                alpha_prev = copy(self.alpha0)
                err1 = self.err_semi + self.err_side
                try:
                    self.alpha0 = None          # if previous mimimum raises an error...
                    self.get_alpha0(self.opts.r_DMD)           # ...reinitialize alpha0 via exact-like dmd
                    get_beta_tilde_alpha(self)  # get projected weighted dynamic modes (beta_tilde) and corresponding frequencies(alpha, including real part representing increase/decrease)
                    success2 = True
                    err2 = self.err_semi + self.err_side
                except:
                    print('   new mimizer failed --> use current minimizer')
                    self.alpha0 = alpha_prev
                if success1 and success2:
                    if err1 < err2:
                        print('   new mimizer (res: {}) worse than current minimizer (res: {}) --> use current minimizer'.format(err2,err1))
                        self.alpha0 = alpha_prev
                        get_beta_tilde_alpha(self)  # get projected weighted dynamic modes (beta_tilde) and corresponding frequencies(alpha, including real part representing increase/decrease)
                        success1 = True
                    else:
                        print('   new mimizer (res: {}) better than current minimizer (res: {}) --> use new minimizer'.format(err2,err1))
        except:
            self.alpha0 = None          # if previous mimimum raises an error...
            self.get_alpha0(self.opts.r_DMD)           # ...reinitialize alpha0 via exact-like dmd
            get_beta_tilde_alpha(self)  # get projected weighted dynamic modes (beta_tilde) and corresponding frequencies(alpha, including real part representing increase/decrease)
            success2 = True
        finally:
            if success1:
                print('   DMD successfull on current mimimizer')
            elif success2:
                print('   DMD successfull on new mimizer')
            else:
                raise Exception('DMD failed')    
        
        # convert modes to phase images and sort by frequency               
        self.alpha0 = copy(self.alpha)  # remember estimated alpha as starting point for next image sequence            
        beta_tilde2phi_b(self)          # split weighted dynamic modes into unit-normalized dynamic modes (phi) and weights (b)             
        self.sort_dynamic_modes()       # sort dynamic modes by increasing frequency                  
        end     = time.time()
        print('CPU time: {} s'.format(np.round((end-start)*100)/100))
