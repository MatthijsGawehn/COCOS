# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:50:51 2020

@author: gawehn
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from Data import Data
# =============================================================================
# from matplotlib.animation import FuncAnimation
# from copy import copy
# from scipy       import interpolate
# =============================================================================

class Plot:
    
    def __init__(self, opts, Video, grid, step):
        print('initialize plotting...')
        self.ini_colorbar = False
        
        # load ground truth data
        self.D_groundTruth = Data.get_GroundTruth(opts, Video, grid, step = step)
        self.D_groundTruth[self.D_groundTruth>35] = np.nan
        
        self.color1 = 'tab:blue'
        self.color2 = 'tab:red'
        color3  = 'tab:green' 
        color4  = 'tab:brown'          
        color5  = 'tab:orange'
        self.fig    = plt.figure(figsize=(3,5))
        mng         = plt.get_current_fig_manager()
        mng.window.showMaximized()
            
        self.ax1    = self.fig.add_subplot(3,5,(1,6))
        self.ax2    = self.fig.add_subplot(3,5,(2,7))   
        self.ax3    = self.fig.add_subplot(3,5,(3,8)) 
        self.ax4    = self.fig.add_subplot(3,5,(4,9)) 
        self.ax5    = self.fig.add_subplot(3,5,(5,10))  
        self.ax6    = self.fig.add_subplot(3,5,12)              
        self.ax6b   = self.ax6.twinx()
        
        self.ax7    = self.fig.add_subplot(3,5,13)
        self.ax8    = self.fig.add_subplot(3,5,14)
        self.ax9    = self.fig.add_subplot(3,5,15)
        self.ax10   = self.fig.add_subplot(3,5,11)
        
             
        self.d_rmse     = np.array([])
        self.d_mrmse    = np.array([])
        self.d_bias     = np.array([])
        
        self.iter_now   = 0
        self.itern      = np.array([]) 
        self.t_iter     = np.array([]) 
# =============================================================================
#         ax3     = fig.add_subplot(3,5,(3,8))        
#         ax3b    = fig.add_subplot(3,5,(4,9))        
#         ax4     = fig.add_subplot(3,5,(5,10))   
# =============================================================================
    def results(self, opts, grid, Results, Kalman, InvStg, d_lims, diff_lims, err_lims, gc_kernel, fft_spectrum, dmd_spectrum, t_shift):
        print('plot update...')
        self.iter_now   = self.iter_now + 1      
        self.itern      = np.append(self.itern,self.iter_now)
        self.t_iter     = np.append(self.t_iter,t_shift)
        
        self.D              = np.reshape(Kalman.derrt_prev[0,:],(grid.Numrows,grid.Numcols),order = 'F')
        self.D_diff_stat    = self.D_groundTruth - self.D
        self.d_bias         = np.append(self.d_bias,np.nanmean(self.D_diff_stat))
        self.d_rmse         = np.append(self.d_rmse,np.sqrt(np.nanmean(self.D_diff_stat**2)))
        D_diff_stat2        = self.D_diff_stat-np.nanmean(self.D_diff_stat)
        self.d_mrmse        = np.append(self.d_mrmse,np.sqrt(np.nanmean(D_diff_stat2**2)))
        
        dKerrors    = Kalman.derrt_prev[1,:]
        dKerrors    = np.reshape(dKerrors, (grid.Numrows,grid.Numcols), order = 'F')

        u   = np.reshape(Kalman.uerrt_prev[0,:],(grid.Numrows,grid.Numcols), order = "F")
        v   = np.reshape(Kalman.verrt_prev[0,:],(grid.Numrows,grid.Numcols), order = "F")
        
        #n layers
        n_used_layers = [np.mean([len(lis) for lis in line]) for line in InvStg.omega_store]
#         #n_used_layers = [len(line[-1]) for line in InvStg.omega_store]
        n_used_layers = np.reshape(n_used_layers, (grid.Numrows,grid.Numcols), order = 'F')

        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax5.cla()
        self.ax6.cla()
        
        self.ax7.cla()
        self.ax8.cla()
        self.ax9.cla()
        self.ax10.cla()
# =============================================================================
#         if self.iter_now % 2 == 1:
# =============================================================================
        im1     = self.ax1.imshow(self.D,extent = [grid.X[0,0],grid.X[0,-1],grid.Y[0,0],grid.Y[-1,0]],cmap = 'jet_r', origin = 'lower') #'terrain_r'                    
        #im1     = self.ax1.imshow(self.D_groundTruth,extent = [grid.X[0,0],grid.X[0,-1],grid.Y[0,0],grid.Y[-1,0]],cmap = 'jet_r', origin = 'lower') #'terrain_r'        
        CS1      = self.ax1.contour(grid.X, grid.Y, self.D_groundTruth, levels = [0.5,2,3.5,5,7.5,10,12.5,15],colors=('k',),linewidths=(1.5,))
        self.ax1.clabel(CS1, CS1.levels, fontsize=14,fmt = '%.1f')#, inline=True
# =============================================================================
#         else:
#             im1     = self.ax1.imshow(self.D,extent = [grid.X[0,0],grid.X[0,-1],grid.Y[0,0],grid.Y[-1,0]],cmap = 'jet_r', origin = 'lower') #'terrain_r'        
# =============================================================================
            
        im2     = self.ax2.imshow(self.D_diff_stat,extent = [grid.X[0,0],grid.X[0,-1],grid.Y[0,0],grid.Y[-1,0]],cmap = 'Spectral_r', origin = 'lower') 
        CS2      = self.ax2.contour(grid.X, grid.Y, self.D_diff_stat, levels = [-1.5,-0.5,0.5,1.5],colors=('k',),linewidths=(1.5,))
        self.ax2.clabel(CS2, CS2.levels, fontsize=14,fmt = '%.1f')#, inline=True

        im3     = self.ax3.imshow(dKerrors,extent = [grid.X[0,0],grid.X[0,-1],grid.Y[0,0],grid.Y[-1,0]], cmap = 'Spectral_r', origin = 'lower')
# =============================================================================
#         im4     = self.ax4.imshow(n_used_layers,extent = [grid.X[0,0],grid.X[0,-1],grid.Y[0,0],grid.Y[-1,0]], cmap = 'Spectral_r', origin = 'lower')
# =============================================================================
        density = 2
        norm    = Normalize(vmin=0, vmax=len(Results.c_omega))
        skip=(slice(None,None,round(grid.Numrows/30*density)),slice(None,None,round(grid.Numcols/30*density)))
        for ii in range(len(Results.c_omega)):
            cc = cm.jet(norm(ii),bytes=True)
            im4     = self.ax4.quiver(grid.X[skip], grid.Y[skip],
                                      np.reshape(Kalman.cxerrt_prev[0,:,ii], (grid.Numrows,grid.Numcols), order = 'F')[skip],
                                      np.reshape(Kalman.cyerrt_prev[0,:,ii], (grid.Numrows,grid.Numcols), order = 'F')[skip],
                                      color = (cc[0]/255,cc[1]/255,cc[2]/255,cc[3]/255))

        im5     = self.ax5.streamplot(grid.X[0,:], grid.Y[:,0], u, v, color=np.sqrt(u**2+v**2), linewidth=1, cmap= 'jet', density=2, arrowstyle='->', arrowsize=1.5)
        
        im6     = self.ax6.plot(self.itern,self.d_mrmse, color=self.color1) 
        im6b    = self.ax6b.plot(self.itern,self.d_bias, color=self.color2)
        
        if np.all(np.isnan(self.D_groundTruth)):
            im7 = self.ax7.scatter(np.nan,np.nan)
        else:
            im7 = self.ax7.scatter(self.D_groundTruth[~np.isnan(self.D_groundTruth)],self.D[~np.isnan(self.D_groundTruth)], 1.5, color=self.color2)
        self.ax7.plot(d_lims, d_lims)
        
        gc_kernel = gc_kernel.astype(float)*-1
        gc_kernel[int(gc_kernel.shape[0]/2),int(gc_kernel.shape[1]/2)] = 1#np.sum(gc_kernel)
        gc_kernel[gc_kernel == 0] = np.nan
        im8     = self.ax8.imshow(gc_kernel, cmap = 'jet', origin = 'lower')    
        
        fft_freqlims_ID = (fft_spectrum[1]/(2*np.pi) > opts.freqlims[1]) & (fft_spectrum[1]/(2*np.pi) < opts.freqlims[0])
        im9     = self.ax9.plot(fft_spectrum[1][fft_freqlims_ID],fft_spectrum[0][fft_freqlims_ID],'s-',color = (0.4,0.9,0)) 
        im9b    = self.ax9.plot(dmd_spectrum[1],dmd_spectrum[0],'*-',color = (1.0,0.5,0))
        
        im10    = self.ax10.stem(self.itern,self.t_iter)#,'|-',color = (0.4,0.9,0)
        
        im1.set_clim(d_lims)
        im2.set_clim(diff_lims)
        im3.set_clim(err_lims)
        #im5.set_clim([0, opts.Ulim])        
        
        self.ax7.set_ylim(d_lims)
        self.ax7.set_xlim(d_lims)    
        
        self.ax1.axis('equal');self.ax1.axis('tight');self.ax1.set_title('d [m]'); self.ax1.set_ylabel('y [m]'); self.ax1.set_xlabel('x [m]')
        self.ax2.axis('equal');self.ax2.axis('tight');self.ax2.set_title('diff. [m]'); self.ax2.set_ylabel('y [m]'); self.ax2.set_xlabel('x [m]')  
        self.ax3.axis('equal');self.ax3.axis('tight');self.ax3.set_title('Kalman d-error [m]'); self.ax3.set_ylabel('y [m]'); self.ax3.set_xlabel('x [m]') 
        self.ax4.axis('equal');self.ax4.axis('tight');self.ax4.set_title('c(T) [m/s]'); self.ax4.set_ylabel('y [m]'); self.ax4.set_xlabel('x [m]') 
        self.ax5.axis('equal');self.ax5.axis('tight');self.ax5.set_title('U [m/s]'); self.ax5.set_ylabel('y [m]'); self.ax5.set_xlabel('x [m]') 
        self.ax6.set_title('med.bias(red) and IQR(blue)'); self.ax6.set_ylabel('IQR [m]');self.ax6b.set_ylabel('med.bias [m]'); self.ax6.set_xlabel('update [#]')
        self.ax7.set_title('Direct comp.'); self.ax7.set_ylabel('d_inv [m]'); self.ax7.set_xlabel('d_meas [m]')
        self.ax8.axis('equal');self.ax8.axis('tight');self.ax8.set_title('SSPC sample locs.'); self.ax8.set_ylabel('gc_y [-]'); self.ax8.set_xlabel('gc_x [-]') 
        self.ax9.axis('equal');self.ax9.axis('tight');self.ax9.set_title('FFT(green) vs. DMD(orange) spectra'); self.ax9.set_ylabel('A [norm. intensity]'); self.ax9.set_xlabel('omega [rad/s]') 
        self.ax10.axis('equal');self.ax10.axis('tight');self.ax10.set_title('CPU time per update'); self.ax10.set_ylabel('t-CPU [s]'); self.ax10.set_xlabel('update [#]') 
        
        if not self.ini_colorbar:
            plt.colorbar(im1,ax = self.ax1)
            plt.colorbar(im2,ax = self.ax2)
            plt.colorbar(im3,ax = self.ax3)
            plt.colorbar(im4,ax = self.ax4)
            plt.colorbar(im5.lines,ax = self.ax5)
# =============================================================================
#             plt.colorbar(im4,ax = self.ax4)
# =============================================================================
            self.ini_colorbar = True            
# =============================================================================
#         plt.pause(1.0);plt.pause(0.05)     
#         
#         self.ax2.cla()
#         im2     = self.ax2.imshow(self.D_diff_stat,extent = [grid.X[0,0],grid.X[0,-1],grid.Y[0,0],grid.Y[-1,0]],cmap = 'Spectral_r', origin = 'lower') #'terrain_r'                
#         im2.set_clim(diff_lims)
#         self.ax2.axis('equal');self.ax2.axis('tight');
# =============================================================================
        plt.pause(0.05)
        plt.tight_layout()
        plt.pause(0.05)