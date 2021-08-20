# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:13:18 2020

@author: gawehn
"""

import numpy as np
import time
from collections import namedtuple
from joblib      import Parallel, delayed

class Kalman():
    __init_counter = 0    
    def __init__(self, opts, grid): 
        print('initialize Kalman filter...')
        self.Ngc    = grid.X.shape[0]*grid.X.shape[1]
        Tmin        = 1/max(opts.freqlims)
        Tmax        = 1/min(opts.freqlims)
        self.Nw     = int((Tmax-Tmin+1)/opts.cxy_deltaT)
        type(self).__init_counter += 1     #use type(self) instead of DMD.counter because DMD has subclass OptDMD

    def Filter(self, opts, Results, t):
        print('Kalman filter d,u,v,cx,cy...', end =" ")
        start = time.time()
        if type(self).__init_counter == 1:
            tgc_prev_d  = np.ones(self.Ngc)*t
            tgc_prev_u  = np.ones(self.Ngc)*t
            tgc_prev_v  = np.ones(self.Ngc)*t 
            tgc_prev_cx = np.ones((self.Ngc,self.Nw))*t 
            tgc_prev_cy = np.ones((self.Ngc,self.Nw))*t  
            
            dk_prev     = Results.d # depth            
            uk_prev     = Results.u # u            
            vk_prev     = Results.v # v
            
            cxk_prev    = Results.cx# cx            
            cyk_prev    = Results.cy# cy
            
            if opts.R_type == 'fit_conf95':
                dk_err_prev     = Results.conf95[:,0]
                uk_err_prev     = Results.conf95[:,1]
                vk_err_prev     = Results.conf95[:,2]
                cxk_err_prev    = Results.var_cxcy[:,:,0]
                cyk_err_prev    = Results.var_cxcy[:,:,1]
            elif opts.R_type == 'x_diff2':
                dk_err_prev     = np.ones(self.Ngc)
                uk_err_prev     = np.ones(self.Ngc)
                vk_err_prev     = np.ones(self.Ngc)
                cxk_err_prev    = np.ones((self.Ngc,self.Nw))
                cyk_err_prev    = np.ones((self.Ngc,self.Nw))
                
            self.derrt_prev  = np.vstack([dk_prev, dk_err_prev, tgc_prev_d])
            self.uerrt_prev  = np.vstack([uk_prev, uk_err_prev, tgc_prev_u])
            self.verrt_prev  = np.vstack([vk_prev, vk_err_prev, tgc_prev_v])
            self.cxerrt_prev = np.reshape(np.vstack([cxk_prev, cxk_err_prev, tgc_prev_cx]),(3,self.Ngc,self.Nw), order = 'C')
            self.cyerrt_prev = np.reshape(np.vstack([cyk_prev, cyk_err_prev, tgc_prev_cy]),(3,self.Ngc,self.Nw), order = 'C')
            type(self).__init_counter = 0
        else:               
            #apply Kalman filter to each gridpoint in time
            #----------------------------------------------            
            for gc in range(self.Ngc):#loop through gridcells
               self.derrt_prev[:,gc] = self.gc_walk_filter_online(gc, opts, Results.d[gc], self.set_R(gc, opts, Results, 'd'), self.derrt_prev[0,gc], self.derrt_prev[1,gc], opts.Q_d, t, self.derrt_prev[2,gc])
            for gc in range(self.Ngc):    
               self.uerrt_prev[:,gc] = self.gc_walk_filter_online(gc, opts, Results.u[gc], self.set_R(gc, opts, Results, 'u'), self.uerrt_prev[0,gc], self.uerrt_prev[1,gc], opts.Q_U, t, self.uerrt_prev[2,gc])
            for gc in range(self.Ngc):    
               self.verrt_prev[:,gc] = self.gc_walk_filter_online(gc, opts, Results.v[gc], self.set_R(gc, opts, Results, 'v'), self.verrt_prev[0,gc], self.verrt_prev[1,gc], opts.Q_U, t, self.verrt_prev[2,gc])            
            for gc in range(self.Ngc):   
                for c_ii in range(self.Nw):  
                    self.cxerrt_prev[:,gc,c_ii] = self.gc_walk_filter_online(gc, opts, Results.cx[gc,c_ii], self.set_R(gc, opts, Results, 'cx', c_ii), self.cxerrt_prev[0,gc,c_ii], self.cxerrt_prev[1,gc,c_ii], opts.Q_C, t, self.cxerrt_prev[2,gc,c_ii])    
            for gc in range(self.Ngc):   
                for c_ii in range(self.Nw):  
                    self.cyerrt_prev[:,gc,c_ii] = self.gc_walk_filter_online(gc, opts, Results.cy[gc,c_ii], self.set_R(gc, opts, Results, 'cy', c_ii), self.cyerrt_prev[0,gc,c_ii], self.cyerrt_prev[1,gc,c_ii], opts.Q_C, t, self.cyerrt_prev[2,gc,c_ii])               
        end  = time.time() 
        print('CPU time: {} s'.format(np.round((end-start)*100)/100))  
        
    def set_R(self, gc, opts, Results, label, c_ii = None): 
        if opts.R_type == 'fit_conf95': # error estimate
            if label == 'd':
                R   = Results.conf95[gc,0]
            if label == 'u':  
                R   = Results.conf95[gc,1]
            if label == 'v':
                R   = Results.conf95[gc,2]
            if label == 'cx':
                R   = Results.var_cxcy[gc,c_ii,0]   
            if label == 'cy':
                R   = Results.var_cxcy[gc,c_ii,1]     
        elif opts.R_type == 'x_diff2':
            if label == 'd':
                R   = np.abs(Results.d[gc]-self.dk_prev[gc])#**2
            if label == 'u':  
                R   = np.abs(Results.u[gc]-self.uk_prev[gc])
            if label == 'v':
                R   = np.abs(Results.v[gc]-self.vk_prev[gc])     
            if label == 'cx':
                R   = np.abs(Results.cx[gc,c_ii]-self.cxk_prev[gc,c_ii])
            if label == 'cy':
                R   = np.abs(Results.cy[gc,c_ii]-self.cyk_prev[gc,c_ii])     
        return R        
    
    def gc_walk_filter_online(self, gc, opts, x_gc, R, xhat_prev_gc, xhat_err_prev_gc, Q, t, tgc_prev_gc):
      
        # fix case where nans precede the first valid estimate
        # --> intialize the Kalman proces for this previously bad grid cell
        if np.isnan(xhat_prev_gc): 
            xhat_gc     = x_gc # becomes xhat_prev at the end
            if opts.R_type == 'fit_conf95': 
                if np.isinf(R): 
                    R = 500
                if R == 0:
                    R = 0.0000001
                xhat_err_prev_gc   = R
            elif opts.R_type == 'x_diff2':
                xhat_err_prev_gc   = 2.0    
            tgc_prev_gc = t
        # fix case where the new case is NaN (also R can be NaN independent of x, 
        # but in that case we assume the fit was bad, otherwise we would have had an estimate for R)
        # Replace this (bad) NaN case with previous values (i.e., assume this measurement 
        # is exactly the same as the previous one)
        elif (np.isnan(x_gc) | np.isnan(R)):                 
            xhat_gc             = xhat_prev_gc
            xhat_err_prev_gc    = xhat_err_prev_gc #remains unchanged
            tgc_prev_gc         = tgc_prev_gc
        # Kalman filter
        else: 
            if np.isinf(R): 
                R = 500 
            if R == 0:
                R = 0.0000001    
            if np.isinf(xhat_err_prev_gc):
                xhat_err_prev_gc = 500     
                
            P                   = xhat_err_prev_gc   
            # variance updating based on unmodeled changes
            Pt                  = P + Q*(t-tgc_prev_gc)
            # Kalman gain based on variance            
            K                   = Pt/(Pt + R)
            if (K == 0) | (np.isinf(K)) | (np.isnan(K)): breakpoint()
            # Kalman filter updating
            xhat_gc             = xhat_prev_gc + K*(x_gc - xhat_prev_gc);
            # estimated variance is updated based on Kalman gain
            xhat_err_prev_gc    = (1-K)*Pt; # P  --> replace old P by updated P
            # store result
            tgc_prev_gc         = t
        
        # use xhat as previous estimate xhat_prev in next Kalman iteration        
        xhat_prev_gc = xhat_gc
        
        return np.array([xhat_prev_gc, xhat_err_prev_gc, tgc_prev_gc])