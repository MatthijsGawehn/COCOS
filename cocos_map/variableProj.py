# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 16:05:26 2017

@author: JamesMichael  https://github.com/kunert/py-optDMD
adapted by Matthijs Gawehn : 
    - increase computational speed
    - include Thikonov regularization as of Askhams most recent MATLAB code https://github.com/askhamwhat
    - make simpler
    - transformed into class with class methods
"""
import numpy as np
#import scipy as sci
#from scipy.sparse import csc_matrix
#from scipy.sparse import lil_matrix
from scipy.linalg import qr, svd
#from fbpca import pca as svd
#from fbpca import mult
from copy import copy

import sys,pdb
import matplotlib.pyplot as plt
import time

class Varpro():
    def __init__(self,
                 lambda0        = 1.0,
                 maxlam         = 52,
                 lamup          = 2.0,
                 marq_flag      = True,
                 maxiter        = 30,
                 tol            = 1.0e-6,
                 eps_stall      = 1.0e-9,
                 fulljac_flag   = False,
                 tikh_flag      = False,
                 gamma          = 1,
                 proxfun_flag   = False #not yet implemented
                 ):
        print('      opt DMD with max # iterations = {}'.format(maxiter))
        print('      opt DMD with full jacobian = {}'.format(fulljac_flag))
        
        self.checkinputrange('lambda0',lambda0,0.0,1.0e16)
        self.checkinputrange('maxlam',maxlam,0,200)
        self.checkinputrange('lamup',lamup,1.0,1.0e16)
        self.checkinputrange('maxiter',maxiter,0,1e12)
        self.checkinputrange('tol',tol,0,1e16)
        self.checkinputrange('eps_stall',eps_stall,-np.Inf,np.Inf)
        self.lambda0        = float(lambda0)
        self.maxlam         = int(maxlam)
        self.lamup          = float(lamup)
        self.marq_flag      = bool(marq_flag)
        self.maxiter        = int(maxiter)
        self.tol            = float(tol)
        self.eps_stall      = float(eps_stall)
        self.fulljac_flag   = bool(fulljac_flag)
        self.tikh_flag      = bool(tikh_flag)
        self.gamma          = float(gamma)
        self.proxfun_flag   = bool(proxfun_flag) 
        
        
    def unpack(self):
        return self.lambda0, self.maxlam, self.lamup, self.marq_flag, self.maxiter, self.tol, self.eps_stall, self.fulljac_flag, self.tikh_flag, self.gamma, self.proxfun_flag

    def backslash(self,A,B):
        x=[]
        for k in range(B.shape[1]):
            b   = B[:,k][:,None]
            x.append(np.linalg.lstsq(A,b,rcond=None)[0])
        return np.hstack(x)
    
    def varpro2expfun(self,alphaf,tf):
        phimat = np.exp(np.reshape(tf,(-1,1)) @ (np.reshape(alphaf,(1,-1))))
        phimat[np.isinf(phimat)] = 0
        return phimat
    
    def varpro2dexpfun(self,alphaf,tf,i):
        m       = tf.size
        n       = alphaf.size
        if (i<0)|(i>=n):
            raise Exception('varpro2dexpfun: i outside of index range for alpha')
        A       = np.zeros((m,n),dtype=complex) #lil_matrix to make faster?!
        A[:,i]  = tf*np.exp(alphaf[i]*tf)
        return A
    
      
    def checkinputrange(self,xname,xval,xmin,xmax):
        if xval>xmax:
            print('Option {:} with value {:} is greater than {:}, which is not recommended'.format(xname,xval,xmin,xmax))
        if xval<xmin:
            print('Option {:} with value {:} is less than {:}, which is not recommended'.format(xname,xval,xmin,xmax))
    
    def proxfun(self,alpha):
        return 1j*np.imag(alpha)
    
    def varpro_LM(self, Xvid_T, t, m = None, n = None, iss = None, ia = None, alpha_init = None):
        
        printRuntimes   = False
        sigmaRankReduc  = True
        print('      use rank reduction: {}'.format(sigmaRankReduc))
        
        if self.tikh_flag:
            self.gamma  = self.gamma*np.eye(ia)
        else:
            self.gamma  = np.zeros(ia)
        self.gamma = self.gamma.astype(complex)
        
        #initialize values
        start   = time.time()
        t       = t-t[0]
        alpha       = copy(alpha_init)
        if self.proxfun_flag:
            alpha       = self.proxfun(alpha)
        alphas      = np.zeros((len(alpha),self.maxiter)).astype(complex)
        if self.tikh_flag:
            djacmat = np.zeros((m*iss+ia,ia)).astype(complex)
            rhstemp = np.zeros(m*iss+ia,).astype(complex)
        else:
            djacmat     = np.zeros((m*iss,ia)).astype(complex)
            rhstemp     = np.zeros(m*iss,).astype(complex)  
            
        err         = np.zeros(self.maxiter)
        scales      = np.zeros(ia)        
        rjac        = np.zeros((2*ia,ia)).astype(complex) #*2 because the second half in calculation of delta is occupied with minimization of vM. --> which enforces a minimization of the step size delta
        phimat      = self.varpro2expfun(alpha,t)
        U,S,Vh      = svd(phimat,full_matrices=False)
        
        S           = np.diag(S); sd = np.diag(S)    
        
        tolrank     = m*np.finfo(float).eps
        if sigmaRankReduc: 
            tolrank     = m*np.finfo(float).eps
            irank       = np.sum(sd>(tolrank*sd[0]))   
            #irank       = irank + irank % 2 #added Matthijs
        else:
            irank       = np.sum(sd>0) 
# =============================================================================
#         if irank < np.sum(sd>0):
#             print('RANK REDUCED TO {}'.format(irank))
# =============================================================================
                
        U           = U[:,:irank]
        S           = S[:irank,:irank]
        #V           = V[:,:irank].T
        #V           = Vh.T[:,:irank]
        V           = Vh.conj().T[:,:irank]
        
        b           = self.backslash(phimat,Xvid_T) 
        
        res         = Xvid_T-(phimat @ b)
        
        err_semi    = np.linalg.norm(res,'fro')**2      #semi norm
        err_side    = np.linalg.norm(self.gamma*alpha)**2    #side constraint
        errlast     = 0.5*(err_semi + err_side)         # residual norm
        w           = np.ones(len(res.ravel()))
        
        imode       = 0
        end = time.time()
        if printRuntimes:print(1,end-start)
        
        for itern in range(self.maxiter):
            #build jacobian matrix, looping over alpha indices
            start = time.time()
            for j in range(ia):
                dphitemp    = self.varpro2dexpfun(alpha,t,j).astype(complex)
                djaca       = (dphitemp-(U@(U.T.conj()@dphitemp))) @ b
                if self.fulljac_flag:
                    #use full expression for jacobian
                    #djacb           = U.dot(backslash(S,V.T.conj().dot(dphitemp.T.conj().dot(res))))                
                    #djacb           = U @ (backslash(S,V.T.conj() @ (dphitemp.T.conj() @ res)))
                    Sinv                = np.diag(1/np.diag(S)) 
                    djacb               = U @ (Sinv @ (V.T.conj() @ (dphitemp.T.conj() @ res))) # --> omitting backslash is MUCH faster, factor 10-50
                    djacmat[:m*iss,j]   = w*(djaca.ravel(order='F')+djacb.ravel(order='F'))
                else:
                    djacmat[:m*iss,j]    = w*(djaca.ravel(order = 'F'))
                scales[j]   = 1.0
                if self.marq_flag:
                    scales[j]   = np.minimum(np.linalg.norm(djacmat[:,j]),1.0)
                    scales[j]   = np.maximum(scales[j],1.0e-6)
            end = time.time()
            if printRuntimes:print(2,end-start,itern)
            
            if self.tikh_flag: # HERE tikhonov is added to Jacobian. As [J;L] where here L == gamma matrix
                djacmat[m*iss:,:] = self.gamma 
                    
            #loop to determine lambda (lambda gives the levenberg part)
            #pre-compute components which don't depend on step-size (lambda)
            #get pivots and lapack-style qr for jacobian matrix
            start = time.time()
            rhstemp[0:m*iss] = w*res.ravel(order='F')
            
            if self.tikh_flag:
                rhstemp[m*iss:] = -self.gamma @ alpha 
                
            g                   = djacmat.conj().T @ (rhstemp)
            
            qout,djacout,jpvt   = qr(djacmat, pivoting=True, mode='economic')
            ijpvt               = np.arange(ia)
            ijpvt[jpvt]         = [ijpvt]
            rjac[:ia,:]         = np.triu(djacout[:ia,:])
            #rjac                = np.triu(djacout[:ia,:])  #replace above with this line, if concatenation is used
            rhstop              = qout.conj().T @ rhstemp   # Q'*P (where P == res not orth. projection)
            scalespvt           = scales[jpvt[:ia]]         # permute scales appropriately...
            rhs                 = np.concatenate((rhstop[:ia], np.zeros(ia,).astype(complex)), axis=0)
            rjac[ia:2*ia,:]     = self.lambda0*np.diag(scalespvt).astype(complex)
                        
            #D       = self.lambda0*np.diag(scalespvt).astype(complex)
            #rjac    = np.concatenate((rjac,D),0)
            
            delta0  = self.backslash(rjac, np.expand_dims(rhs, axis=1))
            delta0  = np.squeeze([delta0[ijpvt]]) #######check if this really works 
    
            alpha0  = alpha.ravel()+delta0.ravel()
            if self.proxfun_flag:
                alpha0  = self.proxfun(alpha0)
                delta0  = alpha0-alpha.ravel()
            
            end = time.time()
            if printRuntimes:print(3,end-start,itern)
           
            start = time.time()
            phimat  = self.varpro2expfun(alpha0,t)
            b0      = self.backslash(phimat,Xvid_T)
            res0    = Xvid_T-phimat @ b0
            err0_semi   = np.linalg.norm(res0,'fro')**2         # semi norm
            err0_side   = np.linalg.norm(self.gamma*alpha0)**2  # side constraint
            err0        = 0.5*(err0_semi + err0_side)           # residual norm

    		#predicted improvement vs actual improvement
            act_impr    = errlast-err0
            pred_impr   = np.real(0.5*delta0.conj().T @ g) 
            impr_rat    = act_impr/pred_impr
            end = time.time()
            if printRuntimes:print(4,end-start,itern)
            #check if this is an improvement
            if err0 < errlast:
                # rescale lambda based on actual vs pred improvement      
                self.lambda0    = self.lambda0*np.max([1.0/3.0,1-(2*impr_rat-1)**3])
                alpha           = copy(alpha0)
                errlast         = copy(err0)
                err_semi        = copy(err0_semi)#semi norm
                err_side        = copy(err0_side) #side constraint
                b       = copy(b0)
                res     = copy(res0)
            else:
                #if not, increase lambda until something works
                #this makes the algorithm more like gradient descent
                start = time.time()
                for j in range(self.maxlam):
                    self.lambda0    = self.lambda0*self.lamup               
                    rjac[ia:2*ia,:] = self.lambda0*np.diag(scalespvt).astype(complex)
                    delta0  = self.backslash(rjac, np.expand_dims(rhs, axis=1))
                    delta0  = np.squeeze([delta0[ijpvt]]) #######check if this really works 
                    
                    alpha0  = alpha.ravel()+delta0.ravel()
                    if self.proxfun_flag:
                        alpha0  = self.proxfun(alpha0)
                        delta0  = alpha0-alpha.ravel()
                    
                    phimat      = self.varpro2expfun(alpha0,t)
                    b0          = self.backslash(phimat,Xvid_T)
                    res0        = Xvid_T-phimat @ b0
                    err0_semi   = np.linalg.norm(res0,'fro')**2     # semi norm
                    err0_side   = np.linalg.norm(self.gamma*alpha0)**2   # side constraint
                    err0        = 0.5*(err0_semi + err0_side)       # residual norm
                    if err0 < errlast:
                        break
                end = time.time()
                if printRuntimes:print(5,end-start,itern)
                if err0 < errlast:
                    alpha   = copy(alpha0)
                    errlast = copy(err0)
                    err_semi    = copy(err0_semi)#semi norm
                    err_side    = copy(err0_side) #side constraint
                    b       = copy(b0)
                    res     = copy(res0)
                else:
                    #no appropriate step length found
                    niter       = itern
                    err[itern]  = errlast
                    imode       = 4
                    print('      failed to find appropriate step length at iteration {:}\n Current residual {:}'.format(itern,errlast))
                    return b, alpha, niter, err, imode, alphas, err_semi, err_side, res, irank
            
            alphas[:,itern] = alpha
            err[itern]      = errlast
            
            #print('step {:} err {:} lambda {:}\n'.format(itern,errlast,self.lambda0))
            
            if errlast<self.tol:
                #tolerance met
                niter   = itern
                return b, alpha, niter, err, imode, alphas, err_semi, err_side, res, irank
            if itern>0:
                #stall detected
                if (err[itern-1]-err[itern] < self.eps_stall*err[itern-1]):
                    niter   = itern
                    imode   = 8
                    print('      stall detected: residual reduced by less than {:} times residual at previous step. \n      iteration {:} \n      current residual {:}'.format(self.eps_stall, itern, errlast))
                    return b, alpha, niter, err, imode, alphas, err_semi, err_side, res, irank
            phimat      = self.varpro2expfun(alpha,t)
            U,S,Vh      = svd(phimat, full_matrices = False)
            S           = np.diag(S); sd = np.diag(S)
            tolrank     = m*np.finfo(float).eps
            if sigmaRankReduc: 
                irank   = np.sum(sd > (tolrank*sd[0]))
                #irank   = irank + irank % 2 #added Matthijs
            else:
                irank   = np.sum(sd > 0)
# =============================================================================
#             if irank < np.sum(sd>0):
#                 print('RANK REDUCED TO {}'.format(irank))     
# =============================================================================
                
            U       = U[:,:irank]
            S       = S[:irank,:irank]
            #V       = V[:,:irank].T
            #V       = Vh.T[:,:irank]
            V       = Vh.conj().T[:,:irank]
            
        #only get here if failed to meet tolerance in maxiter steps
        niter   = self.maxiter
        imode   = 1
        print('      failed to reach tolerance after maxiter={:} iterations \n current residual {:}'.format(self.maxiter,errlast))
        return b, alpha, niter, err, imode, alphas, err_semi, err_side, res, irank  
                                  
# =============================================================================
# def timer(calculation):            
#     start = time.time()
#     res = calculation
#     end = time.time()
#     print(end-start)        
#     return res 
# =============================================================================

