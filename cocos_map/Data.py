# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:49:26 2020

@author: gawehn
"""
import numpy      as np
import mat73
import time

from collections  import namedtuple
from scipy.io     import loadmat
from scipy        import interpolate,integrate
from scipy.signal import detrend, hilbert
from matplotlib import path

class Data():
    def __init__():
        pass

    #@staticmethod
    def inpoly(xq, yq, xv, yv):
        shape   = xq.shape
        xq      = xq.reshape(-1)
        yq      = yq.reshape(-1)
        xv      = xv.reshape(-1)
        yv      = yv.reshape(-1)
        q       = [(xq[i], yq[i]) for i in range(xq.shape[0])]
        p       = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
        
        return p.contains_points(q).reshape(shape)
    
    def set_Video(X, Y, ImgSequence, m, n, l, dx, dt, label):
        VideoStruct     = namedtuple('struct', ['X', 'Y', 'ImgSequence', 'm', 'n', 'l', 'dx', 'dt', 'label'])
        Video           = VideoStruct(X, Y, ImgSequence, m, n, l, dx, dt, label)
        return Video
    
    @classmethod
    def get_Video(cls,label, startx = None, stopx = None, starty = None, stopy = None, step = None):
        print('load video data...', end =" ")
        start = time.time()
        if label == 'duck': # define label
            if step is None: step = 1 #set default step
            DuckArgus   = loadmat('C:/Users/gawehn/OneDrive - Stichting Deltares/Desktop/PhD/data/ArgusStack/Interp_testStack102210Duck.mat')
            X           = DuckArgus['XX']
            Y           = DuckArgus['YY']
            ImgSequence = DuckArgus['TimeStack']
            dx          = DuckArgus['dx'][0][0]
            dt          = DuckArgus['dt'][0][0]      
        if label == 'scheveningen':   
            if step is None: step = 1 #set default step
            SchevDrone  = np.load('c:/Users/gawehn/OneDrive - Stichting Deltares/Desktop/PhD/data/Drone/voorMathijs/MatthijsProducts/Rect_cutoff_DJI_0001_fps_2_cv1150_big_dx2.npz')#'Rect_DJI_0001_fps_2_cv1150_big_dx2.npz')#Rect_cutoff_DJI_0001_fps_2_cv1150_big_dx2
            X           = SchevDrone['X']
            Y           = SchevDrone['Y']
            ImgSequence = SchevDrone['RectMov_gray']
            dx          = SchevDrone['dx']
            dt          = SchevDrone['dt']
        if label == 'narrabeen':      
            if step is None: step = 1 #set default step
            NarraDrone  = np.load('c:/Users/gawehn/OneDrive - Stichting Deltares/Desktop/PhD/data/Drone/Australia/Flight2_thesis/MatthijsProducts/Rect_new_DJI_0009_fps_2_noArrow_origLcpAndDistortCoeffs.npz')#'Rect_new_frame1070_DJI_0009_fps_2_onlyVosGcps.npz')#'Rect_new_DJI_0009_fps_2_noArrow_Flights12LcpAnd_d1_002.npz')#'Rect_new_DJI_0009_fps_2_noArrow_origLcpAndDistortCoeffs.npz')
            X           = NarraDrone['X']
            Y           = NarraDrone['Y']
            ImgSequence = NarraDrone['RectMov_gray']
            dx          = NarraDrone['dx']
            dt          = NarraDrone['dt']  
            mask        = ~cls.inpoly(X,Y,np.array([341000,342400,342700,341000]),np.array([626700,6267200,6270000,6270000]))
            for ii in range(ImgSequence.shape[2]):
                ImgSequence[:,:,ii] *= mask
        if label == 'porthtowan': 
            if step is None: step = 1 #set default step
            PortTArgus  = loadmat('c:/Users/gawehn/OneDrive - Stichting Deltares/Desktop/PhD/data/PorthTowan/Bathy_MAT_files2014/MatthijsProducts/1397121301.Thu.Apr.10_09_15_01.UTC.2014_porthtowan_gridded2_dx5_trunc.mat')#1397027701.Wed.Apr.09_07_15_01.UTC.2014_porthtowan_gridded2_dx5.mat(good:-0.944874362495324)#1397034901.Wed.Apr.09_09_15_01.UTC.2014_porthtowan_gridded2_dx5.mat(-0.258237984059989)#1397038502.Wed.Apr.09_10_15_02.UTC.2014_porthtowan_gridded2_dx5.mat(0.231167373468409)#1397049302.Wed.Apr.09_13_15_02.UTC.2014_porthtowan_gridded2_dx5.mat(1.024765570013350)#1397060101.Wed.Apr.09_16_15_01.UTC.2014_porthtowan_gridded2_dx5.mat(good:0.115154418969808)#'1397121301.Thu.Apr.10_09_15_01.UTC.2014_porthtowan_gridded2_dx5.mat'(very good:-0.964546845297460)#'1397128502.Thu.Apr.10_11_15_02.UTC.2014_porthtowan_gridded2_dx5'(0.231713186333518)#'1397139302.Thu.Apr.10_14_15_02.UTC.2014_porthtowan_gridded2_dx5.mat'(1.431498396339975)#1397142901.Thu.Apr.10_15_15_01.UTC.2014_porthtowan_gridded2_dx5.mat'(1.173524417485523)#'1397146501.Thu.Apr.10_16_15_01.UTC.2014_porthtowan_gridded2_dx5.mat'(very good: 0.601465679011026)#'1397150101.Thu.Apr.10_17_15_01.UTC.2014_porthtowan_gridded2_dx5.mat'(very good: -0.096778174716902)#
            X           = PortTArgus['X'][26:,:181]
            Y           = PortTArgus['Y'][26:,:181]
            ImgSequence = PortTArgus['ImgSequence'][26:,:181]
            dx          = PortTArgus['dx'][0][0]
            dt          = PortTArgus['dt'][0][0]     
            mask        = ~cls.inpoly(X,Y,np.array([0,300,482,0]),np.array([580,580,-300,-300]))
            for ii in range(ImgSequence.shape[2]):
                ImgSequence[:,:,ii] *= mask
        if label == 'capbreton':
            if step is None: step = 2 #set default step            
            CapbretSat  = loadmat('c:/Users/gawehn/OneDrive - Stichting Deltares/Desktop/PhD/data/Pleiades/Movie6.mat')   
            X           = CapbretSat['XX']
            Y           = CapbretSat['YY']
            ImgSequence = CapbretSat['TimeStack']
            dx          = CapbretSat['dx'][0][0]
            dt          = CapbretSat['dt'][0][0] 
            # demean per frame
            Nt          = ImgSequence.shape[2]
            isbad_IX    = (np.isnan(ImgSequence)) | (ImgSequence == 0)
            ImgSequence[isbad_IX] = np.nan
            for ii in range(Nt):
                ImgSequence[:,:,ii] = ImgSequence[:,:,ii] - np.nanmean(ImgSequence[:,:,ii])
            # detrend in time per pixle     
# =============================================================================
#             ImgSequence[isbad_IX] = 0
#             ImgSequence = detrend(ImgSequence, axis = -1, type = 'linear') 
#             ImgSequence[isbad_IX] = np.nan
#             # maxmin normalize in time per pixle 
#             Imgtime_min = np.nanmin(ImgSequence,-1)
#             Imgtime_max = np.nanmax(ImgSequence,-1)
#             for ii in range(Nt):
#                 ImgSequence[:,:,ii] = (ImgSequence[:,:,ii]-Imgtime_min)/(Imgtime_max-Imgtime_min)              
#             ImgSequence[(np.isnan(ImgSequence) | np.isinf(ImgSequence))] = np.nan     
# =============================================================================           
        if label == 'fig3':
            if step is None: step = 1 #set default step
            T   = 600
            dt  = 0.5
            m   = 128
            mult= m/128
            omegas   = np.array([1.75, 3.10, 1.0, 1.50, 4.2, 5.1])/2 #1.85
            IC  = np.array([0.5, -0.35, -0.70, -1.0, 0.55, 0.2])*m**2
            I   = (np.array([5, 6, 3, 5, 9, 12])*mult).astype(int)
            J   = (np.array([0, 3, 2, 2, 4, 5])*mult).astype(int)
            n   = int(T/dt)
            dx  = 1
            K   = 0 #reflection 
            #factor needed for identical amplitudes between Fouerier spectra of progressive and (partly) standing wave fields
            elliptic_integral2ndkind = lambda x,K: (1-4*K/(K+1)**2*np.sin(x)**2)**0.5 #integral of amplitude (i.e. sqrt(1+2K*sin(2phi)+K**2)) in eq 10 of Goda & Suzuki 1976 (neglecting the scaling factor |1+K| in integral, which represents the added amplitude/excursion due to reflected component)
            itg_standWave   = integrate.quad(elliptic_integral2ndkind,a = 0,b = 2*np.pi, args=(K,))
            itg_progWave    = 1*2*np.pi
            fac             = itg_progWave/itg_standWave[0]
            [X,Y] = np.meshgrid(np.linspace(0,m-1,m),np.linspace(0,m-1,m))            
            ImgSequence = np.zeros((m,m,n),dtype = float)
            for t_ID,t in enumerate(np.linspace(0,T,n)):# loop over time
                xtilde = np.zeros((m,m),dtype = 'complex128')
                for k in range(len(omegas)): # loop over waves
                    #wave incident
                    xtilde[I[k],J[k]] = np.exp(1j*omegas[k]*t)*(IC[k])*fac/(1+K)
                    #wave reflected
                    if J[k] == 0:
                        xtilde[m-I[k],0-J[k]] = np.exp(1j*omegas[k]*t)*(IC[k]*K)*fac/(1+K)
                    else:
                        xtilde[m-I[k],m-J[k]] = np.exp(1j*omegas[k]*t)*(IC[k]*K)*fac/(1+K)
                x = np.real(np.fft.ifft2(xtilde))
                ImgSequence[:,:,t_ID] = x  
                
        ix_x  = slice(startx,stopx,step)  
        ix_y  = slice(starty,stopy,step)  
        m,n,l = ImgSequence[ix_y,ix_x,:].shape
        
        end = time.time()
        print('CPU time: {} s'.format(np.round((end-start)*100)/100))
        return Data.set_Video(X[ix_y,ix_x], Y[ix_y,ix_x], ImgSequence[ix_y,ix_x,:], m, n, l, dx*step, dt, label)

    def get_GroundTruth(opts, Video, grid, step = None): 
        print('   load ground truth data...', end =" ")
        start = time.time()
        if Video.label == 'duck':
            WL              = 0.077
            CRABDuck        = loadmat('C:/svn_repo/CIRN/cBathy-Toolbox/19-Oct-2010FRFGridded.mat')
            [Xmeas,Ymeas]   = np.meshgrid(CRABDuck['xm'],CRABDuck['ym'])
            Z_groundTruth   = interpolate.griddata((np.ravel(Xmeas), np.ravel(Ymeas)), np.ravel(CRABDuck['zi']),(grid.X, grid.Y),method='linear')
            D_groundTruth   = -1*Z_groundTruth+WL
        elif Video.label == 'scheveningen':
            WL              = 0.6
            with open('C:/Users/gawehn/OneDrive - Stichting Deltares/Desktop/PhD/data/Drone/voorMathijs/bathy/ScheveningenJanBert_XYZ_RDNAP.txt') as f:
                list_of_lists   = [[x for x in line.split()] for line in f]
                flattened_list  = [y for x in list_of_lists for y in x]
            bathydata   = np.array(flattened_list[19:]).astype('float') 
            bathydata   = np.reshape(bathydata,(3, int(len(bathydata)/3)), order = 'F')
            bathyx  = bathydata[0,:]
            bathyy  = bathydata[1,:]
            bathyz  = bathydata[2,:]
            Z_groundTruth   = interpolate.griddata((bathyx, bathyy), bathyz,(grid.X, grid.Y),method='linear')   
            D_groundTruth   = -1*Z_groundTruth+WL                    
        elif Video.label == 'narrabeen':
            WL              = 0.67
            with open('c:/Users/gawehn/OneDrive - Stichting Deltares/Desktop/PhD/data/Drone/Australia/jetski_surveys/20170529/2017_0529 Narrabeen Post ECL No4 Hydro Depths.xyz') as f:
                list_of_lists   = [line.split() for line in f]
                bathy_xyz       = np.array(list_of_lists, dtype = float)
            Z_groundTruth   = interpolate.griddata((bathy_xyz[:,0], bathy_xyz[:,1]), bathy_xyz[:,2],(grid.X, grid.Y),method='linear')
            D_groundTruth   = Z_groundTruth+WL
        elif Video.label == 'porthtowan':
            WL              = -0.964546845297460
            ErwinPortT      = loadmat('c:/Users/gawehn/OneDrive - Stichting Deltares/Desktop/PhD/data/PorthTowan/PTB_01_09042014_int.mat')
            Xmeas,Ymeas     = ErwinPortT['Xi'],ErwinPortT['Yi']
            Z_groundTruth   = interpolate.griddata((np.ravel(Xmeas), np.ravel(Ymeas)), np.ravel(ErwinPortT['Zi']),(grid.X, grid.Y),method='linear')
            D_groundTruth   = -1*Z_groundTruth+WL    
        elif Video.label == 'capbreton':
            if step is None: step = 2
            WL              = 0.4
            CapbretSat      = loadmat('c:/Users/gawehn/OneDrive - Stichting Deltares/Desktop/PhD/data/Pleiades/Movie6.mat')   
            Z_groundTruth   = interpolate.griddata((np.ravel(Video.X), np.ravel(Video.Y)), np.ravel(CapbretSat['BathySurvey'][::step,::step]),(grid.X, grid.Y),method='nearest')
            D_groundTruth   = -1*Z_groundTruth+WL
        else: 
            print('No ground truth depth provided')
            D_groundTruth = np.nan       
        try:
            D_groundTruth[D_groundTruth<opts.dlims[0]] = np.nan
        except:
            print('no gorund truth')
            
        end = time.time()
        print('CPU time: {} s'.format(np.round((end-start)*100)/100))
        
        return D_groundTruth