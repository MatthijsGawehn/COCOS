# COCOS (COastal COmmunity Scout)
on-the-fly estimation of coastal parameters from (orthorectified) video of a wave field

This is code based on the publication "Gawehn, M.; de Vries, S.; Aarninkhof, S. A self-adaptive method for mapping coastal bathymetry on-the-fly from wave field video. Remote Sens. 2021, 1, 0. https://doi.org/10.3390/rs13234742"

Quick user guide
================

Optional
--------
Although not necessary you may start by downloading anaconda and the anaconda navigator (if you don't have those already). The anaconda navigator is an easy way to create a new environment for the COCOS code and it also enables you to easily open a command terminal within that environment.

Installation of COCOS
---------------------
install directly from PyPI:
1. make a new environment and give it a name e.g. cocos_env (if you use anaconda navigator >environments>create)
2. open terminal in that environment (in anaconda navigator > click on the new environment>press green circled arrow>open Terminal)
3. type >"python -m pip install cocos-map" (this installs all the necessary packages including cocos-map)
4. you currently cannot run cocos-map from the command prompt, instead you need to work with a Matlab-like file structure. This means that you need to navigate to the cocos_map site-package. It is located in the environment (cocos_env), which you've created and where you've just installed cocos-map (e.g., c:\Users\<yourname>\AppData\Local\Continuum\anaconda3\envs\<cocos_env>\Lib\site-packages\cocos_map).
5. copy the site package 'cocos_map'. Create an empty folder named 'COCOS_main' at your preferred location ['my_COCOS_location']. Paste 'cocos_map' into 'COCOS_main".
6. create two additional empty folders in 'COCOS_main' (on the same level as cocos_map). One named 'data' and another one named 'results'.

install from GitHub:
1. download the .zip to your preferred location ['my_COCOS_location'] and unzip
2. make a new environment and give it a name e.g. cocos_env (if you use anaconda navigator >environments>create)
3. open terminal in that environment (in anaconda navigator > click on the new environment>press green circled arrow>open Terminal)
4. navigate to 'my_COCOS_location' by typing "cd " followed by "my_COCOS_location". Be sure you navigate into the 'COCOS_main' folder (in which you see the setup.py)
5. type >"pip install -e ."

Getting the Video and Ground Truth Data
---------------------------------------
Follow the link in the paper to download the data.

To download Narrabeen ground truth data:

Bathymetric validation data for this drone flight is kindly provided by the NSW Department of Planning, Industry and Environment (NSW DPIE, formerly NSW OEH). This data is available on the Australian Ocean Data Network (AODN) Data Portal. To access this data, do the following:
1. Go to https://catalogue-imos.aodn.org.au/geonetwork/srv/eng/catalog.search#/metadata/8b2ddb75-2f29-4552-af6c-eac9b02156a6 
2. Click on “View and download data through the AODN portal”
3. To navigate to Narrabeen Beach, select the bounding box 
	N: --33.70
	S: -33.74
	E: 151.33
	W: 151.29 
4. At the bottom of the page, click “Next”
5. Click on the download link to download the dataset as a zip file. The relevant data file is
“NSWOEH_20170529_NarrabeenNorthenBeaches_STAX_2017_0529_Narrabeen_Post_ECL_No4_Hydro_Depths.xyz”

copy all data in the 'data' folder in 'COCOS_main'

If you want to try your own data:
- go to Data.py
- add your video in the same format as the provided field site videos (under the method "get_Video")and give it a label (e.g., label == 'monterey'). By setting fieldsite = 'monterey' in Main.py, this video will be analysed
- if you have them, you can add ground truth bathy data under the method "get_GroundTruth". Just stick to the example formats again and it should work

Use COCOS
---------
The COCOS code is currently set up in a MATLAB kind of style. Therefore it is advised to use spyder.

1. open a terminal in the COCOS envirnoment (if it isn't still open from installation)
2. type >"spyder"
3. Currently COCOS uses a basic plotter (which will definitely have to be improved in future). To be sure that plotting works, the plotting windows should open outside the spyder console. To check this go to >Tools>Preferences>IPython console>graphics> backend: Automatic
4. open the Main.py file within the cocos_map folder in COCOS_main and just run it.

By default, fieldsite = 'duck'. You may change this variable to another field site. The options currently are: 'duck', 'porthtowan', 'scheveningen', 'narrabeen' 

COCOS options/settings
----------------------

The Main.py file reads the options from Options.py. Therein you find a list of the different options and settings.

If you want to change an option, do this by changing the value of the specific keyword in the Main.py file after "opts = Options(Video, ...)"
(Do not change any values in Options.py, since those are the default values.)

To start with, option keywords you can try out are (default values are listed in Options.py): 
- CPU_speed (e.g., try 'normal', 'fast' or 'accurate') 
- gc_kernel_sampnum (e.g., set to 13)
- standing_wave_flag (try False)
- frame_int (try 8 or 'OnTheFly')
- stationary_time (try 30)

If your video is measured at 2 Hz the default settings should be fine. If your video has different framerates, the optimal settings may change. A detailed guide will soon be provided.

have fun!
---------
