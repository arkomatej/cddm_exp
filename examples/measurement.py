'''
===================================================================================================
    Cross DDM live measurement python example.
    Copyright (C) 2019; Matej Arko, Andrej Petelin
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
===================================================================================================
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
import sys

START_TIMES = []

def normalize_fft(video):
    for frames in video:
        f1,f2 = frames
        ratio = f1[0,0]/f2[0,0]
        f2 = np.multiply(f2,ratio,out = f2)
        yield f1,f2

def save_first_frame(video, fname, id = 0):
    first = True
    for frames in video:
        if first:
            plt.imsave(fname, frames[id],cmap = "gray", vmin = 0, vmax = 2**16-1)
            first = False
        yield frames
        
def save_last_frame(video, fname, count, id = 0):
    i=0
    for frames in video:
        if i == (count-1):
            plt.imsave(fname, frames[id],cmap = "gray", vmin = 0, vmax = 2**16-1)
        i+=1
        yield frames
        
def read_start_time(video):
    first = True
    for frames in video:
        if first:
            START_TIMES.append(time.time())
            first = False
        yield frames
        
if __name__ == "__main__":
    
    from cddm.video import multiply
    from cddm.window import blackman
    from cddm.multitau import iccorr_multi, normalize_multi, log_merge
    from cddm.conf import set_verbose, set_rfft2lib
    from cddm.fft import rfft2
    from cddm_experiment.frame_grabber import frame_grabber, queued_multi_frame_grabber
    from cddm_experiment.trigger import run_simulation
    from cddm_experiment.config import load_config, s
    from cddm.viewer import MultitauViewer
    #from cddm.fft import normalize_fft
    
    #cddm configuration changes
    set_verbose(2)
    set_rfft2lib("pyfftw")
    
    #SETUP
    trigger_config, cam_config, analysis_config = load_config()
    
    PERIOD=trigger_config['n']*2
    count=trigger_config['count']
    cpath=trigger_config['cpath']
    number=int(analysis_config["number"])
    interval=int(analysis_config["interval"])
    kimax=int(analysis_config["kimax"])
    kjmax=int(analysis_config["kjmax"])
    show_viewer=bool(analysis_config["viewer"])
    normalize=bool(analysis_config["normalization"])
    directory=str(analysis_config["output"])
    overwrite=bool(analysis_config["overwrite"])
    m_times=[]
    
    if os.path.exists('./'+directory) and overwrite == False:
        
        print("WARNING! Directory already exists. Change output directory name with -o command or turn on overwrite by setting --ow to 1.")
        print("Exiting...")
        sys.exit()
        
    else:
        try:
            os.chdir('./'+directory)
        except FileNotFoundError:
            os.makedirs('./'+directory)
            os.chdir('./'+directory)
            
        print("Directory changed.")
        
        c=trigger_config
        c.update(cam_config)
        c.update(analysis_config)
        
        now = datetime.datetime.now()
        dtfile=now.strftime("_%d.%m.%Y_%H-%M-%S")
        dtstr="# Date and time : "+now.strftime("%Y-%m-%d %H:%M:%S")+"\n"
        
        with open(cpath+'.ini', 'w') as configfile:
            configfile.write(dtstr+s.format(**c))
            print("Configuration file saved/updated in output folder.")
          
    if show_viewer:
        viewer = MultitauViewer(scale = True, shape = (512,512))
        #initial mask parameters
        viewer.k = 15
        viewer.sector = 10
    else:
        viewer=None
        
    t1,t2=run_simulation(trigger_config)
    
    window = blackman((512,512))
    w = ((window,window),)* trigger_config["count"]
          
    #MAIN LOOP   
    
    for i in range(number):
        
        i+=1
        
        try:
            START_TIMES.append(time.time())
        
            dual_video = queued_multi_frame_grabber(frame_grabber, (trigger_config,cam_config))
            
            #dual_video = read_start_time(dual_video)
            
            dual_video = save_first_frame(dual_video, "camera1_first_{}.jpg".format(i), id = 0)
            dual_video = save_first_frame(dual_video, "camera2_first_{}.jpg".format(i), id = 1)
            
            dual_video = save_last_frame(dual_video, "camera1_last_{}.jpg".format(i), count=count, id = 0)
            dual_video = save_last_frame(dual_video, "camera2_last_{}.jpg".format(i), count=count, id = 1)
            
            dual_video = multiply(dual_video, w)
        
            fdual_video = rfft2(dual_video, kimax = kimax, kjmax = kjmax)
            
            if normalize:
                fdual_video = normalize_fft(fdual_video)
        
            data, bg, var = iccorr_multi(fdual_video, t1, t2, period = PERIOD,
                                      viewer  = viewer,  auto_background = True, binning =  True)
        
            cfast, cslow = normalize_multi(data, background = bg, variance = var, scale=False)
        
            x, logdata = log_merge(cfast,cslow)
        
            np.save(directory+'_time_'+str(i)+'.npy', x)
            np.save(directory+'_data_'+str(i)+'.npy', logdata)
            Tstr=datetime.datetime.fromtimestamp(START_TIMES[-1]).strftime("%H:%M:%S")
            m_times.append(str(i)+'\t'+Tstr)
            np.savetxt(directory+'_measurement_times.txt', m_times, delimiter="\t", fmt="%s")
            print("Measurement saved.")
        
        except:
            print(sys.exc_info())
            print("An error occured. Continuing execution...")
            
        finally:
            
            if i == number:
                print("Measurement finished.")
                break
            else:           
                T=time.time()
                sleep = interval-T+START_TIMES[-1]
                if sleep > 0:
                    print('Waiting until the next measurement...')
                    time.sleep(sleep)
        
        