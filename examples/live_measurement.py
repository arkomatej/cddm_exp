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

def normalize_fft(video):
    for frames in video:
        f1,f2 = frames
        ratio = f1[0,0]/f2[0,0]
        f2 = np.multiply(f2,ratio,out = f2)
        yield f1,f2

def multi_frame_grabber(trigger_config,cam_config):
    dtype = np.dtype("uint16")
    shape = (512,512)
    n = shape[0] * shape[1]
    frame1 = np.ones(shape,dtype)
    time.sleep(1.)
    for i in range(1000):
        time.sleep(0.001)
        print ("frame {} captured".format(i))
        yield frame1.copy(), frame1.copy()#np.random.randn(*shape), np.random.randn(*shape)

if __name__ == "__main__":
    
    from cddm.video import multiply
    from cddm.window import blackman
    from cddm.multitau import iccorr_multi, normalize_multi, log_merge
    from cddm.conf import set_verbose, set_rfft2lib
    from cddm.fft import rfft2#, normalize_fft
    from cddm_experiment.frame_grabber import frame_grabber, queued_multi_frame_grabber, shared_multi_frame_grabber
    from cddm_experiment.trigger import run_simulation
    from cddm_experiment.config import load_config
    from cddm.viewer import MultitauViewer

    set_verbose(2)
    
    set_rfft2lib("pyfftw")

    trigger_config, cam_config = load_config()
    
    window = blackman((512,512))
    window = np.float32(window)
    w = ((window,window),)* trigger_config["count"]

    t1,t2=run_simulation(trigger_config)

    PERIOD=trigger_config['n']*2

    #dual_video = queued_multi_frame_grabber(frame_grabber, (trigger_config,cam_config))
    dual_video = queued_multi_frame_grabber(frame_grabber, (trigger_config,cam_config))
    #dual_video = frame_grabber(trigger_config,cam_config)
    dual_video = multiply(dual_video, w)

    fdual_video = rfft2(dual_video, kimax = 96, kjmax = 96)
    fdual_video = normalize_fft(fdual_video)
    
    #viewer = MultitauViewer(scale = False, shape = (512,512))
    #viewer.k = 15 #initial mask parameters,
    #viewer.sector = 10

    data, bg, var = iccorr_multi(fdual_video, t1, t2, period = PERIOD,
                              viewer  = None,  auto_background = True, binning =  True)#, thread_divisor = 6)

    cfast, cslow = normalize_multi(data, background = bg, variance = var, scale=False)

    x, logdata = log_merge(cfast,cslow)

    np.save('times.npy',x)
    np.save('data.npy',logdata) 
    
    print("Measurement saved.")

    plt.show()
