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

if __name__ == "__main__":

    import cddm_experiment
    from cddm.video import multiply
    from cddm.window import blackman
    from cddm_experiment.frame_grabber import frame_grabber, queued_multi_frame_grabber
    from cddm_experiment.trigger import run_simulation
    from cddm_experiment.config import load_config


    set_verbose(2)

    trigger_config, cam_config = load_config()
    w = ((window,window),)* trigger_config["count"]

    t1,t2=run_simulation(trigger_config)

    PERIOD=trigger_config['n']*2

    dual_video = queued_multi_frame_grabber(frame_grabber, (trigger_config,cam_config))
    #dual_video = frame_grabber(trigger_config,cam_config)
    dual_video = apply_window(dual_video, (w1,w2))
    dual_video = multiply(dual_video, w)

    fdual_video = rfft2(dual_video, kimax = 96, kjmax = 96)


    data, bg = iccorr_multi(fdual_video, t1, t2, period = PERIOD, level = 5,
                              chunk_size = 128, show = True, auto_background = True, binning =  True, return_background = True)

    x, logdata = log_merge(cfast,cslow)

    np.save('x_live.npy',x)
    np.save('logdata_live.npy',logdata)

    plt.show()
