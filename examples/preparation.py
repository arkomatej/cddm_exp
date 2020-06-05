''' 
===================================================================================================
    Cross DDM measurement preparation python example. Used for camera alignment and focus setup.
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

from cddm_experiment.config import load_config
from cddm.video import show_video, play, show_fft, show_diff, normalize_video
from cddm_experiment.frame_grabber import frame_grabber, queued_multi_frame_grabber
from cddm.conf import set_showlib
import numpy as np

set_showlib("cv2")

def norm_func(x): 
    return x/(2**8-1)

def subtract_last(video):
    im1,im2 = 0., 0.
    for frames in video:
        f1,f2 = frames
        yield f1 - im1, f2 - im2
        im1, im2 = frames
        
def normalize_diff(video):
    for frames in video:
        f1,f2 = frames
        ratio = np.mean(f1)/np.mean(f2)
        f2=ratio*f2
        yield f1,f2        

trigger_config, cam_config = load_config()

signal_ratio=1

clip=cam_config["imgheight"]*cam_config["imgwidth"]*signal_ratio
   
video = frame_grabber(trigger_config,cam_config)
#video = queued_multi_frame_grabber(frame_grabber, (trigger_config,cam_config))

video = show_video(video, id=0, norm_func=norm_func)
#video = normalize_video(video)
video = normalize_diff(video)
video = show_diff(video)
video = subtract_last(video)
video = show_fft(video, clip = clip)   
video = play(video, fps=25)

for frames in video:
   pass