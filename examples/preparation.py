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
from cddm.video import show_video, play, show_fft, show_diff
from cddm_experiment.frame_grabber import frame_grabber, queued_multi_frame_grabber

trigger_config, cam_config = load_config()
   
video = frame_grabber(trigger_config,cam_config)
#video = queued_multi_frame_grabber(frame_grabber, (trigger_config,cam_config))

video=show_video(video, id=0)
video=show_diff(video)
video=show_fft(video, clip = 5000)   
video=play(video, fps=30)

for frames in video:
   pass