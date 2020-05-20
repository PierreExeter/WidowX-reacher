#!/bin/bash

ffmpeg -t 30 -i widowx_plot3d.mp4 -vf "fps=10,scale=1280:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 widowx_plot3d.gif
