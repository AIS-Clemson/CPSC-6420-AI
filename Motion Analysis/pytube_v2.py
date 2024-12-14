#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:25:17 2024

@author: maxgray
"""

from pytubefix import YouTube
from pytubefix.cli import on_progress
 
# url = 'http://youtube.com/watch?v=2lAe1cqCOXo'
# url = 'https://www.youtube.com/watch?v=Cdlm-hgPcEY'
url = 'https://www.youtube.com/watch?v=EjY-tILXJCU'
 
yt = YouTube(url, on_progress_callback = on_progress)
print(yt.title)

print(yt.streams)
# yt.streams.filter(res="720p")


# ys = yt.streams.get_highest_resolution()
ys = yt.streams.get_by_itag(302)

ys.download()


