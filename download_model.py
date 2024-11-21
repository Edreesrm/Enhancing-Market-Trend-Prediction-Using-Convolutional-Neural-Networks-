# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 08:23:58 2024

@author: PIXEL
"""

import gdown

# Direct download link
url = 'https://drive.google.com/uc?export=download&id=17ysi_Fld4-VKi67BznHhc0dJ-9CpEW6J'
output = 'model.h5'

gdown.download(url, output, quiet=False)
