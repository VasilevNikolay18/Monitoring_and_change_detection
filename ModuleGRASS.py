#!/usr/bin/env python

import ImageProcessing
import grass.script as gscript
from grass.pygrass.modules.shortcuts import general as g

folder = 'C:\\Documents\\My_works\\Forest_and_rubber_plantation_monitoring\\Cloud shadow snow detection\\Cutting_qqq'
channels = ['B3', 'B5', 'B6', 'BQA']

rasters = gscript.parse_command('g.list' ,type='raster').keys()
g.remove(type='raster', name=rasters, flags='fb')

rasters = ImageProcessing.loadCollection(folder=folder, channels=channels)

ImageProcessing.TOAR(rasters, ['B3','B5','B6'])

T = ImageProcessing.TMaskAlgorithm(images=rasters)

print(T)
print('OK')

