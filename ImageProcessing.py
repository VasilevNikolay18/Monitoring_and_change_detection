# import some useful libraries:
from __future__ import division
import sys
import time
import math
import re
import os
import numpy as np
import rasterio
import cv2
import collections
import copy

# ///////////////////////////////////////////////////////
"""
interface
"""
# ///////////////////////////////////////////////////////

# class Channel:
class Channel:

    def __init__(self, array, res=None, transform=None, crs=None, bounds=None, shape=None, driver=None):
        """
        constructor:
        :param array   : [np.ndarray]           - raster                                  (required!!!)
        :param res     : [list[1-axis, 2-axis]] - resolution of the raster                (default = None)
        :param crs     : [string]               - transformation parameters of the raster (default = None)
        :param bounds  : [list of corners]      - bounds of the raster                    (default = None)
        :param shape   : [list[width,height]]   - shape of the raster                     (default = None)
        :param driver  : [string]               - driver format of the raster             (default = None)
        """

        self._array = array
        self._transform = transform
        self._crs = crs
        self._res = res
        self._driver = driver
        self._bounds = bounds
        self._shape = shape

    def projection(self):

        """
        getting projection of the channel:
        :return: projection
        """

        return self.__helpFunc('projection')

    def transform(self):

        """
        getting transformation parameters of the raster:
        :return: transform
        """

        return self.__helpFunc('transform')

    def res(self):

        """
        getting resolution of the raster:
        :return: res (tuple)
        """

        return self.__helpFunc('res')

    def driver(self):

        """
        getting driver format of the raster:
        :return: driver (string)
        """

        return self.__helpFunc('driver')

    def shape(self):

        """
        getting shape of the raster:
        :return: shape (tuple)
        """

        return self.__helpFunc('shape')

    def bounds(self):

        """
        getting bounds of the raster:
        :return: bounds
        """

        return self.__helpFunc('bounds')

    def __helpFunc(self, arg):

        """
        just assist function for less writings -):
        :param arg : [string] - the specifical name for determine function in which it will be used (required!!!)
        :return object (res/projection/transform/driver/shape/bounds)
        """

        if arg == 'res': result = self._res
        elif arg == 'projection': result = self._crs
        elif arg == 'transform': result = self._transform
        elif arg == 'driver': result = self._driver
        elif arg == 'shape': result = self._shape
        elif arg == 'bounds': result = self._bounds
        return result

    def getPixel(self, pos=[0,0]):

        """
        getting value for giving pixel:
        :param pos: [list[row, column]] - position of the pixel
        :return value (number)
        """

        try:
            return self._array[pos[0]-1,pos[1]-1]
        except IndexError:
            print('Pixel with coordinates %(pos)s does not exist! Available values are: [1-%(maxrow)s ; 1-%(maxcol)s]]'
                  %{'pos': pos, 'maxrow': self.shape()[0], 'maxcol': self.shape()[1]})
            sys.exit()

    def __str__(self):

        """
        giving right representation for Channel object:
        :return description (string)
        """

        return ('%(array)s\n'
                '\'res\': %(res)s\n'
                '\'crs\': %(crs)s\n'
                '\'transform\':\n%(transform)s\n' 
                '\'shape\': %(shape)s\n'
                '\'bounds\': %(bounds)s\n'
                %{'array': self._array, 'res': self._res, 'crs': self._crs, 'transform': self._transform, 'shape': self._shape, 'bounds': self._bounds})


# class Image:
class Image:

    def __init__(self, folder=None, channelDict=None, channels=None, metadata=None):

        """
        constructor:
        :param folder     : [string]                       - name of the folder which will be used for reading files     (required!!!)
        :param channelDict: [OrderedDict{channel objects]} - dictionary of channels which should be merge in image       (required!!!)
                                                  {exclude if 'folder' is indicated}
        :param channels   : [list[name of channels]]       - list of channel names for choosing which files will be read (default = None)
        :param metadata   : [OrderedDict{properties}]      - metadata of the image                                       (default = None)
        """

        if folder != None: self.__load(folder, channels)
        else: self.__construct(channelDict, metadata)

    def __construct(self, channelDict, metadata=None):

        """
        constructor #1 (using folder for downloading channels):
        :param channelDict: [OrderedDict{channel objects]} - dictionary of channels which should be merge in image (required!!!)
        :param metadata:    [OrderedDict{properties}]      - metadata of the image                                 (default = None)
        """
        try:
            self._bands = collections.OrderedDict(sorted(channelDict.items(), key=lambda x: x[0]))
        except AttributeError:
            print('Failed to create object! Type of argument must be \'OrderedDict\', but %s was found!' %(channelDict.__class__))
            sys.exit()
        self._metadata = metadata

    def __load(self, folder, channels=['B2', 'B3', 'B4']):

        """
        constructor #2 (using folder for downloading channels):
        :param folder   : [string]                 - name of the folder which will be used for reading files     (required!!!)
        :param channels : [list[name of channels]] - list of channel names for choosing which files will be read (default = ['B2', 'B3', 'B4'])
        """

        try:
            files = os.listdir(folder)
        except FileNotFoundError:
            print('Folder \'%(foldername)s\' was not found!' % {'foldername': folder})
            sys.exit()
        try:
            metadatafile = [name for name in files if re.findall(r'\w+MTL.txt', name) != []][0]
        except IndexError:
            print('Metadata in the folder \'%s\' was not found!' %(folder))
            sys.exit()
        files = [name for name in files if name.endswith(tuple([".TIF", 'TIFF', 'jpg']))]
        if files == []:
            print()
            sys.exit()
        with open('%(foldername)s\\%(filename)s' % {'foldername': folder, 'filename': metadatafile}) as f:
            listrows = f.readlines()
            for row in listrows:
                if re.findall(r'CLOUD_COVER', row) != []:
                    cloudcoverindex = listrows.index(row)
                if re.findall(r'DATE_ACQUIRED', row) != []:
                    dataacquiredindex = listrows.index(row)
                if re.findall(r'SUN_ELEVATION', row) != []:
                    sunelevationindex = listrows.index(row)
                if re.findall(r'SUN_AZIMUTH', row) != []:
                    sunazimuthindex = listrows.index(row)
            f.close()
        ChannelDict = collections.OrderedDict()
        metadata = collections.OrderedDict([('CLOUD_COVER', listrows[cloudcoverindex].split(' = ')[1][:-1]),
                                            ('DATE_ACQUIRED', listrows[dataacquiredindex].split(' = ')[1][:-1]),
                                            ('SUN_ELEVATION', listrows[sunelevationindex].split(' = ')[1][:-1]),
                                            ('SUN_AZIMUTH', listrows[sunazimuthindex].split(' = ')[1][:-1])])
        self._metadata = metadata
        for filename in files:
            START_TIME_FILE = time.time()
            rasteriodata = rasterio.open('%(foldername)s\\%(filename)s' % {'foldername': folder, 'filename': filename})
            if re.findall(r'B\w+', filename.split('.')[0])[0] in channels:
                newNote = Channel(rasteriodata.read(1), rasteriodata.res, rasteriodata.transform, rasteriodata.crs,
                                  rasteriodata.bounds, rasteriodata.shape, rasteriodata.driver)
                newChannel = {'%s' % (re.findall(r'B\w+', filename.split('.')[0])[0]): newNote}
                ChannelDict.update(newChannel)
                print('file %s was succesfuly downloaded' % (filename))
                print('session time: --- %s seconds ---\n' % (time.time() - START_TIME_FILE))
            rasteriodata.close()
        self._bands = ChannelDict

    def __str__(self):

        """
        giving right representation for Image object:
        :return: description (string)
        """

        s = 'Image object:\n{start}\n\n'
        for key in self._bands:
            s += ('\'channel\': %(channel)s\n'
                  '%(channelblock)s\n'
                  %{'channel': key, 'channelblock': self._bands[key].__str__()})
        s += '\'metadata\': %(metadata)s\n' %{'metadata': self._metadata}
        s += '{end}\n\n'
        return s

    def Map(self, func):

        """
        mapping function func over image
        :param func: [function] - function which was mapping over image channels {requires!!!}
        :return: Image
        """

        copyImage = copy.deepcopy(self)
        channels = list(copyImage._bands.values())
        keys = list(copyImage._bands.keys())
        newDict = collections.OrderedDict(map(lambda key, ch: (key, func(ch._array)), keys, channels))
        return Image(channelDict=newDict, metadata=copyImage._metadata)

    def bandNames(self):

        """
        getting names of all bands:
        :return: band names (list)
        """

        return list(self._bands.keys())

    def select(self, bands):

        """
        selecting channels for getting new image which consist only selecting channels:
        :param bands: [list//string] - list of the band names or band name which should be chosen (required!!!)
        :return: Image
        """

        copyImage = copy.deepcopy(self)
        if type(bands) == str: bands = bands.split()
        selectedBands = [(key, copyImage._bands[key]) for key in copyImage._bands if key in bands]
        if selectedBands == []:
            print('Chosen bands were not found!')
            sys.exit()
        OtherBands = [other for other in bands if not(other in list(copyImage._bands.keys()))]
        if OtherBands != []:
            print('Bands %s were not found in Image.' %(OtherBands))
            sys.exit()
        return Image(channelDict = collections.OrderedDict(selectedBands), metadata = copyImage._metadata)

    def projection(self):

        """
        getting projection of the raster:
        :return: projection
        """

        return self.__helpFunc('projection')

    def transform(self):

        """
        getting transformation parameters of the raster:
        :return: transform
        """

        return self.__helpFunc('transform')

    def res(self):

        """
        getting resolution of the raster:
        :return: res (tuple)
        """

        return self.__helpFunc('res')

    def driver(self):

        """
        getting driver format of the raster:
        :return: driver (string)
        """

        return self.__helpFunc('driver')

    def shape(self):

        """
        getting shape of the raster:
        :return: shape (tuple)
        """

        return self.__helpFunc('shape')

    def bounds(self):

        """
        getting bounds of the raster:
        :return: bounds
        """

        return self.__helpFunc('bounds')

    def getPixel(self, pos=[0,0]):

        """
        getting value for giving pixel:
        :param pos: [list[row, column]] - position of the pixel
        :return: value (number)
        """
        if len(self.bandNames()) != 1:
            print('Attempt to get pixel value for several bands! Only one band is required!')
            sys.exit()
        return self._bands[self.bandNames()[0]].getPixel(pos)

    def get(self, prop):

        """
        getting metadata properties of the raster:
        :param prop: [string] - the name of metadata property (required!!!)
        Now following properties are available: 'CLOUD_COVER', 'DATE_ACQUIRED', 'SUN_ELEVATION', 'SUN_AZIMUTH'
        :return: metadata property (value)
        """
        if not(prop in ['CLOUD_COVER', 'DATE_ACQUIRED', 'SUN_ELEVATION', 'SUN_AZIMUTH']):
            print('Metadata propertie \'%s\' does not exist! Available properties are: \'CLOUD_COVER\', \'DATE_ACQUIRED\','
                  ' \'SUN_ELEVATION\', \'SUN_AZIMUTH\'' %(prop))
            sys.exit()
        return self.__helpFunc('metadata', prop='%s' % (prop))

    def __helpFunc(self, arg, prop='CLOUD_COVER'):

        """
        just assist function for less writings -):
        :param arg : [string] - the specifical name for determine function in which it will be used (required!!!)
        :param prop: [string] - the name of metadata property                                       (not required)
        :return: object (res/projection/transform/driver/shape/bounds/metadata)
        """

        if (len(self.bandNames()) != 1) and (arg != 'metadata'):
            print('Operation can not be applied! Only one band is required!')
            sys.exit()
        channel = self._bands[self.bandNames()[0]]
        if arg == 'res': result = channel.res()
        elif arg == 'projection': result = channel.projection()
        elif arg == 'transform': result = channel.transform()
        elif arg == 'driver': result = channel.driver()
        elif arg == 'shape': result = channel.shape()
        elif arg == 'bounds': result = channel.bounds()
        elif arg == 'metadata': result = self._metadata[prop]
        return result

    def getDate(self):

        """
        getting the date of the image:
        :return: Ordereddict (year, month, day)
        """

        return Date(self.get('DATE_ACQUIRED'))

    def first(self):

        """
        getting the first channel in image:
        :return: Image
        """

        return self.select(self.bandNames()[0])

    def addBands(self, bands):

        """
        adding band to given image:
        :param band: [Image] - band which will be added to image [required!!!]
        :return: Image
        """
        try:
            copyImage = copy.deepcopy(self)
            copyDict = copyImage._bands
            for key,band in bands._bands.items(): copyDict.update({key: band})
            return Image(channelDict = copyDict, metadata = copyImage._metadata)
        except AttributeError:
            print('Attempt to add not \'Image\' object!')
            sys.exit()

    def rename(self, newName):

        """
        rename band:
        :param newName: [string] - new name for band
        :return: Image
        """

        try:
            if type(newName) == str: newName = newName.split(',')
            if len(newName) != len(self.bandNames()):
                print('Count of bands does not match to count of new names! For correct renaming use as new names as bands you have selected! '
                      'Found %(selected)s bands and %(names)s names!' %{'selected': len(self.bandNames()), 'names': len(newName)})
                sys.exit()
            copyImage = copy.deepcopy(self)
            copyDict = collections.OrderedDict(map(lambda name,key: (name, copyImage._bands[key]), newName,list(copyImage._bands.keys())))
            return Image(channelDict = copyDict, metadata = copyImage._metadata)
        except TypeError:
            print('Not available type of band name! It must be \'str\'!')
            sys.exit()

    def __rewrite(self, array):

        """
        rewritting all rasters using new arrays but saving all raster properties:
        :param array: [n-dimension, shape] - 3-dimension array, in which the first axis is channel stack [required!!!]
        :return: nothing!
        """

        try:
            for i in range(len(self.bandNames())):
                self._bands[self.bandNames()[i]]._array = array[i:i + 1, :, :].reshape(array.shape[1], array.shape[2])
        except IndexError:
            print('Not available shape of \'array\' argument! Expected the same shape as Image object has!')
            sys.exit()

    def __add__(self, image2):

        """
        "+" rebooting:
        :param image2: [Image] - the second image which will be used for calculating result with the first image [required!!!]
        :return: Image
        """

        try:
            return self.__binaryOperations(image2, 'add')
        except TypeError:
            print('Attempt to add inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                  %{'first': self.__class__, 'second': image2.__class__})
        sys.exit()

    def __radd__(self, image1):

        """
        reverse "+" rebooting:
        :param image1: [Image] - the first image which will be used for calculating result with the second image [required!!!]
        :return: Image
        """

        try:
            return self.__binaryOperations(image1, 'add')
        except TypeError:
            print('Attempt to add inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                  %{'first': self.__class__, 'second': image1.__class__})
        sys.exit()

    def __sub__(self, image2):

        """
        "-" rebooting:
        :param image2: [Image] - the second image which will be used for calculating result with the first image [required!!!]
        :return: Image
        """

        try:
            return self.__binaryOperations(image2, 'sub')
        except TypeError:
            print('Attempt to subtract inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                  %{'first': self.__class__, 'second': image2.__class__})
        sys.exit()

    def __rsub__(self, image1):

        """
        reverse "-" rebooting:
        :param image1: [Image] - the first image which will be used for calculating result with the second image [required!!!]
        :return: Image
        """

        try:
            return self.__binaryOperations(image1, 'sub')
        except TypeError:
            print('Attempt to subtract inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                  %{'first': self.__class__, 'second': image1.__class__})
        sys.exit()

    def __mul__(self, image2):

        """
        "*" rebooting:
        :param image2: [Image] - the second image which will be used for calculating result with the first image [required!!!]
        :return: Image
        """

        try:
            return self.__binaryOperations(image2, 'mul')
        except TypeError:
            print('Attempt to multiply inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                  %{'first': self.__class__, 'second': image2.__class__})
        sys.exit()

    def __rmul__(self, image1):

        """
        reverse "*" rebooting:
        :param image1: [Image] - the first image which will be used for calculating result with the second image [required!!!]
        :return: Image
        """

        try:
            return self.__binaryOperations(image1, 'mul')
        except TypeError:
            print('Attempt to multiply inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                  %{'first': self.__class__, 'second': image1.__class__})
        sys.exit()

    def __truediv__(self, image2):

        """
        "/" rebooting:
        :param image2: [Image] - the second image which will be used for calculating result with the first image [required!!!]
        :return: Image
        """

        try:
            return self.__binaryOperations(image2, 'div')
        except TypeError:
            print('Attempt to divide inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                  %{'first': self.__class__, 'second': image2.__class__})
            sys.exit()

    def __rtruediv__(self, image1):

        """
        reverse "/" rebooting:
        :param image2: [Image] - the first image which will be used for calculating result with the second image [required!!!]
        :return: Image
        """

        try:
            return self.__binaryOperations(image1, 'div')
        except TypeError:
            print('Attempt to divide inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                  %{'first': self.__class__, 'second': image1.__class__})
            sys.exit()

    def __and__(self, image2):

        """
        "and" rebooting:
        :param image2: [Image] - the second image which will be used for calculating result with the first image [required!!!]
        :return: Image (boolean mask)
        """

        try:
            return self.__binaryOperations(image2, 'and')
        except TypeError:
            print('Attempt to use \'AND\' operation for inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                  %{'first': self.__class__, 'second': image2.__class__})
            sys.exit()

    def __rand__(self, image1):

        """
        reverse "and" rebooting:
        :param image1: [Image] - the first image which will be used for calculating result with the second image [required!!!]
        :return: Image (boolean mask)
        """

        try:
            return self.__binaryOperations(image1, 'and')
        except TypeError:
            print('Attempt to use \'AND\' operation for inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                  %{'first': self.__class__, 'second': image1.__class__})
            sys.exit()

    def __or__(self, image2):

        """
        "or" rebooting:
        :param image2: [Image] - the second image which will be used for calculating result with the first image [required!!!]
        :return: Image (boolean mask)
        """

        try:
            return self.__binaryOperations(image2, 'or')
        except TypeError:
            print(
                'Attempt to use \'OR\' operation for inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': image2.__class__})
            sys.exit()

    def __ror__(self, image1):

        """
        reverse "or" rebooting:
        :param image1: [Image] - the first image which will be used for calculating result with the second image [required!!!]
        :return: Image (boolean mask)
        """

        try:
            return self.__binaryOperations(image1, 'or')
        except TypeError:
            print(
                'Attempt to use \'OR\' operation for inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': image1.__class__})
            sys.exit()

    def __gt__(self, image2):

        """
        ">" rebooting:
        :param image2: [Image] - the second image which will be used for calculating result with the first image [required!!!]
        :return: Image (boolean mask)
        """

        try:
            return self.__binaryOperations(image2, 'gt')
        except TypeError:
            print(
                'Attempt to use \'>\' operation for inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': image2.__class__})
            sys.exit()

    def __ge__(self, image2):

        """
        ">=" rebooting:
        :param image2: [Image] - the second image which will be used for calculating result with the first image [required!!!]
        :return: Image (boolean mask)
        """

        try:
            return self.__binaryOperations(image2, 'ge')
        except TypeError:
            print(
                'Attempt to use \'>=\' operation for inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': image2.__class__})
            sys.exit()

    def __lt__(self, image2):

        """
        "<" rebooting:
        :param image2: [Image] - the second image which will be used for calculating result with the first image [required!!!]
        :return: Image (boolean mask)
        """

        try:
            return self.__binaryOperations(image2, 'lt')
        except TypeError:
            print(
                'Attempt to use \'<\' operation for inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': image2.__class__})
            sys.exit()

    def __le__(self, image2):

        """
        "<=" rebooting:
        :param image2: [Image] - the second image which will be used for calculating result with the first image [required!!!]
        :return: Image (boolean mask)
        """

        try:
            return self.__binaryOperations(image2, 'le')
        except TypeError:
            print(
                'Attempt to use \'<=\' operation for inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': image2.__class__})
            sys.exit()

    def __eq__(self, image2):

        """
        "==" rebooting:
        :param image2: [Image] - the second image which will be used for calculating result with the first image [required!!!]
        :return: Image (boolean mask)
        """

        try:
            return self.__binaryOperations(image2, 'eq')
        except TypeError:
            print(
                'Attempt to use \'==\' operation for inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': image2.__class__})
            sys.exit()

    def __ne__(self, image2):

        """
        "!=" rebooting:
        :param image2: [Image] - the second image which will be used for calculating result with the first image [required!!!]
        :return: Image (boolean mask)
        """

        try:
            return self.__binaryOperations(image2, 'ne')
        except TypeError:
            print(
                'Attempt to use \'!=\' operation for inconsistent objects! Available types are: \'Image\' and number! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': image2.__class__})
            sys.exit()

    def updateMask(self, mask):

        """
        masking:
        :param mask: [Image] - the mask using for masking image [required!!!]
        :return: Image (masked Image)
        """

        return self * mask

    def __binaryOperations(self, image2, ttype):

        """
        assist function for programming binary operations:
        :param image2: [Image]  - the second image which will be used for calculating result with the first image [required!!!]
        :param ttype : [string] - the name of rebooting operation                                                 [required!!!]
        :return: Image
        """

        if image2.__class__ == Image:
            im2shape = image2.first().shape()
            shape2 = [1, im2shape[0], im2shape[1]]
            stack2 = np.concatenate([image2._bands[key]._array.reshape(shape2) for key in image2._bands if True],
                                        axis=0)
        else:
            stack2 = image2
        im1shape = self.first().shape()
        shape1 = [1, im1shape[0], im1shape[1]]
        stack1 = np.concatenate([self._bands[key]._array.reshape(shape1) for key in self._bands if True], axis=0)
        if ttype == 'add': result = stack1 + stack2
        elif ttype == 'sub': result = stack1 - stack2
        elif (ttype == 'mul'): result = stack1 * stack2
        elif ttype == 'div': result = stack1 / stack2
        elif ttype == 'and': result = stack1 & stack2
        elif ttype == 'or': result = stack1 | stack2
        elif ttype == 'gt': result = stack1 > stack2
        elif ttype == 'ge': result = stack1 >= stack2
        elif ttype == 'lt': result = stack1 < stack2
        elif ttype == 'le': result = stack1 <= stack2
        elif ttype == 'eq': result = stack1 == stack2
        elif ttype == 'ne': result = stack1 != stack2
        copyImage = copy.deepcopy(self)
        copyImage.__rewrite(result)
        return copyImage

    def sin(self):

        """
        sin function:
        :return: Image
        """

        return self.__unaryOperations('sin')

    def cos(self):

        """
        cos function:
        :return: Image
        """

        return self.__unaryOperations('cos')

    def tan(self):

        """
        tan function:
        :return: Image
        """

        return self.__unaryOperations('tan')

    def cot(self):

        """
        cot function:
        :return: Image
        """

        return self.__unaryOperations('cot')

    def ln(self):

        """
        ln function:
        :return: Image
        """

        return self.__unaryOperations('ln')

    def invert(self):

        """
        inverse function:
        :return: Image
        """

        return self.__unaryOperations('invert')

    def __unaryOperations(self, ttype):

        """
        assist function for programming binary operations:
        :param image2: [Image]  - the second image which will be used for calculating result with the first image [required!!!]
        :param ttype : [string] - the name of rebooting operation                                                 [required!!!]
        :return: Image
        """

        im1shape = self.first().shape()
        shape1_before = [1, im1shape[0], im1shape[1]]
        stack1 = np.concatenate([self._bands[key]._array.reshape(shape1_before) for key in self._bands if True],
                                axis=0)
        if ttype == 'sin': result = np.sin(stack1)
        elif ttype == 'cos': result = np.cos(stack1)
        elif ttype == 'tan': result = np.tan(stack1)
        elif ttype == 'cot': result = 1 / np.tan(stack1)
        elif ttype == 'ln': result = np.log(stack1)
        elif ttype == 'invert': result = ~stack1
        copyImage = copy.deepcopy(self)
        copyImage.__rewrite(result)
        return copyImage


# class ImageCollection:
class ImageCollection:

    def __init__(self, folder=None, channels=None, ImageDict=None):

        """
        constructor:
        :param folder     : [string]                 - name of the root which will be used for finding folders        (required!!!)
        :param ImageList  : [dict{Image objects}]    - dictionary of Images which should be merge in image collection (required!!!)
                                                {exclude if 'folder' is indicated}
        :param channels   : [list[name of channels]] - list of channel names for choosing which files will be read    (default = None)
        """

        if folder != None: self.__load(folder, channels)
        else: self.__construct(ImageDict)

    def __construct(self, ImageDict):

        """
        constructor #1 (using image dictionary):
        :param ImageDict: [dictionary{Image objects}]  - dictionary of images which should be merge in image collection (required!!!)
        """

        try:
            self._images = collections.OrderedDict(sorted(ImageDict.items(), key=lambda x: x[0]))
        except AttributeError:
            print('Failed to create object! Type of argument must be \'OrderedDict\', but %s was found!' %(ImageDict.__class__))
            sys.exit()

    def __load(self, folder, channels=['B2', 'B3', 'B4']):

        """
        constructor #2 (using root folder for downloading channels):
        :param folder   : [string]                 - name of the root which will be used for finding folders     (required!!!)
        :param channels : [list[name of channels]] - list of channel names for choosing which files will be read (default = ['B2', 'B3', 'B4'])
        """

        try:
            files = os.listdir(folder)
        except FileNotFoundError:
            print('Folder %(foldername)s was not found!' % {'foldername': folder})
            sys.exit()
        self._images = collections.OrderedDict()
        dirlist = [item for item in os.listdir(folder) if os.path.isdir(os.path.join(folder, item))]
        START_TIME_GENERAL = time.time()
        i = 1
        for foldname in dirlist:
            print('%(num)s) Reading folder %(folder)s\n' % {'num': i, 'folder': foldname})
            im = Image(folder='%(root)s\\%(foldname)s' % {'root': folder, 'foldname': foldname}, channels=channels)
            self._images.update({'%s' % (foldname): im})
            print('general time: --- %s seconds ---\n' % (time.time() - START_TIME_GENERAL))
            i += 1
        print('Downloading was successefuly completed!\nTotal number of rasters: %(num)s.\n' % {'num': i - 1})

    def __str__(self):

        """
        giving right representation for ImageCollection object:
        :return: description (string)
        """

        i = 1
        s = 'ImageCollection object:\n{start}\n\n'
        for key in self._images:
            s += '%(num)s) %(name)s  %(bands)s\n' % {'num': i, 'name': key, 'bands': self._images[key].bandNames()}
            i += 1
        s += '{end}\n\n'
        return s

    def size(self):

        """
        getting size of the image collection:
        :return: value (int)
        """

        return len(self._images.keys())

    def first(self):

        """
        getting the first image in image collection:
        :return: Image
        """

        copyImage = copy.deepcopy(self._images[list(self._images.keys())[0]])
        return copyImage

    def get(self, num):

        """
        mapping function over image collection:
        :return: Image
        """
        if num > 0: num0 = num - 1
        elif num < 0: num0 = num
        else:
            print('Image with number \'%(num)s\' does not exist! Use indexes from 1 to %(size)s or from -1 to -%(size)s for starting chose '
                  'image at the end of image collection.' %{'num': num, 'size': self.size()})
            sys.exit()
        try:
            copyImage = copy.deepcopy(self._images[list(self._images.keys())[num0]])
            return copyImage
        except IndexError:
            print('Image with number \'%(num)s\' does not exist! Use indexes from 1 to %(size)s or from -1 to -%(size)s for starting chose '
                  'image at the end of image collection.' %{'num': num, 'size': self.size()})
            sys.exit()

    def Map(self, func):

        """
        mapping function over image collection:
        :return: ImageCollection
        """

        copyCollection = copy.deepcopy(self._images)
        newDict = collections.OrderedDict(map(lambda key,im: (key,im.Map(func)), list(copyCollection.keys()),list(copyCollection.values())))
        return ImageCollection(ImageDict = newDict)

    def select(self, bands):

        """
        selecting channels for getting new image collection which consist only selecting channels:
        :param bands: [list//string] - list of the band names or band name which should be chosen (required!!!)
        :return: ImageCollection
        """

        copyCollection = copy.deepcopy(self._images)
        newDict = collections.OrderedDict(map(lambda key, im: (key,im.select(bands)), list(copyCollection.keys()), list(copyCollection.values())))
        return ImageCollection(ImageDict=newDict)

    def filterDate(self, dateStart, dateEnd):

        """
        filtering image collection by date:
        :param dateStart: [string {'year'-'month'-'day'}] - start date {required}
        :param dateEnd:   [string {'year'-'month'-'day'}] - finish date {required}
        :return: ImageCollection
        """

        copyCollection = copy.deepcopy(self._images)
        newDict = collections.OrderedDict([(key, im) for key, im in copyCollection.items()
                                           if ((im.getDate() >= Date(dateStart)) and (im.getDate() <= Date(dateEnd)))])
        return ImageCollection(ImageDict=newDict)


# class Date:
class Date():

    def __init__(self, dateString):

        """
        constructor:
        :param dateString: [string {'year'-'month'-'day'}] - the string which will be converted in Date {requires!!!}
        """

        try:
            dateList = dateString.split('-')
        except:
            print('Unsupported format of date! It must be \'string\' such as: \'{year}-{month}-{day}\'')
            sys.exit()
        if ((int(dateList[1]) < 1) or (int(dateList[1]) > 12)):
            print('Unsupported format of month! Expected values from 1 to 12; got %s' %(dateList[1]))
            sys.exit()
        elif ((int(dateList[2]) < 1) or (int(dateList[2]) > 31)):
            print('Unsupported format of day! Expected values from 1 to 31; got %s' % (dateList[2]))
            sys.exit()
        self._year = int(dateList[0])
        self._month = int(dateList[1])
        self._day = int(dateList[2])


    def __str__(self):

        """
        'print' rebooting:
        :return: description (string)
        """

        return ('Date object:\n{start}\n\nyear: %(year)s\nmonth: %(month)s\nday: %(day)s\n{end}\n\n' %{'year': self._year, 'month': self._month, 'day': self._day})

    def toJulianDate(self):

        """
        getting Julian Date
        :return: value (number)
        """

        a = math.floor((14 - self._month) / 12)
        y = self._year + 4800 - a
        m = self._month + 12 * a - 3
        return (self._day + math.floor(((153 * m + 2) / 5)) + (365 * y) + math.floor(y / 4) - math.floor(y / 100) + math.floor(y / 400) - 32045)

    def __gt__(self, date2):

        """
        ">" rebooting:
        :param date2: [Date] - the second date which will be compared with the first date [required!!!]
        :return: boolean
        """

        try:
            return self.__dateComparison(date2, 'gt')
        except TypeError:
            print('Attempt to use \'>\' operation for inconsistent objects! Available type is: \'Date\'! Found: %(first)s and %(second)s'
                  %{'first': self.__class__, 'second': date2.__class__})
            sys.exit()

    def __ge__(self, date2):

        """
        ">=" rebooting:
        :param date2: [Date] - the second date which will be compared with the first date [required!!!]
        :return: boolean
        """

        try:
            return self.__dateComparison(date2, 'ge')
        except TypeError:
            print(
                'Attempt to use \'>=\' operation for inconsistent objects! Available type is: \'Date\'! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': date2.__class__})
            sys.exit()

    def __lt__(self, date2):

        """
        "<" rebooting:
        :param date2: [Date] - the second date which will be compared with the first date [required!!!]
        :return: boolean
        """

        try:
            return self.__dateComparison(date2, 'lt')
        except TypeError:
            print(
                'Attempt to use \'<\' operation for inconsistent objects! Available type is: \'Date\'! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': date2.__class__})
            sys.exit()

    def __le__(self, date2):

        """
        "<=" rebooting:
        :param date2: [Date] - the second date which will be compared with the first date [required!!!]
        :return: boolean
        """

        try:
            return self.__dateComparison(date2, 'le')
        except TypeError:
            print(
                'Attempt to use \'<=\' operation for inconsistent objects! Available type is: \'Date\'! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': date2.__class__})
            sys.exit()

    def __eq__(self, date2):

        """
        "==" rebooting:
        :param date2: [Date] - the second date which will be compared with the first date [required!!!]
        :return: boolean
        """

        try:
            return self.__dateComparison(date2, 'eq')
        except TypeError:
            print(
                'Attempt to use \'==\' operation for inconsistent objects! Available type is: \'Date\'! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': date2.__class__})
            sys.exit()

    def __ne__(self, date2):

        """
        "!=" rebooting:
        :param date2: [Date] - the second date which will be compared with the first date [required!!!]
        :return: boolean
        """

        try:
            return self.__dateComparison(date2, 'ne')
        except TypeError:
            print(
                'Attempt to use \'!=\' operation for inconsistent objects! Available type is: \'Date\'! Found: %(first)s and %(second)s'
                % {'first': self.__class__, 'second': date2.__class__})
            sys.exit()

    def __dateComparison(self, date2, ttype):

        """
        assist function for programming binary operations:
        :param date2: [Date] - the second date which will be compared with the first date [required!!!]
        :param ttype : [string] - the name of rebooting operation                                                 [required!!!]
        :return: boolean
        """

        JD1 = self.toJulianDate()
        JD2 = date2.toJulianDate()
        if ttype == 'gt': result = JD1 > JD2
        elif ttype == 'ge': result = JD1 >= JD2
        elif ttype == 'lt': result = JD1 < JD2
        elif ttype == 'le': result = JD1 <= JD2
        elif ttype == 'eq': result = JD1 == JD2
        elif ttype == 'ne': result = JD1 != JD2
        return result