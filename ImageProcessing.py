# import some useful libraries:
import logging,collections,sys,os,re,time,copy,datetime,osr
import numpy as np
from osgeo import gdal
import grass.script as gscript
from grass.script import array as garray

# ///////////////////////////////////////////////////////
"""
interface
"""
# ///////////////////////////////////////////////////////

# get current environment:
environ = gscript.parse_command('g.gisenv')
GISDBASE = environ['GISDBASE'][1:-2]
MAPSET = environ['MAPSET'][1:-2]
LOCATION_NAME = environ['LOCATION_NAME'][1:-2]
PROJECTPATH = os.path.join(GISDBASE,LOCATION_NAME,MAPSET)
TEMPPATH = os.path.join(PROJECTPATH,'TEMP')

if not(os.path.isdir(TEMPPATH)): os.mkdir(TEMPPATH)

# set log file:
logging.basicConfig(level=logging.INFO, format='%(asctime)s \'ImageProcessing\': %(lineno)-8s %(levelname)-8s %(message)s')

# class Channel:
class Channel:


    def __init__(self,array,res=None,transform=None,crs=None,bounds=None,shape=None):
        """
        Constructor:
        :param: array:     [array]                       - array which will be used as raster {required!!!}
        :param: res:       [list[EWres,NSres]]           - resolution of the raster           {default=None}
        :param: transform: [list[parameters]]            - geotransformation of the raster    {default=None}
                transform[0] - top geocoordinate of the raster;
                transform[1] - east-west spatial resolution of the raster;
                transform[2] - the first rotate angle of the raster;
                transform[3] - left geocoordinate of the raster;
                transform[4] - the second rotate angle of the raster;
                transform[5] - north-south spatial resolution of the raster;
        :param: crs:       [PROJ4 format]                - projection of the raster           {default=None}
        :param: bounds:    [dictionary{'N','E','S','W'}] - coordinate of raster's boundary    {default=None}
        :param: shape:     [list[Xsize,Ysize]]           - shape of the raster                {default=None}
        """
        self._array = array
        self._transform = transform
        self._crs = crs
        self._res = res
        self._bounds = bounds
        self._shape = shape


    def projection(self):
        """
        Getting projection of the channel:
        :return: [PROJ4 format]
        """
        return self.__helpFunc('projection')


    def transform(self):
        """
        Getting geotransformation of the channel:
        :return: [list[parameters]]
        """
        return self.__helpFunc('transform')


    def res(self):
        """
        Getting resolution of the channel:
        :return: [list[EWres,NSres]]
        """
        return self.__helpFunc('res')


    def shape(self):
        """
        Getting shape of the channel:
        :return: [list[Xsize,Ysize]]
        """
        return self.__helpFunc('shape')


    def bounds(self):
        """
        Getting bounds of the channel:
        :return: [dictionary{'N','E','S','W'}]
        """
        return self.__helpFunc('bounds')


    def __helpFunc(self,type):
        """
        Just assist function for getting raster parameters:
        :param type: [str] - name of the raster parameter {required!!!}
        :return: raster parameter (projection,bounds,resolution,shape...)
        """
        if type == 'res': result = self._res
        elif type == 'projection': result = self._crs
        elif type == 'transform': result = self._transform
        elif type == 'shape': result = self._shape
        elif type == 'bounds': result = self._bounds
        return result


    def getPixel(self,pos=[1,1]):
        """
        Getting pixel value:
        :param pos: [list[X,Y]] - pixel's coo
        :return:
        """
        return self._array[pos[0] - 1, pos[1] - 1]


    def __str__(self):
        return ('%(array)s\n'
                '\'res\': %(res)s\n'
                '\'crs\': %(crs)s\n'
                '\'transform\':\n%(transform)s\n'
                '\'shape\': %(shape)s\n'
                '\'bounds\': %(bounds)s\n'
                % {'array':self._array,'res':self._res,'crs':self._crs,'transform':self._transform,
                   'shape':self._shape,'bounds':self._bounds})


# class Image:
class Image:

    def __init__(self,folder=None,channelDict=None,channels=None,metadata=None,maps=None,names=None,foldname=None):
        if folder != None: self.__load(folder=folder,channels=channels)
        elif channelDict != None: self.__construct(channelDict=channelDict,metadata=metadata,foldname=foldname)
        else: self.__fromProj(maps=maps,names=names)


    def __fromProj(self,maps,names):
        if type(maps) == str: maps = [maps]
        if type(names) == str: names = [names]
        chanDict = collections.OrderedDict()
        for i in range(len(maps)):
            array = garray.array(maps[i])
            parsing = gscript.parse_command('r.info',map=maps[i],flags='ge')
            res = [int(parsing['ewres']),int(parsing['nsres'])]
            shape = [int(parsing['cols']),int(parsing['rows'])]
            bounds = {'N':int(parsing['north']),'S':int(parsing['south']),
                      'W':int(parsing['west']),'E':int(parsing['east'])}
            crs = gscript.read_command('g.proj',flags='jf')
            transform = self.__getTransformParameters(belowright=[bounds['S'],bounds['E']],
                                                      topleft=[bounds['N'],bounds['W']],rastersize=shape,res=res)
            chanDict.update({names[i]:Channel(array,res=res,transform=transform,crs=crs,bounds=bounds,shape=shape)})
        self.__construct(channelDict=chanDict,metadata=None,foldname=None)


    def __construct(self,channelDict,metadata=None,foldname=None):
        self._bands = collections.OrderedDict(sorted(channelDict.items(), key=lambda x: x[0]))
        self._metadata = metadata
        self._folder = foldname


    def __load(self,folder,channels='ALL'):
        files = os.listdir(folder)
        self._folder = folder
        metadatafile = [name for name in files if re.findall(r'\w+MTL.txt', name) != []][0]
        files = [name for name in files if name.endswith(tuple([".TIF", 'TIFF', 'jpg']))]
        if files == []:
            logging.error('No one files was found!')
            sys.exit()
        with open(os.path.join(folder,metadatafile)) as f:
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
        metadata = collections.OrderedDict([('CLOUD_COVER',listrows[cloudcoverindex].split(' = ')[1][:-1]),
                                            ('DATE_ACQUIRED',listrows[dataacquiredindex].split(' = ')[1][:-1]),
                                            ('SUN_ELEVATION',listrows[sunelevationindex].split(' = ')[1][:-1]),
                                            ('SUN_AZIMUTH',listrows[sunazimuthindex].split(' = ')[1][:-1])])
        self._metadata = metadata
        if channels == 'ALL': channels = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','BQA']
        elif type(channels) == str: channels = [channels]
        for filename in files:
            START_TIME_FILE = time.time()
            if re.findall(r'B\w+', filename.split('.')[0])[0] in channels:
                gdaldata = gdal.Open(os.path.join(folder,filename))
                array = gdaldata.ReadAsArray()
                transform = gdaldata.GetGeoTransform()
                res = [abs(transform[1]),abs(transform[5])]
                crs = gdaldata.GetProjection()
                inSRS_converter = osr.SpatialReference()
                inSRS_converter.ImportFromWkt(crs)
                crs = inSRS_converter.ExportToProj4()
                shape = [gdaldata.RasterXSize,gdaldata.RasterYSize]
                NW = self.__getGeoCoordinates(0,0,transform)
                SE = self.__getGeoCoordinates(shape[0], shape[1], transform)
                bounds = {'N':NW['X'],'S':SE['X'],'W':NW['Y'],'E':SE['Y']}
                newNote = Channel(array=array,res=res,transform=transform,crs=crs,bounds=bounds,shape=shape)
                newChannel = {'%s' % (re.findall(r'B\w+', filename.split('.')[0])[0]): newNote}
                ChannelDict.update(newChannel)
                logging.info('file %(filename)s was succesfuly downloaded\nsession time: --- %(time)s seconds ---\n'
                             % {'filename': filename, 'time': (time.time() - START_TIME_FILE)})
                gdaldata = None
        self._bands = ChannelDict


    def __getGeoCoordinates(X,Y,transform):
        Matrix1 = np.array(transform).reshape(2,3)
        Matrix2 = np.array([1,X,Y]).reshape(3,1)
        Matrix = np.dot(Matrix1,Matrix2).reshape(2,1)
        return {'X': Matrix[0][0], 'Y': Matrix[1][0]}


    __getGeoCoordinates = staticmethod(__getGeoCoordinates)


    def __getTransformParameters(belowright,topleft,rastersize,res):
        rotation1 = (belowright[0]-topleft[0]-res[0]*rastersize[0])/belowright[1]
        rotation2 = (belowright[1]-topleft[1]+abs(res[1])*rastersize[1])/belowright[0]
        return [topleft[0],res[0],rotation1,topleft[1],rotation2,-abs(res[1])]


    __getTransformParameters = staticmethod(__getTransformParameters)


    def __str__(self):
        s = 'Image object:\n{start}\n'
        for key in self._bands:
            s += ('\'channel\': %(channel)s\n'
                  '%(channelblock)s\n'
                  % {'channel': key, 'channelblock': self._bands[key].__str__()})
        s += '\'metadata\': %(metadata)s\n' % {'metadata': self._metadata}
        s += '{end}\n\n'
        return s


    def bandNames(self):
        return list(self._bands.keys())


    def select(self, bands):
        copyImage = copy.deepcopy(self)
        if type(bands) == str: bands = [bands]
        selectedBands = [(key,copyImage._bands[key]) for key in copyImage._bands if key in bands]
        if selectedBands == []:
            logging.error('Chosen bands were not found!')
            sys.exit()
        OtherBands = [other for other in bands if not (other in list(copyImage._bands.keys()))]
        if OtherBands != []:
            logging.error('Bands %s were not found in Image.' % (OtherBands))
            sys.exit()
        return Image(channelDict=collections.OrderedDict(selectedBands),metadata=copyImage._metadata,foldname=copyImage._folder)


    def projection(self):
        return self.__helpFunc('projection')


    def transform(self):
        return self.__helpFunc('transform')


    def res(self):
        return self.__helpFunc('res')


    def shape(self):
        return self.__helpFunc('shape')


    def bounds(self):
        return self.__helpFunc('bounds')


    def getPixel(self,pos=[0,0]):
        if len(self.bandNames()) != 1:
            logging.error('Attempt to get pixel value for several bands! Only one band is required!')
            sys.exit()
        return self._bands[self.bandNames()[0]].getPixel(pos)


    def get(self,prop):
        if not (prop in ['CLOUD_COVER', 'DATE_ACQUIRED', 'SUN_ELEVATION', 'SUN_AZIMUTH']):
            logging.error('Metadata propertie \'%s\' does not exist! Available properties are: '
                          '\'CLOUD_COVER\', \'DATE_ACQUIRED\', \'SUN_ELEVATION\', \'SUN_AZIMUTH\'' % (prop))
            sys.exit()
        return self.__helpFunc('metadata', prop='%s' % (prop))


    def __helpFunc(self,arg, prop='CLOUD_COVER'):
        if (len(self.bandNames())!= 1) and (arg!= 'metadata'):
            logging.error('Operation can not be applied! Only one band is required!')
            sys.exit()
        channel = self._bands[self.bandNames()[0]]
        if arg == 'res': result = channel.res()
        elif arg == 'projection': result = channel.projection()
        elif arg == 'transform': result = channel.transform()
        elif arg == 'shape': result = channel.shape()
        elif arg == 'bounds': result = channel.bounds()
        elif arg == 'metadata': result = self._metadata[prop]
        return result


    def getDate(self):
        DateList = self.get('DATE_ACQUIRED').split('-')
        return datetime.date(int(DateList[0]), int(DateList[1]), int(DateList[2]))


    def first(self):
        return self.select(self.bandNames()[0])


    def addBands(self,bands):
        try:
            copyImage = copy.deepcopy(self)
            copyDict = copyImage._bands
            for key, band in bands._bands.items(): copyDict.update({key: band})
            return Image(channelDict=copyDict,metadata=copyImage._metadata,foldname=copyImage._folder)
        except AttributeError:
            logging.error('Attempt to add not \'Image\' object!')
            sys.exit()


    def rename(self,newName):
        try:
            if type(newName) == str: newName = newName.split(',')
            if len(newName) != len(self.bandNames()):
                logging.error(
                    'Count of bands does not match to count of new names! For correct renaming use as new names as bands you have selected! '
                    'Found %(selected)s bands and %(names)s names!' % {'selected': len(self.bandNames()),
                                                                       'names': len(newName)})
                sys.exit()
            copyImage = copy.deepcopy(self)
            copyDict = collections.OrderedDict(
                map(lambda name, key: (name,copyImage._bands[key]), newName, list(copyImage._bands.keys())))
            return Image(channelDict=copyDict,metadata=copyImage._metadata,foldname=copyImage._folder)
        except TypeError:
            logging.error('Not available type of band name! It must be \'str\'!')
            sys.exit()


    def Save(self,folder):
        first = self.first()
        driver = gdal.GetDriverByName('GTiff')
        cols = first.shape()[0]
        rows = first.shape()[1]
        bands = len(self.bandNames())
        proj = first.projection()
        inSRS_converter = osr.SpatialReference()
        inSRS_converter.ImportFromProj4(proj)
        proj = inSRS_converter.ExportToWkt()
        transform = first.transform()
        path = os.path.join(folder, '%s.tiff' % (name))
        dt = gdal.GDT_Float16
        outData = driver.Create(path, cols, rows, bands, dt)
        outData.SetProjection(proj)
        outData.SetGeoTransform(transform)
        for i in range(bands):
            outData.GetRasterBand(i + 1).WriteArray(self._bands[self.bandNames()[i]]._array)
        outData = None

    def toGRASS(self,i=1):

        folder = self._folder
        channels = self._bands.keys()
        files = os.listdir(folder)
        files = [name for name in files if name.endswith(tuple([".TIF", 'TIFF', 'jpg']))]
        files = sorted(files)
        rasters = []
        for filename in files:
            if re.findall(r'B\w+', filename.split('.')[0])[0] in channels:
                input = os.path.join(folder,filename)
                output = '%(main)s.%(i)s' %{'main':filename.split('.')[0].split('_B')[0],'i':i}
                gscript.run_command('r.in.gdal',input=input,output=output,overwrite=True,flags='k')
                i+=1
                rasters.append(output)
        self._rasters = rasters


    def TOAR(self):
        name = os.path.basename(self._folder)
        input = name + '.'
        output = name + '_toar.'
        metfile = os.path.join(self._folder,'%s_MTL.txt' %(name))
        try:
            gscript.run_command('i.landsat.toar',input=input,output=output,metfile=metfile,overwrite=True)
        except:
            pass
        copyImage = copy.deepcopy(self)
        i=0
        rasters = []
        for key in copyImage._bands.keys():
            i+=1
            copyImage._bands[key]._array = garray.array('%s%s' %(output,i))
            rasters.append(output)
        copyImage._rasters = rasters
        return copyImage


    def remove(self):
        for name in self._rasters:
            try:
                gscript.run_command('g.remove',type='raster',name=name,flags='fb')
            except:
                pass


    def FMask(self,name,radius=3):
        name_new = name + '_BQA_int'
        gscript.run_command('r.mapcalc', expression='%(BQA_int)s=int(%(BQA)s)' %{'BQA_int':name_new,'BQA':name}, overwrite=True)
        CloudMask_expression = 'CloudMask=((%(BQA)s & 32)!=0)&((%(BQA)s & 64)!=0)' %{'BQA':name_new}
        CloudShadowMask_expression = 'CloudShadowMask=((%(BQA)s & 128) !=0)&((%(BQA)s & 256)!=0)' %{'BQA':name_new}
        SnowMask_expression = 'SnowMask=((%(BQA)s & 512)!=0)&((%(BQA)s & 1024)!=0)' %{'BQA':name_new}
        CirrusMask_expression = 'CirrusMask=((%(BQA)s & 2048)!=0)&((%(BQA)s & 4096)!=0)' %{'BQA':name_new}
        ImageMask_expression = 'ImageMask=CloudMask | CloudShadowMask | SnowMask | CirrusMask'
        gscript.run_command('r.mapcalc',expression=CloudMask_expression,overwrite=True)
        gscript.run_command('r.mapcalc',expression=CloudShadowMask_expression,overwrite=True)
        gscript.run_command('r.mapcalc',expression=SnowMask_expression,overwrite=True)
        gscript.run_command('r.mapcalc',expression=CirrusMask_expression,overwrite=True)
        gscript.run_command('r.mapcalc',expression=ImageMask_expression,overwrite=True)
        output = name + '_FMask'
        gscript.run_command('r.grow',input='ImageMask',output=output,radius=radius,overwrite=True)
        copyImage = copy.deepcopy(self)
        copyImage._bands['FMask'] = copyImage._bands.pop('BQA')
        copyImage._bands[copyImage.bandNames()[0]]._array = garray.array(output)
        gscript.run_command('g.remove', type='raster', name=['CloudMask','CloudShadowMask','SnowMask','CirrusMask','ImageMask',name,name_new], flags='fb')
        copyImage._rasters = [output]
        return copyImage


# class ImageCollection:
class ImageCollection:

    def __init__(self,folder=None,channels=None,ImageDict=None):
        if folder != None: self.__load(folder=folder,channels=channels)
        else: self.__construct(ImageDict=ImageDict)


    def __construct(self,ImageDict):
        try:
            self._images = collections.OrderedDict(sorted(ImageDict.items(), key=lambda x: x[0]))
        except AttributeError:
            logging.error('Failed to create object! Type of argument must be \'OrderedDict\', but %s was found!' %(ImageDict.__class__))
            sys.exit()


    def __load(self,folder,channels='ALL'):
        try:
            files = os.listdir(folder)
        except FileNotFoundError:
            logging.error('Folder %(foldername)s was not found!' % {'foldername': folder})
            sys.exit()
        self._images = collections.OrderedDict()
        dirlist = [item for item in os.listdir(folder) if os.path.isdir(os.path.join(folder,item))]
        START_TIME_GENERAL = time.time()
        i=1
        for foldname in dirlist:
            logging.info('%(num)s) Reading folder %(folder)s\n' % {'num': i, 'folder': foldname})
            im = Image(folder=os.path.join(folder,foldname),channels=channels)
            self._images.update({'%s' % (foldname): im})
            logging.info('general time: --- %s seconds ---\n' % (time.time() - START_TIME_GENERAL))
            i+=1
        logging.info('Downloading was successefuly completed!\nTotal number of rasters: %(num)s.\n' % {'num': i - 1})


    def __str__(self):
        i = 1
        s = 'ImageCollection object:\n{start}\n'
        for key in self._images:
            s += '%(num)s) %(name)s  %(bands)s\n' % {'num': i, 'name': key, 'bands': self._images[key].bandNames()}
            i += 1
        s += '{end}\n\n'
        return s


    def size(self):
        return len(self._images.keys())


    def first(self):
        copyImage = copy.deepcopy(self._images[list(self._images.keys())[0]])
        return copyImage


    def get(self,num):
        if num > 0: num0 = num - 1
        elif num < 0: num0 = num
        else:
            logging.error('Image with number \'%(num)s\' does not exist! Use indexes from 1 to %(size)s or from -1 to -%(size)s for starting chose '
                  'image at the end of image collection.' %{'num': num, 'size': self.size()})
            sys.exit()
        try:
            copyImage = copy.deepcopy(self._images[list(self._images.keys())[num0]])
            return copyImage
        except IndexError:
            logging.error('Image with number \'%(num)s\' does not exist! Use indexes from 1 to %(size)s or from -1 to -%(size)s for starting chose '
                  'image at the end of image collection.' %{'num': num, 'size': self.size()})
            sys.exit()


    def Map(self,func):
        copyCollection = copy.deepcopy(self._images)
        newDict = collections.OrderedDict(map(lambda key,im: (key,func(im)), list(copyCollection.keys()),list(copyCollection.values())))
        return ImageCollection(ImageDict = newDict)


    def select(self,bands):
        def _f(im):
            return im.select(bands)
        return self.Map(_f)


    def filterDate(self,dateStart,dateEnd):
        dateStartList = dateStart.split('-')
        dateEndList = dateEnd.split('-')
        dateStartDateform = datetime.date(int(dateStartList[0]), int(dateStartList[1]), int(dateStartList[2]))
        dateEndDateform = datetime.date(int(dateEndList[0]), int(dateEndList[1]), int(dateEndList[2]))
        copyCollection = copy.deepcopy(self._images)
        newDict = collections.OrderedDict([(key, im) for key, im in copyCollection.items()
                                           if ((im.getDate() >= dateStartDateform) and (im.getDate() <= dateEndDateform))])
        return ImageCollection(ImageDict=newDict)


    def toGRASS(self,i=1):
        for image in self._images.values():
            image.toGRASS(i=i)


    def TOAR(self):
        def _f(image):
            return image.TOAR()
        return self.Map(_f)


    def remove(self):
        for image in self._images.values():
            image.remove()


    def FMask(self,radius=3):
        def _f(image):
            name = image._rasters[0]
            return image.FMask(name=name,radius=radius)
        return self.Map(_f)


    def TMaskAlgorithm(self,BACKUP_ALG_THRESHOLD=15,
                            RADIUS_BUFF=3,
                            T_MEDIAN_THRESHOLD=0.04,
                            GREEN_CHANNEL_PURE_SNOW_THRESHOLD=0.4,
                            NIR_CHANNEL_PURE_SNOW_THRESHOLD=0.12,
                            BLUE_CHANNEL_THRESHOLD=0.04,
                            NIR_CHANNEL_CLOUD_SNOW_THRESHOLD=0.04,
                            NIR_CHANNEL_SHADOW_CLEAR_THRESHOLD=-0.04,
                            SWIR1_CHANNEL_SHADOW_CLEAR_THRESHOLD=-0.04):

        # download 'B3' (blue channel), 'B5' (Near Infrared (NIR)) and 'B6' (Shortwave Infrared (SWIR) 1) channels:
        logging.info('downloading blue channel, Near Infrared (NIR) and Shortwave Infrared (SWIR) 1 channels to GRASS:')
        collection = self.select(['B3','B5','B6'])
        collection.toGRASS(i=1)

        # apply TOAR convertion:
        logging.info('Applying TOAR convertion:')
        collection_toar = collection.TOAR()

        # delete source images from projects:
        collection.remove()

        # the size of the collection:
        ImageCounts = collection_toar.size()
        text1 = 'Total number of images: %s' %(ImageCounts)
        text2 = 'Warning: You have less than %s images!' %(BACKUP_ALG_THRESHOLD)
        if ImageCounts>=BACKUP_ALG_THRESHOLD: print(text1)
        else: print(text2)

        # download 'BQA' channel:
        logging.info('downloading \'BQA\' channel to GRASS:')
        collection_BQA = self.select('BQA')
        collection_BQA.toGRASS(i=4)

        # compute FMask algorithm:
        logging.info('Computing FMask algorithm:')
        collection_FMask = collection_BQA.FMask(radius=RADIUS_BUFF)

        return collection_FMask









"""


            


            def reduceCount(self):
                gscript.run_command('r.mapcalc',expression='ConditionMap=0',overwrite=True)
                for key in self._images.keys():
                    expression = 'ConditionMap=ConditionMap + %s_FMask' %(key)
                    gscript.run_command('r.mapcalc',expression=expression,overwrite=True)
                return Image(maps='ConditionMap',names='ConditionMap')
"""







        # calculate total number of non clear pixels for image collection:
        #logging.info('Calculate total number of non clear pixels for image collection:')
        #ConditionMap = collection_FMask.reduceCount()

