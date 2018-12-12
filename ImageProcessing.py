# import some useful libraries:
import logging,collections,sys,os,re,time,copy,datetime,osr
from shutil import copyfile
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
METAPATH = os.path.join(PROJECTPATH,'META')

if not(os.path.isdir(TEMPPATH)): os.mkdir(TEMPPATH)
if not(os.path.isdir(METAPATH)): os.mkdir(METAPATH)

# set log file:
logging.basicConfig(level=logging.INFO, format='%(asctime)s \'ImageProcessing\': %(lineno)-8s %(levelname)-8s %(message)s')

# some useful funtions:
def readMeta(metafile,properties):
    """
    Getting metadata properties from metadata file:
    :param metafile:   [str]         - file name from which metadata will be read {required!!!}
    :param properties: [list[names]] - list of property names which will be read  {required!!!}
    :return: [OrderedDict]
    """
    if type(properties) == str: properties = [properties]
    try:
        iter(properties)
    except:
        logging.error('Type of \'properties\' does not iterable! Use \'list\', \'tupple\' or another types. '
                      'But found %s.' %(type(properties)))
        sys.exit()
    MetaDict = collections.OrderedDict()
    with open(metafile) as f:
        listrows = f.readlines()
        for property in properties:
            try:
                s = [row for row in listrows if re.findall(r'\b%s\b' % (property), row)][0]
            except IndexError:
                logging.error('The property name \'%s\' was not found!' % (property))
                f.close()
                sys.exit()
            s = ''.join(s.split()).split('=')[1]
            if '"' in s: s = s[1:-1]
            MetaDict.update({property: s})
        f.close()
    return MetaDict


# Channel class:
class Channel:

    def __init__(self,name,res=None,transform=None,crs=None,bounds=None,shape=None):
        """
        Constructor:
        :param: name:      [str]                         - the name of raster in GRASS project {required!!!}
        :param: res:       [list[EWres,NSres]]           - resolution of the raster            {default=None}
        :param: transform: [list[parameters]]            - geotransformation of the raster     {default=None}
                transform[0] - top geocoordinate of the raster;
                transform[1] - east-west spatial resolution of the raster;
                transform[2] - the first rotate angle of the raster;
                transform[3] - left geocoordinate of the raster;
                transform[4] - the second rotate angle of the raster;
                transform[5] - north-south spatial resolution of the raster;
        :param: crs:       [PROJ4 format]                - projection of the raster            {default=None}
        :param: bounds:    [dictionary{'N','E','S','W'}] - coordinate of raster's boundary     {default=None}
        :param: shape:     [list[Xsize,Ysize]]           - shape of the raster                 {default=None}
        """
        self._name = name
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
        return self._crs


    def transform(self):
        """
        Getting geotransformation of the channel:
        :return: [list[parameters]]
        """
        return self._transform


    def res(self):
        """
        Getting resolution of the channel:
        :return: [list[EWres,NSres]]
        """
        return self._res


    def shape(self):
        """
        Getting shape of the channel:
        :return: [list[Xsize,Ysize]]
        """
        return self._shape


    def bounds(self):
        """
        Getting bounds of the channel:
        :return: [dictionary{'N','E','S','W'}]
        """
        return self._bounds


#    def getPixel(self,pos=[1,1]):
#        """
#        Getting pixel value:
#        :param pos: [list[X,Y]] - pixel's coordinates {default=[1,1]}
#                                 [1,1] refers to the first pixel
#        :return: [value]
#        """
#        return self._array[pos[0]-1,pos[1]-1]


    def __str__(self):
        """
        Getting string representation for a Channel object:
        :return: [str]
        """
        return ('%(array)s\n'
                '\'res\': %(res)s\n'
                '\'crs\': %(crs)s\n'
                '\'transform\':\n%(transform)s\n'
                '\'shape\': %(shape)s\n'
                '\'bounds\': %(bounds)s\n'
                % {'array':garray.array(self),'res':self._res,'crs':self._crs,'transform':self._transform,
                   'shape':self._shape,'bounds':self._bounds})


# Image class:
class Image:

    def __init__(self,folder=None,channelDict=None,channels='ALL',metadata=None,maps=None,names=None,Imname=None):
        """
        Constructor:
        :param folder:      [str]                 - folder name from which images will be downloaded             {required!!!}
        :param channelDict: [OrderedDict]         - dictionary of channels which will be wrapped in Image object {required!!!}
                                                    (ignored if 'folder' is indicated)
        :param maps:        [list[names]]         - names of images in GRASS project for forming Image object    {required!!!}
                                                    (ignored if 'folder' or 'channelDict' are indicated)
        :param names:       [list[names]]         - names of channels which will be used for images from GRASS   {required!!!}
        :param channels:    [list[channel names]] - channel list which will be choden and downloaded             {default='ALL'}
        :param metadata:    [dictionary]          - the dictionary of metadata properties                        {default=None}
        :param Imname:      [str]                 - image name which will be downloaded from GRASS project       {default=None}
        """
        if folder != None: self.__load(folder=folder,channels=channels)
        elif channelDict != None: self.__construct(channelDict=channelDict,Imname=Imname,metadata=metadata)
        elif (maps != None) and (channels != None): self.__fromProj(maps=maps,names=names,Imname=Imname)
        else:
            logging.error('Image object can\'t be created because of lack required parameters!'
                          'Use \'folder\'/\'channelDict\'/\'maps\'+\'channels\' parameters as required!')
            sys.exit()


    def __fromProj(self,maps,names,Imname):
        """
        Constructor 1: downloading from GRASS project:
        :param maps:   [list[names]] - list of the map names from GRASS project                          {required!!!}
        :param names:  [list[names]] - list of channel names which will be downloaded from GRASS project {required!!!}
        :param Imname: [list[names]] - image name which will be downloaded from GRASS project            {required!!!}
        """
        if type(maps) == str: maps = [maps]
        if type(names) == str: names = [names]
        try:
            iter(maps)
        except:
            logging.error('The \'maps\' attribute does not iterable! Use \'list\', \'tupple\' or another types. '
                          'But found %s.' %(type(maps)))
            sys.exit()
        try:
            iter(names)
        except:
            logging.error('The \'names\' does not iterable! Use \'list\', \'tupple\' or another types. '
                          'But found %s.' %(type(names)))
            sys.exit()
        if len(maps) != len(names):
            logging.error('The count of the \'maps\' and \'names\' should be equal! But found %s and %s.' %(len(maps),len(names)))
            sys.exit()
        chanDict = collections.OrderedDict()
        for i in range(len(maps)):
            name = maps[i]
            parsing = gscript.parse_command('r.info',map=maps[i],flags='ge')
            res = [int(parsing['ewres']),int(parsing['nsres'])]
            shape = [int(parsing['cols']),int(parsing['rows'])]
            bounds = {'N':int(parsing['north']),'S':int(parsing['south']),
                      'W':int(parsing['west']),'E':int(parsing['east'])}
            crs = gscript.read_command('g.proj',flags='jf')
            transform = self.__getTransformParameters(belowright=[bounds['S'],bounds['E']],
                                                      topleft=[bounds['N'],bounds['W']],rastersize=shape,res=res)
            chanDict.update({names[i]:Channel(name=name,res=res,transform=transform,crs=crs,bounds=bounds,shape=shape)})
        self.__construct(channelDict=chanDict,Imname=Imname,metadata=None)
        self._name = Imname


    def __construct(self,channelDict,Imname,metadata=None):
        """
        Constructor 2: wrapping OrderedDict in Image object:
        :param channelDict: [OrderedDict] - dictionary of channels which will be wrapped {requored!!!}
        :param Imname:      [str]         - the name of the image                        {requored!!!}
        :param metadata:    [OrderedDict] - dictionary of metadata properties            {default=None}
        """
        self._bands = collections.OrderedDict(sorted(channelDict.items(), key=lambda x: x[0]))
        self._metadata = metadata
        self._name = Imname


    def __load(self,folder,channels='ALL'):
        """
        Constructor 3: downloading from folder:
        :param folder:   [str]         - name of the folder                             {required!!!}
        :param channels: [list[names]] - list of bands which should be read from folder {default='ALL'}
        """
        try:
            files = os.listdir(folder)
        except WindowsError:
            logging.error('Folder \'%(folder)s\' was not found!' % {'folder': folder})
            sys.exit()
        try:
            metadatafile = [name for name in files if re.findall(r'\w+MTL.txt', name) != []][0]
        except IndexError:
            logging.error('Metadata \'(meta)\' in the folder \'%(folder)s\' was not found!' % {'meta':metadatafile,'folder':folder})
            sys.exit()
        files = [name for name in files if name.endswith(tuple([".TIF", 'TIFF', 'jpg']))]
        if files == []:
            logging.error('No one files was found!')
            sys.exit()
        properties = ['CLOUD_COVER','DATE_ACQUIRED','SUN_ELEVATION','SUN_AZIMUTH']
        metadata = readMeta(metafile=os.path.join(folder,metadatafile),properties=properties)
        self._metadata = metadata
        base = os.path.basename(folder)
        self._name = base
        copyfile(os.path.join(folder,metadatafile),os.path.join(METAPATH,metadatafile))
        ChannelDict = collections.OrderedDict()
        if channels == 'ALL': channels = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','BQA']
        elif type(channels) == str: channels = [channels]
        try:
            iter(channels)
        except:
            logging.error('The \'channels\' attribute does not iterable! Use \'list\', \'tupple\' or another types. '
                          'But found %s.' %(type(channels)))
            sys.exit()
        for filename in files:
            START_TIME_FILE = time.time()
            if re.findall(r'B\w+', filename.split('.')[0])[0] in channels:
                name = filename.split('.')[0]
                input = os.path.join(folder,filename)
                gscript.run_command('r.in.gdal',input=input,output=name,overwrite=True,flags='k')
                parsing = gscript.parse_command('r.info', map=name, flags='ge')
                res = [int(parsing['ewres']), int(parsing['nsres'])]
                shape = [int(parsing['cols']), int(parsing['rows'])]
                bounds = {'N': int(parsing['north']), 'S': int(parsing['south']),
                          'W': int(parsing['west']), 'E': int(parsing['east'])}
                crs = gscript.read_command('g.proj', flags='jf')
                transform = self.__getTransformParameters(belowright=[bounds['S'], bounds['E']],topleft=[bounds['N'],bounds['W']],rastersize=shape,res=res)
                ChannelDict.update({re.findall(r'B\w+', filename.split('.')[0])[0]:Channel(name=name, res=res, transform=transform, crs=crs, bounds=bounds, shape=shape)})
                logging.info('file %(filename)s was succesfuly downloaded\nsession time: --- %(time)s seconds ---\n'
                             % {'filename': filename, 'time': (time.time() - START_TIME_FILE)})
        self._bands = ChannelDict


    def __getGeoCoordinates(X,Y,transform):
        """
        Getting geocordinates of the raster's pixel:
        :param X:         [int]              - X coordinate of raster's pixel     {required!!!}
        :param Y:         [int]              - Y coordinate of raster's pixel     {required!!!}
        :param transform: [list[parameters]] - geotransformation of the raster    {required!!!}
        :return: [dictionary]
        """
        Matrix1 = np.array(transform).reshape(2,3)
        Matrix2 = np.array([1,X,Y]).reshape(3,1)
        Matrix = np.dot(Matrix1,Matrix2).reshape(2,1)
        return {'X': Matrix[0][0], 'Y': Matrix[1][0]}


    __getGeoCoordinates = staticmethod(__getGeoCoordinates)


    def __getTransformParameters(belowright,topleft,rastersize,res):
        """
        Getting full transformation parameters using the raster's size, its spatial resolution and the left-top coordinates:
        :param belowright: []
        :param topleft:    []
        :param rastersize: []
        :param res:        [list[EWres,NSres]]           - resolution of the raster           {default=None}
        :return: [list]
        """
        rotation1 = (belowright[0]-topleft[0]-res[0]*rastersize[0])/belowright[1]
        rotation2 = (belowright[1]-topleft[1]+abs(res[1])*rastersize[1])/belowright[0]
        return [topleft[0],res[0],rotation1,topleft[1],rotation2,-abs(res[1])]


    __getTransformParameters = staticmethod(__getTransformParameters)


    def __str__(self):
        """
        Getting string representation for an Image object:
        :return: [str]
        """
        s = 'Image object:\n{start}\n'
        for key in self._bands:
            s += ('\'channel\': %(channel)s\n'
                  '%(channelblock)s\n'
                  % {'channel': key, 'channelblock': self._bands[key].__str__()})
        s += '\'metadata\': %(metadata)s\n' % {'metadata': self._metadata}
        s += '{end}\n\n'
        return s


    def bandNames(self):
        """
        Getting all band names:
        :return: [list]
        """
        return list(self._bands.keys())


    def select(self, bands):
        """
        Selecting bands from Image object:
        :param bands: [list[names]] - names of bands which will be selected {required!!!}
        :return: [Image object]
        """
        copyImage = copy.deepcopy(self)
        if type(bands) == str: bands = [bands]
        try:
            iter(bands)
        except:
            logging.error('The \'bands\' attribute does not iterable! Use \'list\', \'tupple\' or another types. '
                          'But found %s.' %(type(bands)))
            sys.exit()
        selectedBands = [(key,copyImage._bands[key]) for key in copyImage._bands if key in bands]
        if selectedBands == []:
            logging.error('Chosen bands were not found!')
            sys.exit()
        OtherBands = [other for other in bands if not (other in list(copyImage._bands.keys()))]
        if OtherBands != []:
            logging.error('Bands %s were not found in Image!' %(OtherBands))
            sys.exit()
        return Image(channelDict=collections.OrderedDict(selectedBands),Imname=copyImage._name,metadata=copyImage._metadata)


    def projection(self):
        """
        Getting projection of the image:
        :return: [PROJ4 format]
        """
        if len(self.bandNames())!= 1:
            logging.error('Can\'t get projection from image with more than 1 band! Found: %s' %(len(self.bandNames())))
            sys.exit()
        return self._bands[self.bandNames()[0]].projection()


    def transform(self):
        """
        Getting transform of the image:
        :return: [list[parameters]]
        """
        if len(self.bandNames())!= 1:
            logging.error('Can\'t get transform from image with more than 1 band! Found: %s' %(len(self.bandNames())))
            sys.exit()
        return self._bands[self.bandNames()[0]].transform()


    def res(self):
        """
        Getting resolution of the image:
        :return: [list[EWres,NSres]]
        """
        if len(self.bandNames())!= 1:
            logging.error('Can\'t get resolution from image with more than 1 band! Found: %s' %(len(self.bandNames())))
            sys.exit()
        return self._bands[self.bandNames()[0]].res()


    def shape(self):
        """
        Getting shape of the image:
        :return: [list[Xsize,Ysize]]
        """
        if len(self.bandNames())!= 1:
            logging.error('Can\'t get shape from image with more than 1 band! Found: %s' %(len(self.bandNames())))
            sys.exit()
        return self._bands[self.bandNames()[0]].shape()


    def bounds(self):
        """
        Getting bounds of the image:
        :return: [dictionary{'N','E','S','W'}]
        """
        if len(self.bandNames())!= 1:
            logging.error('Can\'t get bounds from image with more than 1 band! Found: %s' %(len(self.bandNames())))
            sys.exit()
        return self._bands[self.bandNames()[0]].bounds()


#    def getPixel(self,pos=[0,0]):
#        """
#        Getting pixel value:
#        :param pos: [list[X,Y]] - pixel's coordinates {default=[1,1]}
#                                 [1,1] refers to the first pixel
#        :return: [value]
#        """
#        if len(self.bandNames()) != 1:
#            logging.error('Attempt to get pixel value for several bands! Only one band is required! Found: %s' %(len(self.bandNames())))
#            sys.exit()
#        return self._bands[self.bandNames()[0]].getPixel(pos)


    def get(self,prop):
        """
        Getting metadata properties from metadata file:
        :param prop: [list] - names of properties which will be read {required!!!}
        :return: [OrderedDict]
        """
        if type(prop) == str: prop = [prop]
        try:
            iter(prop)
        except:
            logging.error('The \'prop\' attribute does not iterable! Use \'list\', \'tupple\' or another types. '
                          'But found %s.' %(type(prop)))
            sys.exit()
        try:
            return readMeta(metafile=os.path.join(METAPATH,self._metadata['LANDSAT_PRODUCT_ID']),properties=prop)
        except IndexError:
            logging.error('Metadata file does not exist!')
            sys.exit()


    def getDate(self):
        """
        Getting date of image creating:
        :return: [Date]
        """
        DateList = self.get('DATE_ACQUIRED').split('-')
        return datetime.date(int(DateList[0]), int(DateList[1]), int(DateList[2]))


    def first(self):
        """
        Getting the first channel in image:
        :return: [Image object]
        """
        return self.select(self.bandNames()[0])


    def addBands(self,bands):
        """
        Adding new bands in image:
        :param bands: [Image object] - image which will be added as new bands {required!!!}
        :return: [Image object]
        """
        copyImage = copy.deepcopy(self)
        copyDict = copyImage._bands
        try:
            for key, band in bands._bands.items(): copyDict.update({key: band})
            return Image(channelDict=copyDict,metadata=copyImage._metadata)
        except AttributeError:
            logging.error('Attempt to add not \'Image\' object!')
            sys.exit()


    def rename(self,newName):
        """
        Renaming all bands in image:
        :param newName: [list] - new names for bands in image {required!!!}
        :return: [Image object]
        """
        if type(newName) == str: newName = newName.split(',')
        try:
            iter(newName)
        except:
            logging.error('The \'newName\' attribute does not iterable! Use \'list\', \'tupple\' or another types. '
                          'But found %s.' %(type(newName)))
            sys.exit()
        if len(newName) != len(self.bandNames()):
            logging.error(
                'Count of bands does not match to count of new names! For correct renaming use as new names as bands you have selected! '
                'Found %(selected)s bands and %(names)s names!' % {'selected': len(self.bandNames()),
                                                                   'names': len(newName)})
            sys.exit()
        copyImage = copy.deepcopy(self)
        for i in range(len(self.bandNames())):
            gscript.run_command('g.rename',raster=(self._bands[self.bandNames()[i]]._name,newName[i]),overwrite=True)
        copyDict = collections.OrderedDict(
                map(lambda name, key: (name,copyImage._bands[key]), newName, list(copyImage._bands.keys())))
        return Image(channelDict=copyDict,metadata=copyImage._metadata)


    def Save(self,folder,name):
        """
        Saving image to file:
        :param folder: [str] - path to save image    {required!!!}
        :param name:   [str] - name for saving image {required!!!}
        """
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


    def TOAR(self):
        """
        Evaluating top of atmosphere reflectance:
        :return: [Image object]
        """
        name = self._name
        metfile = os.path.join(METAPATH,'%s_MTL.txt' %(name))
        num = [key[1:] for key in self.bandNames()]
        prop1 = ['REFLECTANCE_MULT_BAND_%s' %(i) for i in num]
        prop2 = ['REFLECTANCE_ADD_BAND_%s' % (i) for i in num]
        prop = prop1 + prop2
        try:
            dict = readMeta(metafile=metfile, properties=prop)
        except IndexError:
            logging.error('Metadata file does not exist!')
            sys.exit()
        rasters = []
        for i in num:
            REFLECTANCE_MULT = dict['REFLECTANCE_MULT_BAND_%s' %(i)]
            REFLECTANCE_ADD = dict['REFLECTANCE_ADD_BAND_%s' %(i)]
            input = '%(name)s_B%(i)s' %{'name':name,'i':i}
            output = '%(name)s_toar_B%(i)s' %{'name':name,'i':i}
            expression = '%(output)s=%(A)s * %(input)s + %(B)s' %{'output':output,'A':REFLECTANCE_MULT,'input':input,'B':REFLECTANCE_ADD}
            logging.info('Calculating TOAR for image %(name)s: band B%(i)s' %{'name':name,'i':i})
            gscript.run_command('r.mapcalc',expression=expression,overwrite=True)
            rasters.append(output)
        Imname = name + '_toar'
        return Image(maps=rasters,names=self.bandNames(),Imname=Imname)


    def remove(self):
        """
        Removing images from GRASS project:
        """
        for band in self._bands.values():
            name = band._name
            gscript.run_command('g.remove',type='raster',name=name,flags='fb')


    def FMask(self,radius=3):
        """
        Calculating FMask algorithm:
        :param radius: [value] - buffer radius for influencing pixels {default=3}
        :return: [Image object]
        """
        name = self._name + '_BQA'
        CloudMask = 'CloudMask=((int(%(BQA)s) & 32)!=0)&((int(%(BQA)s) & 64)!=0)' %{'BQA':name}
        CloudShadowMask = 'CloudShadowMask=((int(%(BQA)s) & 128) !=0)&((int(%(BQA)s) & 256)!=0)' %{'BQA':name}
        SnowMask = 'SnowMask=((int(%(BQA)s) & 512)!=0)&((int(%(BQA)s) & 1024)!=0)' %{'BQA':name}
        CirrusMask = 'CirrusMask=((int(%(BQA)s & 2048)!=0)&((int(%(BQA)s) & 4096)!=0)' %{'BQA':name}
        expression = 'ImageMask=eval([%(e1)s,%(e2)s,%(e3)s,%(e4)s](round(%(e1)s | %(e2)s | %(e3)s | %(e4)s)))' \
                     %{'e1':CloudMask,'e2':CloudShadowMask,'e3':SnowMask,'e4':CirrusMask}
        gscript.run_command('r.mapcalc', expression=expression, overwrite=True)
        output = name + '_FMask'
        gscript.run_command('r.grow',input='ImageMask',output=output,radius=radius,overwrite=True)
        gscript.run_command('g.remove',type='raster',name=['ImageMask'],flags='fb')
        return Image(maps=output, names='FMask', Imname=output)


    def updateMask(self,mask):
        name = self._name
        mask = mask._name
        rasters = []
        for band in self.bandNames():
            input = name + '_' + band
            output = name + '_masked_' + band
            expression = '%(output)s=%(input)s*%(mask)s' %{'output':output,'input':input,'mask':mask}
            gscript.run_command('r.mapcalc',expression=expression,overwrite=True)
            rasters.append(output)
        imNew = Image(maps=rasters,names=self.bandNames(),Imname='%s' %(name + '_masked'))


# class ImageCollection:
class ImageCollection:

    def __init__(self,folder=None,channels='ALL',ImageDict=None):
        """
        Constructor:
        :param folder:    [str]         - folder name from which images will be downloaded             {required!!!}
        :param ImageDict: [OrderedDict] - dictionary of channels which will be wrapped in Image object {required!!!}
                                          (ignored if 'folder' is indicated)
        :param channels:  [list[names]] - channel list which will be choden and downloaded {default='ALL'}
        """
        if folder != None: self.__load(folder=folder,channels=channels)
        elif ImageDict != None: self.__construct(ImageDict=ImageDict)
        else:
            logging.error('ImageCollection object can\'t be created because of lack required parameters!'
                          'Use \'folder\'/\'ImageDict\' parameters as required!')
            sys.exit()


    def __construct(self,ImageDict):
        """
        Constructor 1: wrapping OrderedDict in ImageCollection object:
        :param ImageDict: [OrderedDict] - dictionary of channels which will be wrapped in Image object {required!!!}
        """
        try:
            self._images = collections.OrderedDict(sorted(ImageDict.items(), key=lambda x: x[0]))
        except AttributeError:
            logging.error('Failed to create object! Type of argument must be \'OrderedDict\', but %s was found!' %(ImageDict.__class__))
            sys.exit()


    def __load(self,folder,channels='ALL'):
        """
        Constructor 2: downloading from folder:
        :param folder:   [str]         - folder name from which images will be downloaded {required!!!}
        :param channels: [list[names]] - channel list which will be choden and downloaded {default='ALL'}
        """
        try:
            files = os.listdir(folder)
        except WindowsError:
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
        """
        Getting string representation for an ImageCollection object:
        :return: [str]
        """
        i = 1
        s = 'ImageCollection object:\n{start}\n'
        for key in self._images:
            s += '%(num)s) %(name)s  %(bands)s\n' % {'num': i, 'name': key, 'bands': self._images[key].bandNames()}
            i += 1
        s += '{end}\n\n'
        return s


    def size(self):
        """
        Getting size of the image collection:
        :return: [int]
        """
        return len(self._images.keys())


    def first(self):
        """
        Getting the first image from image collection:
        :return: [Image object]
        """
        copyImage = copy.deepcopy(self._images[list(self._images.keys())[0]])
        return copyImage


    def get(self,num):
        """
        Getting the num-th image from image collection:
        :param num: [int] - the number of image will be returned from ImageCollection {required!!!}
        :return: [Image object]
        """
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
        """
        Mapping a function over image collection:
        :param func: [func] - function which will be mapped over ImageCollection {required!!!}
        :return: [ImageCollection object]
        """
        copyCollection = copy.deepcopy(self._images)
        newDict = collections.OrderedDict(map(lambda key,im: (key,func(im)), list(copyCollection.keys()),list(copyCollection.values())))
        return ImageCollection(ImageDict=newDict)


    def select(self,bands):
        """
        Selecting bands in every image of the image collection:
        :param bands: [list[names]] - names of bands which will be selected over ImageCollection {required!!!}
        :return: [ImageCollection oblect]
        """
        def _f(im):
            return im.select(bands)
        return self.Map(_f)


    def filterDate(self,dateStart,dateEnd):
        """
        Filtering image collection by dates:
        :param dateStart: [str(yy-mm-dd)] - start date  {required!!!}
        :param dateEnd:   [str(yy-mm-dd)] - finish date {required!!!}
        :return: [ImageCollection object]
        """
        dateStartList = dateStart.split('-')
        dateEndList = dateEnd.split('-')
        dateStartDateform = datetime.date(int(dateStartList[0]), int(dateStartList[1]), int(dateStartList[2]))
        dateEndDateform = datetime.date(int(dateEndList[0]), int(dateEndList[1]), int(dateEndList[2]))
        copyCollection = copy.deepcopy(self._images)
        newDict = collections.OrderedDict([(key, im) for key, im in copyCollection.items()
                                           if ((im.getDate() >= dateStartDateform) and (im.getDate() <= dateEndDateform))])
        return ImageCollection(ImageDict=newDict)


    def TOAR(self):
        """
        Evaluating top of atmosphere reflectance:
        :return: [ImageCollection object]
        """
        def _f(image):
            return image.TOAR()
        return self.Map(_f)


    def remove(self):
        """
        Removing image collection from GRASS project:
        """
        for image in self._images.values():
            image.remove()


    def FMask(self,radius=3):
        """
        Calculating FMask algorithm:
        :param radius: [value] - buffer radius for influencing pixels {default=3}
        :return: [ImageCollection object]
        """
        def _f(image):
            return image.FMask(radius=radius)
        return self.Map(_f)

    def reducebyGrass(self,type):
        """
        Reducing ImageCollection for calculating count of non-zero or non Nan pixels:
        :param: type [str] - the method which will be used for reducing ImageCollection {required!!!}
        :return: [Image object]
        """
        numChan = [len(im.bandNames()) for im in self._images.values()]
        if min(numChan) != max(numChan):
            logging.error('Reducing can\'t be applied to ImageCollection with images having different count of bands!')
            sys.exit()
        try:
            input = [im._name for im in self._images.values()]
        except AttributeError:
            logging.error('Reducing ImageCollection was failed! Some rasters were not downloaded to GRASS project.')
            sys.exit()
        method = type
        rasters = []
        for i in range(len(self.first().bandNames())):
            output = 'Count%s' %(i)
            gscript.run_command('r.series',input=input,output=output,method=method,overwrite=True)
            rasters.append(output)
        return Image(maps=rasters,names=rasters,Imname='ReduceCount')


    def updateMask(self, mask):
        def _f(image):
            return image.updateMask()
        return self.Map(_f)




def TMaskAlgorithm(folder,BACKUP_ALG_THRESHOLD=15,
                          RADIUS_BUFF=3,
                          T_MEDIAN_THRESHOLD=0.04,
                          GREEN_CHANNEL_PURE_SNOW_THRESHOLD=0.4,
                          NIR_CHANNEL_PURE_SNOW_THRESHOLD=0.12,
                          BLUE_CHANNEL_THRESHOLD=0.04,
                          NIR_CHANNEL_CLOUD_SNOW_THRESHOLD=0.04,
                          NIR_CHANNEL_SHADOW_CLEAR_THRESHOLD=-0.04,
                          SWIR1_CHANNEL_SHADOW_CLEAR_THRESHOLD=-0.04):

    # download 'B3' (blue channel), 'B5' (Near Infrared (NIR)), 'B6' (Shortwave Infrared (SWIR) 1) and 'BQA' channels:
    logging.info('downloading blue channel, Near Infrared (NIR), Shortwave Infrared (SWIR) 1 and Mask channels to GRASS:')
    collection = ImageCollection(folder=folder,channels=['B3','B5','B6','BQA'])
    collection_main = collection.select(['B3','B5','B6'])
    collection_BQA = collection.select('BQA')

    # the size of the collection:
    ImageCounts = collection.size()
    text1 = 'Total number of images: %s' % (ImageCounts)
    text2 = 'Warning: You have less than %s images!' % (BACKUP_ALG_THRESHOLD)
    if ImageCounts >= BACKUP_ALG_THRESHOLD: print(text1)
    else: print(text2)

    # apply TOAR convertion:
    logging.info('Applying TOAR convertion:')
    collection_toar = collection_main.select(['B3','B5','B6']).TOAR()

    # remove source images:
    collection_main.remove()

    # compute FMask algorithm:
    #logging.info('Computing FMask algorithm:')
    #collection_FMask = collection_BQA.FMask(radius=RADIUS_BUFF)

    return collection_toar




        # reducing collection_FMask:
        #logging.info('Computing count of non-zero raster of FMask collection:')
        #ConditionMap = collection_FMask.reducebyGrass(type='sum')

        # detect which part of data should be used for BackUp algorithm:
        #logging.info('Detecting which part of data should be used for BackUp algorithm:')
        #expression = 'ClearSmall=%(rast)s>(%(count)s-%(tresh)s)' % {'rast': 'Count0', 'count': ImageCounts,
        #                                                            'tresh': BACKUP_ALG_THRESHOLD}
        #gscript.run_command('r.mapcalc', expression=expression, overwrite=True)
        #ClearSmall = Image(maps='ClearSmall',names='Count',Imname='ClearSmall')
        #ClearSmall._rasters = ['ClearSmall']

        # forming non-snow pixels:
        #def BackUp_mask(image):
        #    base = image._metadata['LANDSAT_PRODUCT_ID']
        #    output = base + '_nonSnow'
        #    expression = 'not(%(output)s=((int(%(base)s_BQA)) & 512)!=0)&((int(%(base)s_BQA)) & 1024)!=0))' %{'base': base,'output':output}
        #    newIm = Image(maps=output, names='Mask', Imname=output)
        #    newIm._rasters = output
        #   im = image.select('B3').updateMask(ClearSmall).updateMask(newIm)
        #    newIm.remove()
        #    return im

        # set BackUp mask for image collection and exclude all snow pixels:
        #logging.info('Setting BackUp mask for image collection and exclude all snow pixels:')
        #B3Channel_masked = collection_toar.Map(BackUp_mask)

        # calculate mediana for potential clear pixels in BackUp approach:
        #logging.info('Calculating mediana for potential clear pixels in BackUp approach:')
        #mediana = B3Channel_masked.reducebyGrass(type='median')

        # delete all 'BQA' channels and 'B3Channel_masked':
        #collection_BQA.remove()
        #B3Channel_masked.remove()







        # create mask for BackUp Algorithm using FMask:
        #BackUpAlgorithm = def(image):
        #    image1 = image.select('B3');
        #    image2 = image.select('BQA');
        #    Mask = image1.gt(Mediana.add(ee.Image(T_MEDIAN_THRESHOLD)));


        #
        #CloudMask = image2.bitwiseAnd(_iBit_(5)).neq(0).bitwiseAnd(image2.bitwiseAnd(_iBit_(6)).neq(0));
        #SnowMask = image2.bitwiseAnd(_iBit_(9)).neq(0).bitwiseAnd(image2.bitwiseAnd(_iBit_(10)).neq(0));
        #Composite = CloudMask.multiply(3).add(SnowMask.multiply(2));
        #Mask = Mask.multiply(Composite);
        #return Mask;




