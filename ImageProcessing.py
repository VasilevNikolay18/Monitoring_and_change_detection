# import some useful libraries:
import os, sys, collections, re, logging, math, datetime
from shutil import copyfile
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
import numpy as np
import grass.script as gscript
from grass.script import array as garray
from grass.pygrass.modules.shortcuts import general as g
from grass.pygrass.modules.shortcuts import raster as r

# get current environment:
environ = gscript.parse_command('g.gisenv')
GISDBASE = environ['GISDBASE'][1:-2]
MAPSET = environ['MAPSET'][1:-2]
LOCATION_NAME = environ['LOCATION_NAME'][1:-2]
PROJECTPATH = os.path.join(GISDBASE, LOCATION_NAME, MAPSET)
METAPATH = os.path.join(PROJECTPATH, 'META')
if not (os.path.isdir(METAPATH)): os.mkdir(METAPATH)

# downloading image to GRASS:
def loadImage(folder, channels='ALL'):
    files = os.listdir(folder)
    metadatafile = [name for name in files if name.endswith("MTL.txt")][0]
    copyfile(os.path.join(folder, metadatafile), os.path.join(METAPATH, metadatafile))
    files = [name for name in files if name.endswith(tuple([".TIF", 'TIFF', 'jpg', ".tif"]))]
    if channels == 'ALL':
        channels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'BQA']
    elif type(channels) == str:
        channels = [channels]
    for filename in files:
        file = str(filename.split('.')[0])
        Imname = '_'.join(file.split('_')[:-1])
        Bandname = file.split('_')[-1]
        if Bandname in channels:
            input = os.path.join(folder, filename)
            output = Imname + '.' + Bandname
            r.in_gdal(input=input, output=output, overwrite=True, flags='k')

# downloading image collection to GRASS:
def loadCollection(folder, channels='ALL'):
    dirlist = [item for item in os.listdir(folder) if os.path.isdir(os.path.join(folder, item))]
    for foldname in dirlist: loadImage(folder=os.path.join(folder, foldname), channels=channels)

# reading metadata:
def readMeta(metapath, properties):
    if type(properties) == str: properties = [properties]
    MetaDict = collections.OrderedDict()
    f = open(metapath)
    listrows = f.readlines()
    for property in properties:
        s = [row for row in listrows if re.findall(r'\b%s\b' % (property), row)][0]
        s = ''.join(s.split()).split('=')[1]
        if '"' in s: s = s[1:-1]
        MetaDict.update({property: s})
        f.close()
    return MetaDict

# evaluating top of atmosphere reflectance:
def TOAR(images):
    for name in images:
        basename = name.split('.')[0]
        num = name.split('.')[1]
        i = num[1:]
        metapath = os.path.join(METAPATH, basename + '_MTL.txt')
        prop1 = 'REFLECTANCE_MULT_BAND_%s' % (i)
        prop2 = 'REFLECTANCE_ADD_BAND_%s' % (i)
        properties = [prop1, prop2]
        metadata = readMeta(metapath=metapath, properties=properties)
        A = metadata['REFLECTANCE_MULT_BAND_%s' % (i)]
        B = metadata['REFLECTANCE_ADD_BAND_%s' % (i)]
        output = basename + '.' + num + '_toar'
        expression = '%(output)s=%(A)s * %(input)s + %(B)s' % {'output': output, 'A': A, 'input': name, 'B': B}
        r.mapcalc(expression=expression, overwrite=True)

# calculating julian date using gregorian date:
def JulianDate(year, month, day):
    a = math.floor((14 - month) / 12)
    y = year + 4800 - a
    m = month + a * 12 - 3
    return day + math.floor((m * 153 + 2) / 5) + 365 * y + math.floor(y / 4) - math.floor(y / 100) + math.floor(
        y / 400) - 32045

# getting property from metadata:
def get(image, prop):
    metapath = os.path.join(METAPATH, image.split('.')[0] + '_MTL.txt')
    return readMeta(metapath=metapath, properties=prop)

# getting data from metadata
def getDate(image):
    DateList = get(image=image, prop='DATE_ACQUIRED')['DATE_ACQUIRED'].split('-')
    return datetime.date(int(DateList[0]), int(DateList[1]), int(DateList[2]))

# selecting channels from GRASS:
def selectFromGrass(collection,channels):
    if type(channels) == str: channels = [channels]
    return sorted([im for im in collection if im.split('.')[1] in channels])

# selecting channel:
def select(collection,channel):
    return sorted([im[im.index(im[0].split('.')[0] + '.' + channel)] for im in collection])

# getting reconstruction using regression analyzis:
def reconstruction(images,coeffs,name):
    expression = '%s=' %(name)
    for m1, m2 in zip(images, coeffs):
        expression += '+%(m1)s*%(m2)s' % {'m1': m1, 'm2': m2}
    r.mapcalc(expression=expression, overwrite=True)
    g.remove(type='raster', name=images, flags='fb')
    return name

# getting coefficients for RLM:
def coefficients(image,N):
    Imdate = getDate(image)
    Julian = JulianDate(year=Imdate.year, month=Imdate.month, day=Imdate.day)
    pi = math.pi
    T = 365
    coeff1 = 1
    coeff2 = math.cos(2 * pi * Julian / T)
    coeff3 = math.sin(2 * pi * Julian / T)
    coeff4 = math.cos(2 * pi * Julian / (T * N))
    coeff5 = math.sin(2 * pi * Julian / (T * N))
    return np.array([coeff1, coeff2, coeff3, coeff4, coeff5])

# getting time for series images:
def timeSeries(collection):
    times = [getDate(im) for im in collection]
    minDate = min(times)
    maxDate = max(times)
    return math.ceil(relativedelta(maxDate, minDate).years)

# getting components and coefficients from RLM:
def RobustRegression(collection):
    parsing = gscript.parse_command('r.info', map=collection[0], flags='ge')
    shape = [int(parsing['cols']), int(parsing['rows'])]
    rasters = []
    for rast in collection:
        out = os.path.join(METAPATH,rast.split('.')[0] + '_coordinates.txt')
        r.out_xyz(input=rast, separator='space', output=out)
        f = open(out, 'r')
        rasters.append(f.readlines())
        f.close()
        os.remove(out)
    N = timeSeries(collection)
    X_row = [coefficients(im,N) for im in collection]
    X_matrix = np.array(X_row).reshape(len(rasters), 5)
    result = np.zeros((5, shape[0], shape[1]))
    for i in range(shape[1]):
        for j in range(shape[0]):
            Y_matrix = np.array([rast[i*shape[1]+j].split(' ')[2] for rast in rasters], dtype='f')
            try:
                rlm_model = sm.RLM(endog=Y_matrix, exog=sm.add_constant(X_matrix), M=sm.robust.norms.HuberT(), missing='drop')
                rlm_results = rlm_model.fit()
                coeff = np.array(rlm_results.params)
            except:
                coeff = np.empty(5)
                coeff[:] = np.nan
            result[:,j,i] = coeff
    collection_new = []
    #for lay in range(5):
    #    name = 'coeff.' + str(lay)
    #    map = garray.array()
    #    for i in range(shape[1]):
    #        for j in range(shape[0]):
    #            map[j,i] = result[lay,j,i]
    #    map.write(mapname=name, overwrite=True)
    #    collection_new.append(name)
    return X_row#collection_new, X_row

# TMask algorithm:
def TMaskAlgorithm(rasters, BACKUP_ALG_THRESHOLD=4, RADIUS_BUFF=3, T_MEDIAN_THRESHOLD=0.04,
                   GREEN_CHANNEL_PURE_SNOW_THRESHOLD=0.4, NIR_CHANNEL_PURE_SNOW_THRESHOLD=0.12,
                   BLUE_CHANNEL_THRESHOLD=0.04, NIR_CHANNEL_CLOUD_SNOW_THRESHOLD=0.04,
                   NIR_CHANNEL_SHADOW_CLEAR_THRESHOLD=-0.04, SWIR1_CHANNEL_SHADOW_CLEAR_THRESHOLD=-0.04):
    """
    # sorting:
    B3 = selectFromGrass(rasters,''B3_toar'')
    #B5 = selectFromGrass(rasters,''B5_toar'')
    #B6 = selectFromGrass(rasters,''B6_toar'')
    BQA = selectFromGrass(rasters,''BQA'')
    #images = [[ch1,ch2,ch3,ch4] for (ch1,ch2,ch3,ch4) in zip(B3,B5,B6,BQA)]
    images = [[ch1, ch2] for ch1, ch2 in zip(B3, BQA)]

    # the size of the collection:
    ImageCounts = len(images)
    text1 = 'Total number of images: %s' % (ImageCounts)
    text2 = 'Warning: You have less than %s images!' % (BACKUP_ALG_THRESHOLD)
    if ImageCounts >= BACKUP_ALG_THRESHOLD:
        logging.info(text1)
    else:
        logging.info(text2)

    # FMask, composite and non-snow masks:
    def FMask(image):
        baseName = image[0].split('.')[0]
        image_BQA = image[image.index(baseName+'.BQA')]
        raster_FMask = baseName + '.FMask'
        raster_nonSnow = baseName + '.nonSnow'
        raster_composite = baseName + '.Composite'
        expression = 'eval(BQA_int=int(%(BQA)s), ' \
                     'Clouds=((BQA_int & 32)!=0)&((BQA_int & 64)!=0), ' \
                     'CloudShadows=((BQA_int & 128)!=0)&((BQA_int & 256)!=0), ' \
                     'Snow=((BQA_int & 512)!=0)&((BQA_int & 1024)!=0), ' \
                     'Cirrus=((BQA_int & 2048)!=0)&((BQA_int & 4096)!=0)); ' \
                     '%(out1)s=Clouds || CloudShadows || Snow || Cirrus; ' \
                     '%(out2)s=not(Snow); ' \
                     '%(out3)s=(Clouds || Cirrus)*3 + Snow*2' \
                     %{'BQA':image_BQA, 'out1': raster_FMask, 'out2': raster_nonSnow, 'out3': raster_composite}
        r.mapcalc(expression=expression, overwrite=True)
        image.append(raster_composite)
        image.append(raster_nonSnow)
        g.remove(type='raster', name=image_BQA, flags='fb')
        image.pop(image.index(image_BQA))
        return raster_FMask
    FMask_collection = list(map(FMask,images))

    # reducing FMask collection:
    r.series(input=FMask_collection, output='ConditionMap.const', method='sum', overwrite=True)
    ConditionMap = 'ConditionMap.const'
    map(lambda im: g.remove(type='raster', name=im, flags='fb'), FMask_collection)

    # detect which part of data should be used for BackUp algorithm:
    expression = 'ClearSmall.const=%(im)s>%(thresh)s' % {'im': ConditionMap, 'thresh': BACKUP_ALG_THRESHOLD}
    r.mapcalc(expression=expression, overwrite=True)
    ClearSmall = 'ClearSmall.const'
    g.remove(type='raster', name=ConditionMap, flags='fb')

    # forming non-snow pixels:
    def BackUp_mask(image):
        baseName = image[0].split('.')[0]
        raster_out = baseName + '.B3_masked'
        B3_toar = image[image.index(baseName + '.B3_toar')]
        NonSnow = image[image.index(baseName + '.nonSnow')]
        expression = '%(out)s=%(inp1)s*%(inp2)s*%(inp3)s' \
                     % {'out': raster_out, 'inp1': B3_toar, 'inp2': NonSnow, 'inp3': ClearSmall}
        r.mapcalc(expression=expression, overwrite=True)
        g.remove(type='raster', name=NonSnow, flags='fb')
        image.pop(image.index(NonSnow))
        return raster_out
    B3Channel_masked = list(map(BackUp_mask,images))

    # calculate mediana for potential clear pixels in BackUp approach:
    r.series(input=B3Channel_masked, output='Mediana.const', method='median', overwrite=True)
    Mediana = 'Mediana.const'
    map(lambda im: g.remove(type='raster', name=im, flags='fb'), B3Channel_masked)

    # BackUp algorithm:
    def BackUpAlgorithm(image):
        baseName = image[0].split('.')[0]
        raster_out = baseName + '.BackUpMask'
        B3_toar = image[image.index(baseName + '.B3_toar')]
        Composite = image[image.index(baseName + '.Composite')]
        expression = 'eval(Mask=%(B3)s*%(Clearsmall)s, ' \
                     'Threshold=%(Mediana)s+%(Thresh)s); ' \
                     '%(output)s=(Mask>Threshold)*%(Composite)s;' \
                     %{'B3': B3_toar, 'Clearsmall': ClearSmall, 'Mediana': Mediana, 'Thresh': T_MEDIAN_THRESHOLD,
                       'output': raster_out, 'Composite': Composite}
        image.append(raster_out)
        r.mapcalc(expression=expression, overwrite=True)
        g.remove(type='raster', name=Composite, flags='fb')
        image.pop(image.index(Composite))
        return image
    images = list(map(BackUpAlgorithm, images))
    g.remove(type='raster', name=Mediana, flags='fb')

    # create mask for TMask algorithm:
    def TMaskp_mask(image):
        baseName = image[0].split('.')[0]
        raster_out = baseName + '.B3_masked'
        B3_toar = image[image.index(baseName + '.B3_toar')]
        BackUpMask = image[image.index(baseName + '.BackUpMask')]
        expression = 'eval(Clear1=not(%(BackUpMask)s), ' \
                     'Clear2=not(%(ClearSmall)s)); ' \
                     '%(out)s=%(B3)s*(Clear1+Clear2)' \
                     % {'BackUpMask': BackUpMask, 'ClearSmall': ClearSmall, 'out': raster_out, 'B3': B3_toar}
        r.mapcalc(expression=expression, overwrite=True)
        image.append(raster_out)
        return image
    images = list(map(TMaskp_mask, images))
    g.remove(type='raster', name=ClearSmall, flags='fb')
    """
    im1 = 'LC08_L1TP_112025_20130429_20170505_01_T1.B3_toar'
    im2 = 'LC08_L1TP_112025_20130429_20170505_01_T1.B3_masked'
    im3 = 'LC08_L1TP_112025_20130429_20170505_01_T1.BackUpMask'
    image1 = [im2,im1,im3]
    im1 = 'LC08_L1TP_112025_20141126_20170417_01_T1.B3_toar'
    im2 = 'LC08_L1TP_112025_20141126_20170417_01_T1.B3_masked'
    im3 = 'LC08_L1TP_112025_20141126_20170417_01_T1.BackUpMask'
    image2 = [im2,im1,im3]
    im1 = 'LC08_L1TP_112025_20150521_20170408_01_T1.B3_toar'
    im2 = 'LC08_L1TP_112025_20150521_20170408_01_T1.B3_masked'
    im3 = 'LC08_L1TP_112025_20150521_20170408_01_T1.BackUpMask'
    image3 = [im2, im1, im3]
    im1 = 'LC08_L1TP_112025_20131022_20170429_01_T1.B3_toar'
    im2 = 'LC08_L1TP_112025_20131022_20170429_01_T1.B3_masked'
    im3 = 'LC08_L1TP_112025_20131022_20170429_01_T1.BackUpMask'
    image4 = [im2, im1, im3]
    im1 = 'LC08_L1TP_112025_20140705_20170421_01_T1.B3_toar'
    im2 = 'LC08_L1TP_112025_20140705_20170421_01_T1.B3_masked'
    im3 = 'LC08_L1TP_112025_20140705_20170421_01_T1.BackUpMask'
    image5 = [im2, im1, im3]
    im1 = 'LC08_L1TP_112025_20150724_20180204_01_T1.B3_toar'
    im2 = 'LC08_L1TP_112025_20150724_20180204_01_T1.B3_masked'
    im3 = 'LC08_L1TP_112025_20150724_20180204_01_T1.BackUpMask'
    image6 = [im2, im1, im3]
    im1 = 'LC08_L1TP_112025_20141009_20170418_01_T1.B3_toar'
    im2 = 'LC08_L1TP_112025_20141009_20170418_01_T1.B3_masked'
    im3 = 'LC08_L1TP_112025_20141009_20170418_01_T1.BackUpMask'
    image7 = [im2, im1, im3]
    im1 = 'LC08_L1TP_112025_20150113_20170414_01_T1.B3_toar'
    im2 = 'LC08_L1TP_112025_20150113_20170414_01_T1.B3_masked'
    im3 = 'LC08_L1TP_112025_20150113_20170414_01_T1.BackUpMask'
    image8 = [im2, im1, im3]
    im1 = 'LC08_L1TP_112025_20150403_20170411_01_T1.B3_toar'
    im2 = 'LC08_L1TP_112025_20150403_20170411_01_T1.B3_masked'
    im3 = 'LC08_L1TP_112025_20150403_20170411_01_T1.BackUpMask'
    image9 = [im2, im1, im3]
    im1 = 'LC08_L1TP_112025_20150622_20170407_01_T1.B3_toar'
    im2 = 'LC08_L1TP_112025_20150622_20170407_01_T1.B3_masked'
    im3 = 'LC08_L1TP_112025_20150622_20170407_01_T1.BackUpMask'
    image10 = [im2, im1, im3]
    images = [image1,image2,image3,image4,image5,image6,image7,image8,image9,image10]

    # regression for blue channel:
    B3 = select(images,'B3_masked')
    RLR_B3_maps,RLR_B3_coeffs = RobustRegression(B3)
    #RLR_B3_recon = reconstruction(RLR_B3_maps,RLR_B3_coeffs,'reconstruction.B3')




    return RLR_B3_maps