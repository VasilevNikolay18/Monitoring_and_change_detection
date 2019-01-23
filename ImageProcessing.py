# import some useful libraries:
import os, sys, collections, re, logging, math, datetime
from shutil import copyfile
from dateutil.relativedelta import relativedelta
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
    image = []
    for filename in files:
        file = str(filename.split('.')[0])
        Imname = '_'.join(file.split('_')[:-1])
        Bandname = file.split('_')[-1]
        if Bandname in channels:
            input = os.path.join(folder, filename)
            output = Imname + '.' + Bandname
            image.append(output)
            r.in_gdal(input=input, output=output, overwrite=True, flags='k')
    return sorted(image)

# downloading image collection to GRASS:
def loadCollection(folder, channels='ALL'):
    dirlist = [item for item in os.listdir(folder) if os.path.isdir(os.path.join(folder, item))]
    collection = []
    for foldname in dirlist:
        collection.append(loadImage(folder=os.path.join(folder, foldname), channels=channels))
    return collection

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
def TOAR(images, bands):
    for im in images:
        for band in bands:
            basename = im[0].split('.')[0]
            name = basename + '.' + band
            i = band[1:]
            metapath = os.path.join(METAPATH, basename + '_MTL.txt')
            prop1 = 'REFLECTANCE_MULT_BAND_%s' % (i)
            prop2 = 'REFLECTANCE_ADD_BAND_%s' % (i)
            properties = [prop1, prop2]
            metadata = readMeta(metapath=metapath, properties=properties)
            A = metadata['REFLECTANCE_MULT_BAND_%s' % (i)]
            B = metadata['REFLECTANCE_ADD_BAND_%s' % (i)]
            output = basename + '.' + band + '_toar'
            im.append(output)
            expression = '%(output)s=%(A)s * %(input)s + %(B)s' % {'output': output, 'A': A, 'input': name, 'B': B}
            r.mapcalc(expression=expression, overwrite=True)
            delete(im, band)

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

# selecting channel from collection:
def selectFromCollection(collection,channel):
    return sorted([im[im.index(im[0].split('.')[0] + '.' + channel)] for im in collection])

# selecting channel from image:
def selectFromImage(image,channel):
    return image[image.index(image[0].split('.')[0] + '.' + channel)]

# getting time for series images:
def timeSeries(collection):
    times = [getDate(im) for im in collection]
    return math.ceil(relativedelta(max(times), min(times)).years)

# getting residuals:
def getResidual(image,reconstruction,channel):
    basename = image[0].split('.')[0]
    im = selectFromImage(image, channel)
    raster_out = basename + '.' + channel + '_residual'
    expression = '%(out)s=%(im1)s-%(im2)s;' %{'out':raster_out, 'im1':im, 'im2':reconstruction}
    r.mapcalc(expression=expression, overwrite=True)
    image.append(raster_out)

# getting components and coefficients from RLM:
def RobustRegression(collection,band,fet,dod,order,delta,iterates):
    suff = '_lwr'
    output = selectFromCollection(collection,band)
    for iter in range(iterates):
        input = output
        r.series_lwr(input=input, suffix='_lwr', order=order, fet=fet, dod=dod, delta=delta, flags='lh')
        output = [im + suff for im in input]
        if iter != 0:
            g.remove(type='raster', name=input, flags='fb')
        if iter == iterates-1:
            for im, im_new in zip(collection, output):
                im.append(im_new)

# FMask, composite and non-snow masks:
def FMask(image, radius):
    baseName = image[0].split('.')[0]
    image_BQA = selectFromImage(image,'BQA')
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
    #r.grow(input=raster_FMask, output=raster_FMask, radius=radius, overwrite=True)
    image.append(raster_composite)
    image.append(raster_nonSnow)
    return raster_FMask

# deleting image channel:
def delete(image, channel):
    im = selectFromImage(image, channel)
    g.remove(type='raster', name=im, flags='fb')
    image.pop(image.index(im))

# forming non-snow pixels:
def BackUp_mask(image,ClearSmall):
    baseName = image[0].split('.')[0]
    raster_out = baseName + '.B3_masked'
    B3_toar = selectFromImage(image,'B3_toar')
    NonSnow = selectFromImage(image,'nonSnow')
    expression = '%(out)s=%(inp1)s*%(inp2)s*%(inp3)s' \
                 % {'out': raster_out, 'inp1': B3_toar, 'inp2': NonSnow, 'inp3': ClearSmall}
    r.mapcalc(expression=expression, overwrite=True)
    image.append(raster_out)

# BackUp algorithm:
def BackUpAlgorithm(image, ClearSmall, Mediana, T_MEDIAN_THRESHOLD):
    baseName = image[0].split('.')[0]
    raster_out = baseName + '.BackUpMask'
    B3_toar = selectFromImage(image,'B3_toar')
    Composite = selectFromImage(image,'Composite')
    expression = 'eval(Mask=%(B3)s*%(Clearsmall)s, ' \
                 'Threshold=%(Mediana)s+%(Thresh)s); ' \
                 '%(output)s=(Mask>Threshold)*%(Composite)s;' \
                 % {'B3': B3_toar, 'Clearsmall': ClearSmall, 'Mediana': Mediana, 'Thresh': T_MEDIAN_THRESHOLD,
                    'output': raster_out, 'Composite': Composite}
    image.append(raster_out)
    r.mapcalc(expression=expression, overwrite=True)

# create mask for TMask algorithm:
def TMaskp_mask(image, ClearSmall):
    baseName = image[0].split('.')[0]
    raster_out1 = baseName + '.B3_masked'
    raster_out2 = baseName + '.B5_masked'
    raster_out3 = baseName + '.B6_masked'
    B3_toar = selectFromImage(image,'B3_toar')
    B5_toar = selectFromImage(image, 'B5_toar')
    B6_toar = selectFromImage(image, 'B6_toar')
    BackUpMask = selectFromImage(image,'BackUpMask')
    expression = 'eval(Clear1=not(%(BackUpMask)s), ' \
                 'Clear2=not(%(ClearSmall)s)); ' \
                 '%(out1)s=%(B3)s*(Clear1+Clear2);' \
                 '%(out2)s=%(B5)s*(Clear1+Clear2);' \
                 '%(out3)s=%(B6)s*(Clear1+Clear2)' \
                 % {'BackUpMask': BackUpMask, 'ClearSmall': ClearSmall, 'out1': raster_out1, 'B3': B3_toar,
                    'out2': raster_out2, 'B5': B5_toar, 'out3': raster_out3, 'B6': B6_toar}
    r.mapcalc(expression=expression, overwrite=True)
    image.append(raster_out1)
    image.append(raster_out2)
    image.append(raster_out3)

# classification:
def classify(image, const1, const2, const3, const4, const5, const6, recon_B3, recon_B6):
    baseName = image[0].split('.')[0]
    raster_out = baseName + '.TMask'
    B3_observe = selectFromImage(image,'B3_masked')
    res_B3 = selectFromImage(image,'B3_masked_residual')
    res_B5 = selectFromImage(image,'B5_masked_residual')
    res_B6 = selectFromImage(image,'B6_masked_residual')
    rec_B3 = selectFromImage(image,recon_B3)
    rec_B6 = selectFromImage(image,recon_B6)
    expression = 'eval(T_snow=(%(const1)s-%(recon_B6)s)*(%(observed_B3)s-%(recon_B3)s)/(%(const2)s-%(recon_B3)s), ' \
                 'Step1=(%(resB3)s)>(%(const3)s), ' \
                 'Step2=((%(resB5)s)>(%(const4)s))&&((%(resB6)s)<T_snow), ' \
                 'Step3=((%(resB5)s)<(%(const5)s))&&((%(resB6)s)<(%(const6)s)), ' \
                 'Snow=Step1&&Step2, ' \
                 'Cloud=Step1&&(not(Step2)), ' \
                 'Cloud_shadow=(not(Step1))&&Step3); ' \
                 '%(out)s=Cloud*3 + Snow*2 + Cloud_shadow;' \
                 % {'const1': const1, 'recon_B6': rec_B6, 'observed_B3': B3_observe, 'recon_B3': rec_B3,
                    'const2': const2, 'resB3': res_B3, 'const3': const3, 'resB5': res_B5, 'const4': const4,
                    'resB6': res_B6, 'const5': const5, 'const6': const6, 'out': raster_out}
    r.mapcalc(expression=expression, overwrite=True)
    image.append(raster_out)

# TMask algorithm:
def TMaskAlgorithm(images, BACKUP_ALG_THRESHOLD=4, RADIUS_BUFF=3, T_MEDIAN_THRESHOLD=0.04,
                   BLUE_CHANNEL_PURE_SNOW_THRESHOLD=0.4, NIR_CHANNEL_PURE_SNOW_THRESHOLD=0.12,
                   BLUE_CHANNEL_THRESHOLD=0.04, NIR_CHANNEL_CLOUD_SNOW_THRESHOLD=0.04,
                   NIR_CHANNEL_SHADOW_CLEAR_THRESHOLD=-0.04, SWIR1_CHANNEL_SHADOW_CLEAR_THRESHOLD=-0.04):

    # the size of the collection:
    ImageCounts = len(images)
    text1 = 'Total number of images: %s' % (ImageCounts)
    text2 = 'Warning: You have less than %s images!' % (BACKUP_ALG_THRESHOLD)
    if ImageCounts >= BACKUP_ALG_THRESHOLD:
        logging.info(text1)
    else:
        logging.info(text2)

    # FMask, composite and non-snow masks:
    FMask_collection = [FMask(im, RADIUS_BUFF) for im in images]
    for im in images:
        delete(im,'BQA')

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
    for im in images:
        BackUp_mask(im,ClearSmall)
        delete(im,'nonSnow')

    # calculate mediana for potential clear pixels in BackUp approach:
    r.series(input=selectFromCollection(images,'B3_masked'), output='Mediana.const', method='median', overwrite=True)
    Mediana = 'Mediana.const'
    for im in images:
        delete(im,'B3_masked')

    # BackUp algorithm:
    for im in images:
        BackUpAlgorithm(im, ClearSmall, Mediana, T_MEDIAN_THRESHOLD)
        delete(im,'Composite')
    g.remove(type='raster', name=Mediana, flags='fb')

    # create mask for TMask algorithm:
    for im in images:
        TMaskp_mask(im, ClearSmall)
        delete(im, 'B3_toar')
        delete(im, 'B5_toar')
        delete(im, 'B6_toar')
    g.remove(type='raster', name=ClearSmall, flags='fb')

    # regression for blue, NIR, SWIR channel:
    RobustRegression(images, 'B3_masked', fet=0.5, dod=0.5, order=1, delta=0.5, iterates=2)
    RobustRegression(images, 'B5_masked', fet=0.5, dod=0.5, order=1, delta=0.5, iterates=2)
    RobustRegression(images, 'B6_masked', fet=0.5, dod=0.5, order=1, delta=0.5, iterates=2)

    # getting residuals:
    for im in images:
        getResidual(im, selectFromImage(im, 'B3_masked_lwr_lwr'), 'B3_masked')
        getResidual(im, selectFromImage(im, 'B5_masked_lwr_lwr'), 'B5_masked')
        getResidual(im, selectFromImage(im, 'B6_masked_lwr_lwr'), 'B6_masked')
        delete(im, 'B5_masked')
        delete(im, 'B6_masked')
        delete(im, 'B5_masked_lwr_lwr')

    # classification:
    const1 = NIR_CHANNEL_PURE_SNOW_THRESHOLD
    const2 = BLUE_CHANNEL_PURE_SNOW_THRESHOLD
    const3 = BLUE_CHANNEL_THRESHOLD
    const4 = NIR_CHANNEL_CLOUD_SNOW_THRESHOLD
    const5 = NIR_CHANNEL_SHADOW_CLEAR_THRESHOLD
    const6 = SWIR1_CHANNEL_SHADOW_CLEAR_THRESHOLD
    for im in images:
        classify(im, const1, const2, const3, const4, const5, const6, 'B3_masked_lwr_lwr', 'B6_masked_lwr_lwr')
        delete(im, 'B3_masked_lwr_lwr')
        delete(im, 'B6_masked_lwr_lwr')
        delete(im, 'B3_masked')
        delete(im, 'B3_masked_residual')
        delete(im, 'B5_masked_residual')
        delete(im, 'B6_masked_residual')

    for im in images:
        basename = im[0].split('.')[0]
        out = basename + '.Mask'
        expression = '%(out)s=(%(mask1)s) + (%(mask2)s)' \
                     %{'out':out, 'mask1': selectFromImage(im,'BackUpMask'), 'mask2': selectFromImage(im,'TMask')}
        r.mapcalc(expression=expression, overwrite=True)
        delete(im, 'BackUpMask')
        delete(im, 'TMask')
        im.append(out)

    return images