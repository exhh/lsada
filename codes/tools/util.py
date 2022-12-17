import cv2
import os

def get_seed_name(threshhold, min_len):
    name  =('t_'   + '{:01.02f}'.format(threshhold) \
             + '_r_'+  '{:02.02f}'.format(min_len)).replace('.','_')
    return name

def getfilelist(Imagefolder, inputext):
    '''inputext: ['.json'] '''
    if type(inputext) is not list:
        inputext = [inputext]
    filelist = []
    filenames = []
    for f in os.listdir(Imagefolder):
        if os.path.splitext(f)[1] in inputext and os.path.isfile(os.path.join(Imagefolder,f)):
               filelist.append(os.path.join(Imagefolder,f))
               filenames.append(os.path.splitext(os.path.basename(f))[0])
    return filelist, filenames

def imread(imgfile):
    assert os.path.exists(imgfile), '{} does not exist!'.format(imgfile)
    srcBGR = cv2.imread(imgfile)
    destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
    return destRGB
