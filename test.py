import os
import click
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from codes.data.datasets import SingleDataset
from codes.models import get_model
from codes.models.models import models
from codes.tools.evaluation import eval_folder
from codes.tools.util import get_seed_name
import numpy as np
import scipy.io as sio
import copy
from scipy.ndimage import uniform_filter
from  skimage.feature import peak_local_max

@click.command()
@click.argument('outdir')
@click.option('--train_val_test', required=True, multiple=True)
@click.option('--target', default="")
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--detector', default='micronet', type=click.Choice(models.keys()))
@click.option('--use_tanh/--no_use_tanh', default=False)
@click.option('--model_path', default="", type=click.Path(exists=True))
@click.option('--checkpoint', default="")
@click.option('--gpu', default='0')
@click.option('--quan_eval/--no_quan_eval', default=True)
def main(outdir, train_val_test, target, datadir, detector, use_tanh, model_path, checkpoint, gpu, quan_eval):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    print('Models:', 'detector=' + detector, 'use_tanh = ' + str(use_tanh), sep=' ')
    print('Model path: ', 'model_path=' + model_path, 'checkpoint=' + checkpoint, sep=' ')
    print('Data: ', 'target=' + target)

    dim_A = 3
    dataname = model_path.split('/')[-1]
    learned_model_path = os.path.join(model_path, checkpoint+'.pth')
    weights_dict = torch.load(learned_model_path, map_location=lambda storage, loc: storage)
    det_net = get_model(detector).cuda()
    det_net.load_state_dict(weights_dict['det_net'])

    for data_split in train_val_test:
        dataset = SingleDataset(root=datadir, dataA=target, crop_size=None, split=data_split, mode='test', use_tanh=use_tanh)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        local_max_len = 3
        avg_filter_size = 2 * local_max_len + 1
        pred_threshold_pool = np.arange(0.0, 1.01, 0.05)
        savefolder = os.path.join(outdir, data_split, target, dataname, checkpoint)
        savefolder_det = os.path.join(outdir, data_split, target, dataname, checkpoint, target)
        if not os.path.exists(savefolder_det):
            os.makedirs(savefolder_det)
        resultsDict = dict()
        pred_map_name = 'pred_map'
        for batch_index, (real_A, _, name_A) in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                name_A = name_A[0]
                real_A = real_A.cuda()
                resultDictPath_mat = os.path.join(savefolder_det, name_A + '.mat')
                pred_map = np.squeeze(det_net.predict(real_A).cpu().data.numpy())
                resultsDict[pred_map_name] = np.copy(pred_map)

            for threshhold in pred_threshold_pool:
                pred_map_copy = copy.deepcopy(pred_map)
                pred_map_copy[pred_map_copy < 0] = 0.0
                pred_map_copy = uniform_filter(pred_map_copy, size=avg_filter_size)
                pred_map_copy[pred_map_copy < threshhold*np.max(pred_map_copy[:])] = 0
                localseedname = get_seed_name(threshhold, local_max_len)
                coordinates = peak_local_max(pred_map_copy, min_distance=local_max_len, indices = True)

                if coordinates.size == 0:
                    coordinates = np.asarray([])
                    print("Detect: Empty coordinates for img:{s} for parameter t_{thd:3.2f}_r_{rad:3.2f}".format(s=name_A, thd=threshhold, rad=local_max_len))

                resultsDict[localseedname] = coordinates
            sio.savemat(resultDictPath_mat, resultsDict)

        imgfolder = os.path.join(datadir, target, 'images', data_split)
        print('Quantitative evaluation: ', quan_eval)
        radius_pool = [16]
        gd_name = 'centers'
        if quan_eval:
            modelfolder = checkpoint
            eval_savefolder = savefolder
            eval_folder(imgfolder= imgfolder, resfolder= savefolder_det, savefolder=eval_savefolder,
                                        radius_pool=radius_pool, resultmask = modelfolder, thresh_pool=pred_threshold_pool,
                                        len_pool= [local_max_len], imgExt=['.bmp', '.jpg','.png'], contourname=gd_name)

if __name__ == '__main__':
    main()
