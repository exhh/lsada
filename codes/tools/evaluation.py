import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import pairwise_distances
from .util import get_seed_name, getfilelist, imread
import deepdish as dd
from scipy.io import loadmat
import json
from numba import jit, autojit
import matplotlib.pyplot as plt
import deepdish as dd
import numpy as np
from numpy.core.umath_tests import inner1d
import pdb
# A = np.array([[1,2],[3,4],[5,6],[7,8]])
# B = np.array([[2,3],[4,5],[6,7],[8,9],[10,11]])

def eval_folder(imgfolder = '.', resfolder = '.', savefolder = '.', radius_pool = [16], resultmask = '',
                thresh_pool = [0.50], len_pool = [5],imgExt =['.tif'], contourname = 'centers',
                matExt=["", '_gt', '_label']):
    #resultmask is just added for legency compatability
    print('Currently evaluating: ', imgfolder)
    img_list, img_pool = getfilelist(imgfolder, imgExt)

    num_img, num_thresh, num_len = len(img_pool), len(thresh_pool), len(len_pool)

    TP_res = np.zeros((num_img, num_thresh, num_len))
    FP_res = np.zeros((num_img, num_thresh, num_len))
    FN_res = np.zeros((num_img, num_thresh, num_len))
    pre_res = np.zeros((num_img, num_thresh, num_len))
    rec_res = np.zeros((num_img, num_thresh, num_len))
    f1_res  = np.zeros((num_img, num_thresh, num_len))
    diff_res = np.zeros((num_img, num_thresh, num_len))

    #distance_res = [[None]*num_len]*num_thresh

    res_json = {}
    save_path = os.path.join(savefolder, resultmask + '_res.json')
    for radius in radius_pool:
        for th_idx, thresh in enumerate(thresh_pool):
            for len_idx, maxlen in enumerate(len_pool):
                distance_tmp = []
                distance_avgimage_tmp = np.zeros(num_img)
                for idx, imgname in enumerate(img_pool):
                    #if resultmask is not  "":
                    #    resultDictPath = os.path.join(resfolder, imgname + resultmask + '.h5')
                    #else:
                    # resultDictPath = os.path.join(resfolder, imgname + '.h5')
                    resultDictPath = os.path.join(resfolder, imgname + '.mat')
                    for thismatext in matExt:
                        matfolder = imgfolder.split('/')
                        matfolder =  os.path.join(matfolder[0], matfolder[1], 'mats', matfolder[3])
                        gtPath = os.path.join(matfolder, imgname + thismatext + '.mat')
                        #print(matfolder, imgname, gtPath)
                        if os.path.exists(gtPath):
                           loaded_mt = loadmat(gtPath)
                           break


                    # maybe should the centers or Centers variable to get the groudn truth, because centers are provided by original authors
                    if contourname == 'Contours':
                        contour_mat = loaded_mt[contourname].tolist()[0]
                        seeds_list = [np.mean(a,1) for a in contour_mat]
                        gt_xy = np.stack(seeds_list, 0) # (nsamples * 2)
                        gt = np.zeros_like(gt_xy)
                        gt[:,0] = gt_xy[:,1]
                        gt[:, 1] = gt_xy[:, 0]
                    elif contourname == 'centers' or contourname == 'Centers':
                        gt_xy = loaded_mt[contourname] # 2 x N numpy array
                        gt_xy = np.transpose(gt_xy, (1,0))
                        gt = np.zeros_like(gt_xy)
                        gt[:,0] = gt_xy[:,1]
                        gt[:,1] = gt_xy[:, 0]

                    resultDict = loadmat(resultDictPath)

                    key_name = get_seed_name(thresh, maxlen)
                    thisresult = resultDict[key_name]

                    imgpath = img_list[idx]
                    #thisImg = imread(imgpath)
                    #imgrow, imgcol = thisImg.shape[0:2]
                    if thisresult.shape[0] == 0:
                        print('You have zero detected seeds.')
                        thisresult = np.array([[-1000,-1000]])
                    if gt.shape[0] == 0:
                        print('You have zero ground truth seeds.')
                        gt = np.array([[-2000,-2000]])

                    # (pre, rec, f1), distance, difference, (valid_res, valid_gt) = \
                    #                 detect_eval(thisresult, gt, radius)
                    performdict, (valid_res, valid_gt) = detect_eval(thisresult, gt, radius)
                    # res_true = thisresult[valid_res]

                    #plt.figure('showres')
                    #plt.imshow(img)
                    #plt.plot(res_true[:, 1], res_true[:, 0], 'r.')
                    #plt.show()

                    TP_res[idx, th_idx, len_idx] = performdict['detect']['TP']
                    FP_res[idx, th_idx, len_idx] = performdict['detect']['FP']
                    FN_res[idx, th_idx, len_idx] = performdict['detect']['FN']
                    pre_res[idx, th_idx, len_idx] = performdict['detect']['precision']
                    rec_res[idx, th_idx, len_idx] = performdict['detect']['recall']
                    f1_res[idx, th_idx, len_idx] = performdict['detect']['f1']
                    diff_res[idx, th_idx, len_idx] = performdict['detect']['diff']
                    distance_tmp.append(performdict['detect']['distance'])
                    distance_avgimage_tmp[idx] = np.mean(performdict['detect']['distance'])

                detect_dict, detect_avgimage_dict = {}, {}
                detect_dict['TP'] = np.sum(TP_res[:,th_idx, len_idx])
                detect_dict['FP'] = np.sum(FP_res[:,th_idx, len_idx])
                detect_dict['FN'] = np.sum(FN_res[:,th_idx, len_idx])
                detect_dict['precision'] = float(np.sum(TP_res[:,th_idx, len_idx]))/(np.sum(TP_res[:,th_idx, len_idx]) + np.sum(FP_res[:,th_idx, len_idx]) + 1e-10)
                detect_dict['recall'] = float(np.sum(TP_res[:,th_idx, len_idx]))/(np.sum(TP_res[:,th_idx, len_idx]) + np.sum(FN_res[:,th_idx, len_idx]) + 1e-10)
                detect_dict['f1'] = (2.0*detect_dict['precision']*detect_dict['recall'])/(detect_dict['precision']+detect_dict['recall'] + 1e-10)
                distance_cat = np.concatenate(distance_tmp)
                #distance_res[th_idx][len_idx] = distance_cat
                detect_dict['distance'] = np.mean(distance_cat) # distance average
                detect_dict['distance_std']  = np.std(distance_cat)

                # detect_avgimage_dict['precision_mean'] = np.mean(pre_res[:,th_idx, len_idx])
                # detect_avgimage_dict['precision_std']  = np.std(pre_res[:,th_idx, len_idx])
                # detect_avgimage_dict['recall_mean'] = np.mean(rec_res[:,th_idx, len_idx])
                # detect_avgimage_dict['recall_std']  = np.std(rec_res[:,th_idx, len_idx])
                # detect_avgimage_dict['f1_mean'] = np.mean(f1_res[:,th_idx, len_idx])
                # detect_avgimage_dict['f1_std']  = np.std(f1_res[:,th_idx, len_idx])
                # detect_avgimage_dict['num_difference_mean'] = np.mean(diff_res[:, th_idx, len_idx])
                # detect_avgimage_dict['num_difference_std'] = np.std(diff_res[:, th_idx, len_idx])
                # detect_avgimage_dict['distance_mean'] = np.mean(distance_avgimage_tmp)
                # detect_avgimage_dict['distance_std'] = np.std(distance_avgimage_tmp)


                # t=threshold, r=radius, gtr=ground-truth radius (used to define ground-truth region)
                marker = 't_' + '{:01.02f}'.format(thresh) + '_r_' + '{:02d}'.format(maxlen) + '_gtr_'+  '{:02.02f}'.format(radius)
                res_json[marker] = {
                    "detect_dict":detect_dict
                    # "detect_avgimage_dict":detect_avgimage_dict
                }

    with open(save_path, 'w') as outfile:
        json.dump(res_json, outfile)

def eval_folder_rho(imgfolder = '.', resfolder = '.', savefolder = '.', radius_pool = [16,8], resultmask = '',
                rho_pool = [0.5], thresh_pool = [0.50], len_pool = [5],imgExt =['.tif'], contourname = 'Contours',
                matExt=["",'_withcontour', '_gt','_seg']):
    #resultmask is just added for legency compatability
    print('Currently evaluating: ', imgfolder)
    img_list, img_pool = getfilelist(imgfolder, imgExt)

    num_img, num_thresh, num_len = len(img_pool), len(thresh_pool), len(len_pool)

    TP_res = np.zeros((num_img, num_thresh, num_len))
    FP_res = np.zeros((num_img, num_thresh, num_len))
    FN_res = np.zeros((num_img, num_thresh, num_len))
    pre_res = np.zeros((num_img, num_thresh, num_len))
    rec_res = np.zeros((num_img, num_thresh, num_len))
    f1_res  = np.zeros((num_img, num_thresh, num_len))
    diff_res = np.zeros((num_img, num_thresh, num_len))

    #distance_res = [[None]*num_len]*num_thresh

    res_json = {}
    save_path = os.path.join(savefolder, resultmask + '_res.json')
    for radius in radius_pool:
        for rho in rho_pool:
            print('Evaluate rho = ', rho)
            for th_idx, thresh in enumerate(thresh_pool):
                for len_idx, maxlen in enumerate(len_pool):
                    distance_tmp = []
                    distance_avgimage_tmp = np.zeros(num_img)
                    for idx, imgname in enumerate(img_pool):
                        #if resultmask is not  "":
                        #    resultDictPath = os.path.join(resfolder, imgname + resultmask + '.h5')
                        #else:
                        # resultDictPath = os.path.join(resfolder, imgname + '.h5')
                        resultDictPath = os.path.join(resfolder, imgname + '.mat')
                        for thismatext in matExt:
                            matfolder = imgfolder.split('/')
                            matfolder =  os.path.join(matfolder[0], matfolder[1], 'mats', matfolder[3])
                            gtPath = os.path.join(matfolder, imgname + thismatext + '.mat')
                            #print(matfolder, imgname, gtPath)
                            if os.path.exists(gtPath):
                               loaded_mt = loadmat(gtPath)
                               break


                        # maybe should the centers or Centers variable to get the groudn truth, because centers are provided by original authors
                        if contourname == 'Contours':
                            contour_mat = loaded_mt[contourname].tolist()[0]
                            seeds_list = [np.mean(a,1) for a in contour_mat]
                            gt_xy = np.stack(seeds_list, 0) # (nsamples * 2)
                            gt = np.zeros_like(gt_xy)
                            gt[:,0] = gt_xy[:,1]
                            gt[:, 1] = gt_xy[:, 0]
                        elif contourname == 'centers' or contourname == 'Centers':
                            gt_xy = loaded_mt[contourname] # 2 x N numpy array
                            gt_xy = np.transpose(gt_xy, (1,0))
                            gt = np.zeros_like(gt_xy)
                            gt[:,0] = gt_xy[:,1]
                            gt[:,1] = gt_xy[:, 0]

                        resultDict = loadmat(resultDictPath)
                        key_name = get_seed_name(thresh, maxlen) + '_p_' + '{:01.02f}'.format(rho).replace('.','_')
                        thisresult = resultDict[key_name]

                        imgpath = img_list[idx]
                        if thisresult.shape[0] == 0:
                            print('You have zero detected seeds.')
                            thisresult = np.array([[-10000,-10000]])
                        if gt.shape[0] == 0:
                            print('You have zero ground truth seeds.')
                            gt = np.array([[10000,10000]])

                        # (pre, rec, f1), distance, difference, (valid_res, valid_gt) = \
                        #                 detect_eval(thisresult, gt, radius)
                        performdict, (valid_res, valid_gt) = detect_eval(thisresult, gt, radius)
                        # res_true = thisresult[valid_res]

                        #plt.figure('showres')
                        #plt.imshow(img)
                        #plt.plot(res_true[:, 1], res_true[:, 0], 'r.')
                        #plt.show()

                        TP_res[idx, th_idx, len_idx] = performdict['detect']['TP']
                        FP_res[idx, th_idx, len_idx] = performdict['detect']['FP']
                        FN_res[idx, th_idx, len_idx] = performdict['detect']['FN']
                        pre_res[idx, th_idx, len_idx] = performdict['detect']['precision']
                        rec_res[idx, th_idx, len_idx] = performdict['detect']['recall']
                        f1_res[idx, th_idx, len_idx] = performdict['detect']['f1']
                        diff_res[idx, th_idx, len_idx] = performdict['detect']['diff']
                        distance_tmp.append(performdict['detect']['distance'])
                        distance_avgimage_tmp[idx] = np.mean(performdict['detect']['distance'])

                    detect_dict, detect_avgimage_dict = {}, {}
                    detect_dict['TP'] = np.sum(TP_res[:,th_idx, len_idx])
                    detect_dict['FP'] = np.sum(FP_res[:,th_idx, len_idx])
                    detect_dict['FN'] = np.sum(FN_res[:,th_idx, len_idx])
                    detect_dict['precision'] = float(np.sum(TP_res[:,th_idx, len_idx]))/(np.sum(TP_res[:,th_idx, len_idx]) + np.sum(FP_res[:,th_idx, len_idx]) + 1e-10)
                    detect_dict['recall'] = float(np.sum(TP_res[:,th_idx, len_idx]))/(np.sum(TP_res[:,th_idx, len_idx]) + np.sum(FN_res[:,th_idx, len_idx]) + 1e-10)
                    detect_dict['f1'] = (2.0*detect_dict['precision']*detect_dict['recall'])/(detect_dict['precision']+detect_dict['recall'] + 1e-10)
                    distance_cat = np.concatenate(distance_tmp)
                    #distance_res[th_idx][len_idx] = distance_cat
                    detect_dict['distance'] = np.mean(distance_cat) # distance average
                    detect_dict['distance_std']  = np.std(distance_cat)

                    detect_avgimage_dict['precision_mean'] = np.mean(pre_res[:,th_idx, len_idx])
                    detect_avgimage_dict['precision_std']  = np.std(pre_res[:,th_idx, len_idx])
                    detect_avgimage_dict['recall_mean'] = np.mean(rec_res[:,th_idx, len_idx])
                    detect_avgimage_dict['recall_std']  = np.std(rec_res[:,th_idx, len_idx])
                    detect_avgimage_dict['f1_mean'] = np.mean(f1_res[:,th_idx, len_idx])
                    detect_avgimage_dict['f1_std']  = np.std(f1_res[:,th_idx, len_idx])
                    detect_avgimage_dict['num_difference_mean'] = np.mean(diff_res[:, th_idx, len_idx])
                    detect_avgimage_dict['num_difference_std'] = np.std(diff_res[:, th_idx, len_idx])
                    detect_avgimage_dict['distance_mean'] = np.mean(distance_avgimage_tmp)
                    detect_avgimage_dict['distance_std'] = np.std(distance_avgimage_tmp)


                    # t=threshold, r=radius, gtr=ground-truth radius (used to define ground-truth region)
                    marker ='t_' + '{:01.02f}'.format(thresh) + '_r_' + '{:02d}'.format(maxlen) + '_p_' + '{:01.02f}'.format(rho) + '_gtr_'+  '{:02.02f}'.format(radius)
                    res_json[marker] = {
                        "detect_dict":detect_dict,
                        "detect_avgimage_dict":detect_avgimage_dict
                    }

    with open(save_path, 'w') as outfile:
        json.dump(res_json, outfile)

def detect_eval(res, gt, radius):
    '''
    res: N*2 tensor for (row, col) seeds.
    gt:  N*2 tensor for (row, col) gt seeds.
    '''

    num_det = res.shape[0]
    num_gt  = gt.shape[0]
    valid_row, valid_col = graph_match(res, gt, radius)

    TP = len(valid_row)
    FP = num_det - len(valid_row)
    FN = num_gt  - len(valid_row)

    pre = float(TP)/(TP + FP + 1e-10)
    rec = float(TP)/(TP + FN + 1e-10)
    f1 = (2.0*pre*rec)/(pre+rec + 1e-10)
    #spec = float(TN)/(FP + TN + 1e-10)

    matched_res = res[valid_row]
    matched_gt  = gt[valid_col]

    difference = abs(num_det - num_gt)
    distance = np.sqrt( np.sum(  (matched_gt - matched_res)**2, axis=1) )

    performdict = {}
    # overall performance for one image
    performdict['detect'] = {}
    performdict['detect']['TP'] = TP
    performdict['detect']['FP'] = FP
    performdict['detect']['FN'] = FN
    performdict['detect']['precision'] = pre
    performdict['detect']['recall'] = rec
    performdict['detect']['f1'] = f1
    performdict['detect']['diff'] = difference
    performdict['detect']['distance'] = distance
    return performdict, (valid_row, valid_col)

def graph_match(res, gt, radius):
    '''
    Parameters
    ----------
    res: N*2
    gt:  N*2
    '''
    num_det = res.shape[0]
    num_gt  = gt.shape[0]

    distmatrix = pairwise_distances(res, gt,metric = 'euclidean')
    #distmatrix[distmatrix > radius] = np.Inf
    #paddist = np.zeros((distmatrix.shape[0] + 1,distmatrix.shape[1] + 1))
    #paddist[0:num_det, 0:num_gt] = distmatrix
    #paddist[num_det,:] = 1e8
    #paddist[:,num_gt] = 1e8
    #the true positive is the one that does not choose boaders.
    row_ind, col_ind = linear_sum_assignment(distmatrix)

    valid_mask = distmatrix[row_ind, col_ind] <= radius
    #valid_mask = col_ind != num_gt and row_ind != num_det
    valid_col = col_ind[valid_mask]
    valid_row = row_ind[valid_mask]
    return valid_row, valid_col

@autojit
def seg_label2contour(label_img, org=None, print_color = [0,0,1], linewidth = 2, alpha = 1, returnImg =True):
    row, col = label_img.shape
    contour_img = np.zeros(label_img.shape, dtype=bool)
    regions = regionprops(label_img)
    contourlist = [np.array([-1,-1])]*len(regions) #because numba can not work with []
    infolist = [((1,1,1,1),1)]*len(regions) #because numba can not work with []

    for id, props in enumerate(regions):
        minr, minc, maxr, maxc = props.bbox
        rs, re = max(minr-1,0),min(maxr+1, row)
        cs, ce = max(minc-1,0), min(maxc+1, col)
        thispatch = label_img[rs:re, cs:ce] == props.label
        contours  = measure.find_contours(thispatch, 0)
        thiscontour = (contours[0] + [rs, cs]).astype(int)

        contourlist[id] = safe_boarder(thiscontour, row, col)
        infolist[id] = (( rs, re, cs, ce ), props.label )
    return contourlist, infolist


def get_segmentation(probmap=None, coordinates=None, thresh_seg= 0.5,tiny_size=100):
    seg_prob = probmap.copy()
    label_img = overal_watershed(seg_prob, thresh_water = 0.5, thresh_seg = thresh_seg,
                                    ratio = 0.3, dist_f = np.median, tiny_size=tiny_size)
    class_label, new_label, new_coord = residual_markers(label_img, coordinates)
    marker_map = np.zeros_like(seg_prob).astype(np.uint8)
    marker_map[new_coord[:, 0], new_coord[:, 1]] = 1
    residual_label = overal_watershed_marker(seg_prob*(new_label>0), marker_map, thresh_seg= thresh_seg, ratio = 0.05, dist_f = np.median, tiny_size=tiny_size)
    final_label = class_label + (residual_label + np.max(class_label))*( residual_label !=0 )

    contours, infolist = seg_label2contour(final_label, returnImg=False)
    return  contours, infolist, final_label

def get_seed(VotingMap, thresh, min_len):
    VotingMap[VotingMap < thresh * np.max(VotingMap[:])] = 0

    coordinates = peak_local_max(VotingMap, min_distance=min_len, indices=True)  # N by 2,
    if coordinates.size == 0:
        coordinates = np.asarray([])
    return coordinates

def seg_reward(res_seed_map, gt_seed_map, res_mask, gt_mask, det_thresh,min_len, radius):
    '''
    res_contours:
            a list of N*2 points
    gt_contours:
            a list of N*2 points.
    '''
    gt_seed = get_seed(gt_seed_map, det_thresh, min_len)
    res_seed =  get_seed(res_seed_map, det_thresh, min_len)

    res_contorus, res_infolist, res_label = get_segmentation(res_mask, res_seed, thresh_seg, tiny_size)
    gt_contorus, gt_infolist, gt_label    = get_segmentation(gt_mask, gt_seed, thresh_seg, tiny_size)

    res_seeds = [np.mean(x, 0) for x in res_contours]
    gt_seeds =  [np.mean(x, 0) for x in gt_contours]
    res_seeds = np.array(res_seeds)
    gt_seeds = np.array(gt_seeds)

    matched_res, matched_gt = graph_match(res_seeds, gt_seeds, radius = radius)
    hsdist = []
    dsc_list = []
    for id, (res_id, gt_id) in zip(matched_res, matched_gt):
        this_res = res_contours[res_id]
        this_gt  = gt_contours[gt_id]
        res_info = res_infolist[res_id]
        gt_info = gt_infolist[gt_id]

        (res_rs, res_re, res_cs, res_ce), res_indica = res_info
        (gt_rs, gt_re, gt_cs, gt_ce), gt_indica = gt_info

        rs, re, cs, ce = min(res_rs, gt_rs), max(res_re, gt_re), min(res_cs, gt_cs), max(res_ce, gt_ce)
        this_res_patch = res_label[rs:re, cs:ce]==res_indica
        this_gt_patch = gt_label[rs:re, cs:ce]==gt_indica

        this_dsc = 2.0*np.sum(np.logical_and(this_res_patch, this_gt_patch))\
                   /(np.sum(this_res_patch) + np.sum(this_gt_patch))

        hsdist.append(ModHausdorffDist(this_res,this_gt))
        dsc_list.append(this_dsc)

    return -mean(hsdist), mean(dsc_list)



# Hausdorff Distance
def HausdorffDist(A,B):
    """
    Created on Mon Jun 16 23:56:41 2014
    @author: Edward
    """
    # Hausdorf Distance: Compute the Hausdorff distance between two point
    # clouds.
    # Let A and B be subsets of metric space (Z,dZ),
    # The Hausdorff distance between A and B, denoted by dH(A,B),
    # is defined by:
    # dH(A,B) = max(h(A,B),h(B,A)),
    # where h(A,B) = max(min(d(a,b))
    # and d(a,b) is a L2 norm
    # dist_H = hausdorff(A,B)
    # A: First point sets (MxN, with M observations in N dimension)
    # B: Second point sets (MxN, with M observations in N dimension)
    # ** A and B may have different number of rows, but must have the same
    # number of columns.
    #
    # Edward DongBo Cui; Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
    return(dH)

def ModHausdorffDist(A,B):
    """
    Created on Mon Jun 16 23:56:41 2014
    @author: Edward
    """
    #This function computes the Modified Hausdorff Distance (MHD) which is
    #proven to function better than the directed HD as per Dubuisson et al.
    #in the following work:
    #
    #M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
    #matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
    #http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
    #
    #The function computed the forward and reverse distances and outputs the
    #maximum/minimum of both.
    #Optionally, the function can return forward and reverse distance.
    #
    #Format for calling function:
    #
    #[MHD,FHD,RHD] = ModHausdorffDist(A,B);
    #
    #where
    #MHD = Modified Hausdorff Distance.
    #FHD = Forward Hausdorff Distance: minimum distance from all points of B
    #      to a point in A, averaged for all A
    #RHD = Reverse Hausdorff Distance: minimum distance from all points of A
    #      to a point in B, averaged for all B
    #A -> Point set 1, [row as observations, and col as dimensions]
    #B -> Point set 2, [row as observations, and col as dimensions]
    #
    #No. of samples of each point set may be different but the dimension of
    #the points must be the same.
    #
    #Edward DongBo Cui Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Calculating the forward HD: mean(min(each col))
    FHD = np.mean(np.min(D_mat,axis=1))
    # Calculating the reverse HD: mean(min(each row))
    RHD = np.mean(np.min(D_mat,axis=0))
    # Calculating mhd
    MHD = np.max(np.array([FHD, RHD]))
    return(MHD, FHD, RHD)
