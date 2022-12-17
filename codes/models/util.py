import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable

def Indexflow(Totalnum, batch_size, random=True):
    numberofchunk = int(Totalnum + batch_size - 1)// int(batch_size)
    totalIndx = np.arange(Totalnum).astype(np.int)
    if random is True:
        totalIndx = np.random.permutation(totalIndx)

    chunkstart = 0
    for chunkidx in range(int(numberofchunk)):
        thisnum = min(batch_size, Totalnum - chunkidx*batch_size)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]
        chunkstart += thisnum
        yield thisInd

def to_variable(x, requires_grad=False, cuda=False, var=True):
    if type(x) is Variable:
        return x
    if type(x) is np.ndarray:
        x = torch.from_numpy(x.astype(np.float32))
    if var:
        x = Variable(x, requires_grad=requires_grad)
    if cuda:
       return x.cuda()
    else:
       return x

def to_device(src, ref, var = True, requires_grad=False):
    src = to_variable(src, requires_grad=requires_grad, var=var)
    return src.cuda(ref.get_device()) if ref.is_cuda else src

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def run_disc_ada(disc_X, real_X, label_X=None, augment_probability=None, augmentpipe=None):
    if (augment_probability is not None) and (augment_probability > 0) and (augmentpipe is not None):
        real_X, _, _ = augmentpipe(real_X)
    return disc_X(real_X)

def run_disc_joint(disc_X, real_X, fake_X, label_X=None, augment_probability=None, augmentpipe=None):
    if (augment_probability is not None) and (augment_probability > 0) and (augmentpipe is not None):
        real_X, label_X, fake_X = augmentpipe(real_X, label_X, fake_X)
    return disc_X(real_X), disc_X(fake_X)

def get_disc_loss_joint(real_X, fake_X, disc_X, adv_criterion, augment_probability=None, augmentpipe=None):
    real_X_pred, fake_X_pred = run_disc_joint(disc_X, real_X, fake_X.detach(), augment_probability=augment_probability, augmentpipe=augmentpipe)
    real_X_disc_loss = adv_criterion(real_X_pred, torch.ones_like(real_X_pred))
    fake_X_disc_loss = adv_criterion(fake_X_pred, torch.zeros_like(fake_X_pred))
    disc_loss = (real_X_disc_loss + fake_X_disc_loss) * 0.5
    return disc_loss

def get_gen_adversarial_loss_ada(real_X, disc_Y, gen_XY, adv_criterion, augment_probability=None, augmentpipe=None):
    fake_Y = gen_XY(real_X)
    fake_Y_disc_pred = run_disc_ada(disc_Y, fake_Y, augment_probability=augment_probability, augmentpipe=augmentpipe)
    adversarial_loss = adv_criterion(fake_Y_disc_pred, torch.ones_like(fake_Y_disc_pred))
    return adversarial_loss, fake_Y

def get_identity_loss(real_X, gen_YX, identity_criterion):
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(identity_X, real_X)
    return identity_loss, identity_X

def get_gen_loss_aug(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion,
    det_net=None, label_A=None, params=None, update_det=False, lambda_identity=0.1,
    lambda_det=1.0, use_crossdomain=False, det_net_A=None, cross_domain_criterion=None, lambda_crodom=1.0,
    augment_probability=None, augmentpipe=None):
    adv_loss_A, fake_A = get_gen_adversarial_loss_ada(real_B, disc_A, gen_BA, adv_criterion,
                        augment_probability=augment_probability, augmentpipe=augmentpipe)
    adv_loss_B, fake_B = get_gen_adversarial_loss_ada(real_A, disc_B, gen_AB, adv_criterion,
                        augment_probability=augment_probability, augmentpipe=augmentpipe)

    identity_loss_A, identity_A = get_identity_loss(real_A, gen_BA, identity_criterion)
    identity_loss_B, identity_B = get_identity_loss(real_B, gen_AB, identity_criterion)

    if update_det:
        det_loss = get_detect_loss(det_net, fake_B, label_A, params)
        if use_crossdomain:
            cross_domain_loss = get_cross_domain_loss(fake_A, real_B, det_net_A, det_net, cross_domain_criterion)
            gen_loss = adv_loss_A + adv_loss_B + lambda_identity * (identity_loss_A + identity_loss_B) \
                + lambda_det * det_loss + lambda_crodom * cross_domain_loss
        else:
            gen_loss = adv_loss_A + adv_loss_B + lambda_identity * (identity_loss_A + identity_loss_B) \
                + lambda_det * det_loss
    else:
        gen_loss = adv_loss_A + adv_loss_B + lambda_identity * (identity_loss_A + identity_loss_B)
    return gen_loss, fake_A, fake_B

def get_weight_mask(y_true, params = None):
    if params is not None:
        y_true = y_true.float() / 255.0 * params['scale']
        mean_label = torch.mean(torch.mean(y_true, dim = -1, keepdim=True), dim=-2, keepdim=True)
        y_mask = params['beta']*y_true + params['alpha']*mean_label
    else:
        y_mask = torch.ones(y_true.size())
    return y_true, y_mask

def mean_squared_error(y_true, y_pred):
    diff = y_pred - y_true
    last_dim = len(y_pred.size()) - 1
    return torch.mean(diff*diff, dim=last_dim)

def weighted_loss(y_true, y_pred, y_mask):
    assert y_pred.dim() == 4, 'dimension is not matched!!'
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    y_true = y_true.permute(0,2,3,1)
    y_pred = y_pred.permute(0,2,3,1)
    naive_loss = mean_squared_error(y_true,y_pred)
    masked =  naive_loss * y_mask
    return torch.mean(masked)

def get_detect_loss(det_net_X, fake_Y, label_X, validation=False, params=None):
    if params is None:
        params = dict()
        params['scale'] = 5.0
        params['alpha'] = 5.0 / params['scale']
        params['beta'] = 1.0 / params['scale']
    label_X_scale, label_X_mask = get_weight_mask(label_X, params)
    if validation:
        pred_fake_Y = det_net_X.predict(fake_Y)
    else:
        pred_fake_Y = det_net_X(fake_Y)
    det_loss = weighted_loss(label_X_scale, pred_fake_Y, label_X_mask)
    return det_loss

def get_cross_domain_loss(fake_X, real_Y, det_net_X, det_net_Y, cross_domain_criterion):
    pred_fake_X = det_net_X(fake_X)
    pred_real_Y = det_net_Y(real_Y)
    cross_domain_loss = cross_domain_criterion(pred_fake_X, pred_real_Y)
    return cross_domain_loss
