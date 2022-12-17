import os
import click
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from codes.data.datasets import DoubleDataset, SingleDataset
from codes.models import get_model
from codes.models.models import models
from codes.models.util import weights_init, get_disc_loss_joint, get_gen_loss_aug, get_detect_loss, get_cross_domain_loss
from codes.models.augment import AugmentPipe
# torch.manual_seed(1)

@click.command()
@click.argument('outdir')
@click.option('--source', default="")
@click.option('--target', default="")
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--crop_size', default=256)
@click.option('--augment_probability',  default=0.6)
@click.option('--use_validation/--no_use_validation', default=False)
@click.option('--generator', default='unetgen', type=click.Choice(models.keys()))
@click.option('--discriminator', default='patchgan', type=click.Choice(models.keys()))
@click.option('--detector', default='micronet', type=click.Choice(models.keys()))
@click.option('--pretrain_weights', default=None)
@click.option('--use_tanh/--no_use_tanh', default=False)
@click.option('--use_crossdomain/--no_use_crossdomain', default=False)
@click.option('--pretrain_weights_a', default=None)
@click.option('--update_weights_a/--no_update_weights_a', default=False)
@click.option('--n_epochs', default=100)
@click.option('--batch_size', default=1)
@click.option('--gpu', default='0')
def main(outdir, source, target, datadir, crop_size, augment_probability, use_validation,
        generator, discriminator, detector, pretrain_weights, use_tanh, use_crossdomain, pretrain_weights_a, update_weights_a,
        n_epochs, batch_size, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    pretrain_weights_A = pretrain_weights_a
    update_weights_A = update_weights_a
    print('Models:', 'generator=' + generator, 'discriminator=' + discriminator, 'detector=' + detector, 'use_tanh = ' + str(use_tanh), sep=' ')
    print('parameters: ', 'n_epochs=' + str(n_epochs), 'batch_size=' + str(batch_size), sep=' ')
    print('Data: ', 'source=' + source, 'target=' + target, sep=' ')
    print('crop_size=' + str(crop_size), 'augment_probability=' + str(augment_probability), 'use_validation = ' + str(use_validation), sep=' ')
    print('use_crossdomain='+str(use_crossdomain), 'pretrain_weights_A='+str(pretrain_weights_A), 'update_weights_A='+str(update_weights_A),
        'pretrain_weights='+str(pretrain_weights), sep=' ')

    dim_A = 3
    dim_B = 3
    lr = 0.0002
    lr_det = 1e-4

    dataname = source + '2' + target
    dataset = DoubleDataset(root=datadir, dataA=source, dataB=target, crop_size=crop_size,
                split='train', mode='train', use_tanh=use_tanh)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if use_validation:
        valid_dataset = SingleDataset(root=datadir, dataA=target,
                    split='val', mode='validation', use_tanh=use_tanh)
        valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4)

    adv_criterion = nn.MSELoss()
    recon_criterion = nn.L1Loss()
    gen_AB = get_model(generator, input_channels=dim_A, output_channels=dim_B, use_tanh=use_tanh).cuda()
    gen_BA = get_model(generator, input_channels=dim_B, output_channels=dim_A, use_tanh=use_tanh).cuda()
    gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=lr, betas=(0.5, 0.999))
    disc_A = get_model(discriminator, input_channels=dim_A).cuda()
    disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_B = get_model(discriminator, input_channels=dim_B).cuda()
    disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=lr, betas=(0.5, 0.999))
    gen_AB = gen_AB.apply(weights_init)
    gen_BA = gen_BA.apply(weights_init)
    disc_A = disc_A.apply(weights_init)
    disc_B = disc_B.apply(weights_init)

    det_net = get_model(detector).cuda()
    if pretrain_weights is not None:
        pretrain_weights_dict = torch.load(pretrain_weights, map_location=lambda storage, loc: storage)
        det_net.load_state_dict(pretrain_weights_dict['det_net'])
    det_net_opt = torch.optim.SGD(det_net.parameters(), lr=lr_det, momentum=0.99, nesterov = True, weight_decay=1e-06)

    if use_crossdomain:
        det_net_A = get_model(detector).cuda()
        if pretrain_weights_A is not None:
            pretrain_weights_A_dict = torch.load(pretrain_weights_A, map_location=lambda storage, loc: storage)
            det_net_A.load_state_dict(pretrain_weights_A_dict['det_net'])
        det_net_A_opt = torch.optim.SGD(det_net_A.parameters(), lr=lr_det, momentum=0.99, nesterov = True, weight_decay=1e-06)
        cross_domain_criterion = nn.L1Loss()

    mean_generator_loss = 0
    mean_discriminatorA_loss = 0
    mean_discriminatorB_loss = 0
    mean_detector_loss = 0
    mean_detectorA_loss = 0
    cur_step = 0
    display_step = 500
    n_iterations = 100000
    checkpoints = [x for x in range(10000,n_iterations+1,20000)]
    step_count_det = 0
    lambda_crodom = 0.0001
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if use_validation:
        validation_step = 200
        best_score = 1e9
        valid_count = 0
        tolerance_freq = 100

    augmentpipe = None
    if (augment_probability is not None) and (augment_probability > 0):
        augmentpipe = AugmentPipe(xflip=augment_probability, rotate90=augment_probability, xint=augment_probability,
                    scale=augment_probability, rotate=augment_probability, aniso=augment_probability, xfrac=augment_probability,
                    brightness=augment_probability, contrast=augment_probability,
                    lumaflip=augment_probability, hue=augment_probability, saturation=augment_probability)
    print(augmentpipe)
    for epoch in range(n_epochs):
        for real_A, label_A, _, real_B, label_B, _ in tqdm(dataloader):
            if cur_step in checkpoints:
                torch.save({
                    'gen_AB': gen_AB.state_dict(),
                    'gen_BA': gen_BA.state_dict(),
                    'disc_A': disc_A.state_dict(),
                    'disc_B': disc_B.state_dict(),
                    'det_net_A': det_net_A.state_dict(),
                    'det_net': det_net.state_dict(),
                }, f"{outdir}/GANdet_{cur_step}.pth")

            with torch.no_grad():
                if use_validation and cur_step % validation_step == 0:
                    running_valid_loss = 0.0
                    running_valid_count = 0
                    for valid_im, valid_label, _ in tqdm(valid_dataloader):
                        valid_im = valid_im.cuda()
                        valid_label = valid_label.cuda()
                        det_valid_loss_batch = get_detect_loss(det_net, valid_im, valid_label, validation=True)
                        running_valid_loss += det_valid_loss_batch.item()
                        running_valid_count += 1
                    det_valid_loss = running_valid_loss / running_valid_count
                    print('\nValidation loss: {}, best_score: {}'.format(det_valid_loss, best_score))
                    if det_valid_loss <= best_score:
                        best_score = det_valid_loss
                        print('update to new best_score: {}'.format(best_score))
                        torch.save({
                            'gen_AB': gen_AB.state_dict(),
                            'gen_BA': gen_BA.state_dict(),
                            'disc_A': disc_A.state_dict(),
                            'disc_B': disc_B.state_dict(),
                            'det_net_A': det_net_A.state_dict(),
                            'det_net': det_net.state_dict(),
                        }, f"{outdir}/GANdet_best.pth")
                        print('Save best weights to: ', '{}/GANdet_best.pth'.format(outdir))
                        valid_count = 0
                    else:
                        valid_count += 1
                    if valid_count >= tolerance_freq:
                        torch.save({
                            'gen_AB': gen_AB.state_dict(),
                            'gen_BA': gen_BA.state_dict(),
                            'disc_A': disc_A.state_dict(),
                            'disc_B': disc_B.state_dict(),
                            'det_net_A': det_net_A.state_dict(),
                            'det_net': det_net.state_dict(),
                        }, f"{outdir}/GANdet_{cur_step}.pth")
                        assert 0, 'performance not imporoved for a long time'

            det_net.train()
            real_A = real_A.cuda()
            real_B = real_B.cuda()
            label_A = label_A.cuda()
            label_B = label_B.cuda()

            disc_A_opt.zero_grad()
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            disc_A_loss = get_disc_loss_joint(real_A, fake_A, disc_A, adv_criterion, augment_probability=augment_probability, augmentpipe=augmentpipe)# joint augment discriminator
            disc_A_loss.backward(retain_graph=True)
            disc_A_opt.step()

            disc_B_opt.zero_grad()
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_disc_loss_joint(real_B, fake_B, disc_B, adv_criterion, augment_probability=augment_probability, augmentpipe=augmentpipe)
            disc_B_loss.backward(retain_graph=True)
            disc_B_opt.step()

            if use_crossdomain:
                if update_weights_A:
                    det_net_A_opt.zero_grad()
                    det_A_loss = get_detect_loss(det_net_A, real_A, label_A) \
                        + lambda_crodom * get_cross_domain_loss(fake_A.detach(), real_B, det_net_A, det_net, cross_domain_criterion)
                    det_A_loss.backward(retain_graph=True)
                    det_net_A_opt.step()
                else:
                    det_A_loss = None

                det_net_opt.zero_grad()
                det_loss = get_detect_loss(det_net, fake_B.detach(), label_A) \
                    + lambda_crodom * get_cross_domain_loss(fake_A.detach(), real_B, det_net_A, det_net, cross_domain_criterion)
                det_loss.backward(retain_graph=True)
                det_net_opt.step()
            else:
                det_net_opt.zero_grad()
                det_loss = get_detect_loss(det_net, fake_B.detach(), label_A)
                det_loss.backward(retain_graph=True)
                det_net_opt.step()
                det_A_loss = None

            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss_aug(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion, det_net, label_A,
                update_det=True, use_crossdomain=use_crossdomain, det_net_A=det_net_A,
                cross_domain_criterion=cross_domain_criterion, lambda_crodom=lambda_crodom,
                augment_probability = augment_probability, augmentpipe=augmentpipe
            )
            gen_loss.backward()
            gen_opt.step()

            mean_discriminatorA_loss += disc_A_loss.item() / display_step
            mean_discriminatorB_loss += disc_B_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            if det_loss is not None:
                mean_detector_loss += det_loss.item()
                step_count_det += 1
                if det_A_loss is not None:
                    mean_detectorA_loss += det_A_loss.item()

            cur_step += 1

            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, DiscA loss: {mean_discriminatorA_loss}, DiscB loss: {mean_discriminatorB_loss}.")
                mean_generator_loss = 0
                mean_discriminatorA_loss = 0
                mean_discriminatorB_loss = 0

                if det_loss is not None:
                    if det_A_loss is None:
                        print(f"Epoch {epoch}: Step {cur_step}: Detector loss: {mean_detector_loss / step_count_det}.")
                    else:
                        print(f"Epoch {epoch}: Step {cur_step}: Detector loss: {mean_detector_loss / step_count_det}: DetectorA loss: {mean_detectorA_loss / step_count_det}.")
                        mean_detectorA_loss = 0
                    mean_detector_loss = 0
                    step_count_det = 0

        # Save final-epoch model, if n_epochs * (data_size / batch_size) < n_iterations
        if epoch == n_epochs - 1:
            torch.save({
                'gen_AB': gen_AB.state_dict(),
                'gen_BA': gen_BA.state_dict(),
                'disc_A': disc_A.state_dict(),
                'disc_B': disc_B.state_dict(),
                'det_net_A': det_net_A.state_dict(),
                'det_net': det_net.state_dict(),
            }, f"{outdir}/GANdet_{cur_step}.pth")

        # Save final-iteration model
        if cur_step >= n_iterations:
            torch.save({
                'gen_AB': gen_AB.state_dict(),
                'gen_BA': gen_BA.state_dict(),
                'disc_A': disc_A.state_dict(),
                'disc_B': disc_B.state_dict(),
                'det_net_A': det_net_A.state_dict(),
                'det_net': det_net.state_dict(),
            }, f"{outdir}/GANdet_{cur_step}.pth")
            print('Reached maximum iteration!')
            break

if __name__ == '__main__':
    main()
