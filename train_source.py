import os
import click
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from codes.data.datasets import SingleDataset
from codes.models import get_model
from codes.models.models import models
from codes.models.util import get_detect_loss
# torch.manual_seed(1)

@click.command()
@click.argument('outdir')
@click.option('--source', default="")
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--crop_size', default=256)
@click.option('--use_validation/--no_use_validation', default=False)
@click.option('--detector', default='micronet', type=click.Choice(models.keys()))
@click.option('--pretrain_weights', type=click.Path(exists=True))
@click.option('--use_tanh/--no_use_tanh', default=False)
@click.option('--n_epochs', default=100)
@click.option('--batch_size', default=1)
@click.option('--gpu', default='0')
def main(outdir, source, datadir, crop_size, use_validation,
        detector, pretrain_weights, use_tanh, n_epochs, batch_size, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('Models:', 'detector=' + detector, 'use_tanh=' + str(use_tanh), sep=' ')
    print('pretrain_weights: ', pretrain_weights)
    print('parameters: ', 'n_epochs=' + str(n_epochs), 'batch_size=' + str(batch_size), sep=' ')
    print('Data: ', 'source=' + source)
    print('crop_size=' + str(crop_size), 'use_validation=' + str(use_validation), sep=' ')

    dim_A = 3
    lr_det = 1e-3
    dataname = source
    dataset = SingleDataset(root=datadir, dataA=source, crop_size=crop_size,
                split='train', mode='train', use_tanh=use_tanh)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if use_validation:
        valid_dataset = SingleDataset(root=datadir, dataA=source,
                        split='val', mode='validation', use_tanh=use_tanh)
        valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4)

    det_net = get_model(detector).cuda()
    if pretrain_weights is not None:
        pretrain_weights_dict = torch.load(pretrain_weights, map_location=lambda storage, loc: storage)
        det_net.load_state_dict(pretrain_weights_dict)
    det_net_opt = torch.optim.SGD(det_net.parameters(), lr=lr_det, momentum=0.99, nesterov = True, weight_decay=1e-06)

    mean_detector_loss = 0
    step_count_det = 0
    cur_step = 0
    display_step = 500
    n_iterations = 100000
    checkpoints = [x for x in range(10000,n_iterations+1,20000)]

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if use_validation:
        validation_step = 200
        best_score = 1e9
        valid_count = 0
        tolerance_freq = 100

    for epoch in range(n_epochs):
        for real_A, label_A, _ in tqdm(dataloader):
            if cur_step in checkpoints:
                torch.save({
                    'det_net': det_net.state_dict()
                }, f"{outdir}/detector_{cur_step}.pth")

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
                            'det_net': det_net.state_dict()
                        }, f"{outdir}/detector_best.pth")
                        print('Save best weights to: ', '{}/detector_best.pth'.format(outdir))
                        valid_count = 0
                    else:
                        valid_count += 1
                    if valid_count >= tolerance_freq:
                        torch.save({
                            'det_net': det_net.state_dict()
                        }, f"{outdir}/detector_{cur_step}.pth")
                        assert 0, 'performance not imporoved for a long time'

            det_net.train()
            real_A = real_A.cuda()
            label_A = label_A.cuda()

            det_net_opt.zero_grad()
            det_loss = get_detect_loss(det_net, real_A, label_A)
            det_loss.backward()
            det_net_opt.step()

            mean_detector_loss += det_loss.item()
            step_count_det += 1
            cur_step += 1

            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Detector loss: {mean_detector_loss / step_count_det}.")
                mean_detector_loss = 0
                step_count_det = 0

        # Save final-iteraion model, if n_epochs * (data_size / batch_size) < n_iterations
        if epoch == n_epochs - 1:
            torch.save({
                'det_net': det_net.state_dict()
            }, f"{outdir}/detector_{cur_step}.pth")

        # Save final-iteration model
        if cur_step >= n_iterations:
            torch.save({
                'det_net': det_net.state_dict()
            }, f"{outdir}/detector_{cur_step}.pth")
            print('Reached maximum iteration!')
            break

if __name__ == '__main__':
    main()
