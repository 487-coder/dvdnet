import argparse, time
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from new_model import DVDnetFull
from dataloader import ValDataset, VideoSequenceDataset
from utils import orthogonal_conv_weights, close_logger, init_logging, normalize_augment
from new_train_common import resume_training, lr_scheduler, log_train_psnr, \
    validate_and_log, save_model_checkpoint


def main(**kwargs):
    # Load dataset
    dataset_val = ValDataset(data_dir=kwargs['valset_dir'], gray_mode=False)

    dataset_train = VideoSequenceDataset(kwargs['trainset_dir'], kwargs['temp_patch_size'], kwargs['patch_size'],
                                         kwargs['max_number_patches'],
                                         random_shuffle=True,
                                         temp_stride=3
                                         )

    loader_train = DataLoader(dataset_train, batch_size=kwargs['batch_size'], shuffle=False, num_workers=1,
                              pin_memory=True)

    num_minibatches = int(kwargs['max_number_patches'] // kwargs['batch_size'])
    ctrl_fr_idx = (kwargs['temp_patch_size'] - 1) // 2
    print("\t# of training samples: %d\n" % int(kwargs['max_number_patches']))

    # Init loggers
    writer, logger = init_logging(kwargs)

    # Define GPU devices
    device_ids = [0]
    torch.backends.cudnn.benchmark = True  # CUDNN optimization

    # Create model
    model = DVDnetFull()
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    # Define loss
    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])

    # Resume training or start anew
    start_epoch, training_params = resume_training(kwargs, model, optimizer)

    # Training
    start_time = time.time()
    for epoch in range(start_epoch, kwargs['epochs']):
        # Set learning rate
        current_lr, reset_orthog = lr_scheduler(kwargs, epoch)
        if reset_orthog:
            training_params['no_orthog'] = True

        # set learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print(f'learning rate {current_lr}')

        for i, (seq, gt) in enumerate(loader_train):
            # 1) 切换到训练模式
            model.train()
            # 2) 梯度置零
            optimizer.zero_grad()

            # 3) 归一化增强 & 提取中心帧
            #    seq: [N, C*seq_len, H, W], gt: [N, C, H, W]  (都在 [0,1])
            img_train, gt_train = normalize_augment(seq, ctrl_fr_idx)

            device = next(model.parameters()).device
            img_train = img_train.to(device, non_blocking=True)
            gt_train = gt_train.to(device, non_blocking=True)

            N, _, H, W = img_train.size()

            stdn = torch.empty((N, 1, 1, 1), device=device).uniform_(
                kwargs['noise_ival'][0], kwargs['noise_ival'][1]
            )
            noise = torch.zeros_like(img_train)
            noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
            imgn_train = img_train + noise

            noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True)

            # 7) 前向 + 反向
            out_train = model(imgn_train, noise_map)
            print(gt_train.device, out_train.device)
            # loss = criterion(gt_train, out_train) / (N * 2)

            loss = criterion(out_train, gt_train) / (N * 2)
            #       print(f"▶ [epoch {epoch+1}][step {training_params['step']}] "
            # f"out_train.requires_grad={out_train.requires_grad}, "
            # f"loss.requires_grad={loss.requires_grad}")
            loss.backward()

            optimizer.step()

            # 8) 定期正交化 + 记录 PSNR
            if training_params['step'] % kwargs['save_every'] == 0:
                if not training_params['no_orthog']:
                    model.apply(orthogonal_conv_weights)
                log_train_psnr(out_train, gt_train, loss, writer, epoch, i, num_minibatches, training_params)
            training_params['step'] += 1

        # Call to model.eval() to correctly set the BN layers before inference
        model.eval()
        validate_and_log(
            model=model,
            dataset_val=dataset_val,
            valnoisestd=kwargs['val_noiseL'],
            temp_psz=kwargs['temp_patch_size'],
            writer=writer,
            epoch=epoch,
            lr=current_lr,
            logger=logger,
            trainimg=img_train
        )
        # save model and checkpoint
        torch.set_grad_enabled(True)
        model.train()

        training_params['start_epoch'] = epoch + 1
        save_model_checkpoint(model, kwargs, optimizer, training_params, epoch)

    # Print elapsed time
    elapsed_time = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    # Close logger file
    close_logger(logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the denoiser")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=10, help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true',
                        help="resume training from a previous checkpoint")
    parser.add_argument("--milestone", nargs=2, type=int, default=[6, 8],
                        help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--no_orthog", action='store_true', help="Don't perform orthogonalization as regularization")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Number of training steps to log psnr and perform orthogonalization")
    parser.add_argument("--save_every_epochs", type=int, default=5, help="Number of training epochs to save state")
    parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55], help="Noise training interval")
    parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
    # Preprocessing parameters
    parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
    parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
    parser.add_argument("--max_number_patches", "--m", type=int, default=256000, help="Maximum number of patches")
    # Dirs
    parser.add_argument("--log_dir", type=str, default="logs", help='path of log files')
    parser.add_argument("--trainset_dir", type=str, default="train_data", help='path of trainset')
    parser.add_argument("--valset_dir", type=str, default="eval_data", help='path of validation set')
    args = parser.parse_args()

    # Normalize noise between [0, 1]
    args.val_noiseL /= 255.0
    args.noise_ival[0] /= 255.0
    args.noise_ival[1] /= 255.0
    args_dict = vars(args)
    print("### Training FastDVDnet denoiser model ###")
    print("> Parameters:")
    for k, v in args_dict.items():
        print(f'{k}: {v}')
    print('\n')
    print("\n### Training FastDVDnet denoiser model ###")
    print("> Parameters:")

    main(**vars(args))
