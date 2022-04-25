# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License
# Written by Guoqiang Wei
# --------------------------------------------------------

import os
import sys
import time
import json
import math
import datetime
import pathlib
import argparse
import numpy as np
from contextlib import suppress

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.tensorboard as tb

from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, ModelEma, AverageMeter, accuracy


import utils
from dataloader import build_dataloader
from models import *


def parse_args():
    parser = argparse.ArgumentParser('AcitveMLP')

    # ------------------------ data ------------------------
    parser.add_argument('--batch-size',     default=64, type=int, help='training batch size')
    parser.add_argument('--input-size',     default=224, type=int)
    parser.add_argument('--num-workers',    default=10, type=int)
    parser.add_argument('--pin-mem',        action='store_true', help='pin CPU memory for dataLoader')
    parser.add_argument('--no-pin-mem',     action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--data-path',      default='', type=str, help='dataset path')
    parser.add_argument('--data-set',       default='IMNET', type=str)
    parser.add_argument('--map-path',       type=str, default='data', help='path to map.txt')

    # ------------------------ augmentation ------------------------
    # repeated augmentation
    parser.add_argument('--repeated-aug',   dest='repeated_aug', action='store_true')
    parser.add_argument('--no-repeated-aug', dest='repeated_aug', action='store_false')
    parser.set_defaults(repeated_aug=True)
    # mixup
    parser.add_argument('--mixup',          type=float, default=0.8, help='mixup alpha, disabled if = 0.')
    parser.add_argument('--cutmix',         type=float, default=1.0, help='cutmix alpha, disabled if = 0.')
    parser.add_argument('--cutmix-minmax',  type=float, nargs='+', default=None, help='cutmix min/max ratio')
    parser.add_argument('--mixup-prob',     type=float, default=1.0, help='prob. of performing mixup or cutmix')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='switching to cutmix when both enabled')
    parser.add_argument('--mixup-mode',     type=str, default='batch', help='way of applying mixup/cutmix. ["batch", "pair", "elem"]')
    # augmentation parameters
    parser.add_argument('--color-jitter',   type=float, default=0.4, help='color jitter factor')
    parser.add_argument('--aa',             type=str, default='rand-m9-mstd0.5-inc1', help=' AutoAugment policy.')
    parser.add_argument('--smoothing',      type=float, default=0.1, help='label smoothing')
    parser.add_argument('--train-interpolation', type=str, default='bicubic', help='training interpolation [random, bilinear, bicubic]')
    # random erasing
    parser.add_argument('--reprob',         type=float, default=0.25, help='random erasing')
    parser.add_argument('--remode',         type=str, default='pixel', help='random erasing')
    parser.add_argument('--recount',        type=int, default=1, help='random erasing count')
    parser.add_argument('--resplit',        action='store_true', default=False, help='do not random erase on first split')

    # ------------------------ training ------------------------
    parser.add_argument('--start_epoch',    default=0, type=int)
    parser.add_argument('--epochs',         default=300, type=int)
    # optimizer
    parser.add_argument('--opt',            default='adamw', type=str, help='optimizer')
    parser.add_argument('--opt-eps',        default=1e-8, type=float, help='optimizer epsilon')
    parser.add_argument('--opt-betas',      default=None, type=float, nargs='+', help='optimizer betas')
    parser.add_argument('--clip-grad',      type=float, default=None, help='clip gradient norm')
    parser.add_argument('--momentum',       type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay',   type=float, default=0.05, help='weight decay')
    # learning rate schedule
    parser.add_argument('--sched',          default='cosine', type=str, help='LR scheduler')
    parser.add_argument('--lr',             type=float, default=5e-4, help='learning rate')
    parser.add_argument('--lr-noise',       type=float, nargs='+', default=None, help='lr noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct',   type=float, default=0.67, help='learning rate noise limit percent')
    parser.add_argument('--lr-noise-std',   type=float, default=1.0, help='learning rate noise std-dev')
    parser.add_argument('--warmup-lr',      type=float, default=1e-6, help='warmup learning rate')
    parser.add_argument('--min-lr',         type=float, default=1e-5, help='lower lr bound for cyclic schedulers that hit 0')
    #
    parser.add_argument('--finetune',       default='', help='path to checkpoint used for finetuning')
    parser.add_argument('--auto_resume',    action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--resume',         default='', help='checkpoint to be resumed from')
    # lr
    parser.add_argument('--decay-epochs',   type=float, default=30, help='epoch interval to decay lr')
    parser.add_argument('--warmup-epochs',  type=int, default=5, help='epochs to warmup lr')
    parser.add_argument('--cooldown-epochs', type=int, default=10, help='epochs to cooldown lr at min_lr, after cyclic schedule ends')
    parser.add_argument('--decay-rate',     type=float, default=0.1, help='lr decay rate')

    # ------------------------ model ------------------------
    parser.add_argument('--model',          default='ActiveT', type=str, help='model name')
    parser.add_argument('--model-ema',      action='store_true')
    parser.add_argument('--no-model-ema',   action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='model ema decay rate')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False)
    # activemlp
    parser.add_argument('--drop-path-rate', type=float, default=0.1, help='drop path rate')

    # ------------------------ misc ------------------------
    parser.add_argument('--seed',           default=0, type=int)
    parser.add_argument('--output-dir',     default='exp', help='path to save logs and checkpoints')
    parser.add_argument('--device',         default='cuda', help='device for training/test')
    parser.add_argument('--amp',            dest='amp', action='store_true', help='using amp (default)')
    parser.add_argument('--no_amp',         dest='amp', action='store_false', help='not using amp')
    parser.set_defaults(amp=True)
    parser.add_argument('--eval',           action='store_true', help='evaluation only')
    parser.add_argument('--dist-eval',      action='store_true', default=False, help='distributed evaluation')
    parser.add_argument('--throughput',     action='store_true', default=False, help='calculate throughput on GPU')
    parser.add_argument('--save-ckpt-freq', type=int, default=1, help='frequency to save checkpoint')
    parser.add_argument('--print-freq',     type=int, default=25, help='frequency to display logger')
    # DDP
    parser.add_argument('--world_size',     default=1, type=int, help='# of distributed processes')
    parser.add_argument('--dist_url',       default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def main(args):
    # ------------------------ prepare ------------------------
    acc_max = .0
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    # output log and tensorboard
    output_dir = pathlib.Path(args.output_dir)
    logger = utils.create_logger(output_dir=args.output_dir, dist_rank=global_rank)
    if global_rank == 0:
        tb_logger = tb.SummaryWriter(log_dir=args.output_dir)

    # print info
    logger.info('>' * 25 + ' args ' + '<' * 25 + '\n' + utils.dict_to_string(vars(args), sort_keys=True))
    logger.info('>' * 25 + ' system info ' + '<' * 25 + '\n' + utils.dict_to_string(utils.collect_sys_info()))

    # scale learning rate
    lr_sclaed = args.lr * args.batch_size * world_size / 512.0
    args.lr = lr_sclaed

    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ------------------------ dataset ------------------------
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_dataloader(args)

    # ------------------------ model ------------------------
    logger.info(f">>> creating model: {args.model}")
    device = torch.device(args.device)
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path_rate,
    )
    # logger.info(f"{model.__repr__()}")
    # flops and # parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops_count = utils.cal_flops(model)
    logger.info(f">>> created {args.model}")
    logger.info(f"    FLOPs(MAdds): {flops_count:.3f}G | # params: {n_parameters / 1e6:.3f}M")

    # finetune from checkpoint
    if args.finetune:
        logger.info(f">>> finetune from {args.finetune}")
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        # filter classifier
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"    removing key {k} from loaded checkpoint")
                del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        logger.info(f"    {msg}")

    model.to(device)
    optimizer = create_optimizer(args, model)

    # AMP with torch native implementation (default)
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        logger.info('>>> training in mixed precision with torch AMP')
    else:
        amp_autocast = suppress
        loss_scaler = None
        logger.info('>>> training in float32')

    model_ema = None
    if args.model_ema:
        # !important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '',
                             resume='')

    model_without_ddp = model
    if args.distributed:
        if args.amp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # ------------------------ lr & loss ------------------------
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.mixup > 0.:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # ------------------------ auto resume ------------------------
    if args.auto_resume and args.resume == '':
        checkpoint = os.path.join(output_dir, 'checkpoint-last.pth')
        if os.path.isfile(checkpoint):
            args.resume = checkpoint
    if args.resume:
        logger.info(f">>> resume from {args.resume}")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'])
        logger.info(f"    loaded keys: {checkpoint.keys()}")
        logger.info(f"    missing keys: {missing_keys}")
        logger.info(f"    unexpected keys: {unexpected_keys}")
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            acc_max = checkpoint['acc_max'] if 'acc_max' in checkpoint.keys() else acc_max
            if args.model_ema:
                logger.info("    + loading model_ema")
                utils.load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                logger.info("    + loading loss_scaler")
                loss_scaler.load_state_dict(checkpoint['scaler'])

    # ------------------------ throughput ------------------------
    if args.throughput:
        utils.throughput(model, data_loader_val)
        return

    # ------------------------ eval ------------------------
    if args.eval:
        # eval
        logger.info(f'>>> start testing')
        acc1, acc5, _ = evaluate(args, data_loader_val, model, device, amp_autocast=amp_autocast, logger=logger)
        logger.info(f"    accuracy on {len(dataset_val)} test images: {acc1:.3f}% acc@1 | {acc5:.3f}% acc@5")
        # ema
        logger.info(f'>>> start testing with ema model')
        acc1_ema, acc5_ema, _ = evaluate(args, data_loader_val, model_ema.ema, device, amp_autocast=amp_autocast, logger=logger)
        logger.info(f"    [ema] accuracy on {len(dataset_val)} test images: {acc1_ema:.3f}% acc@1 | {acc5_ema:.3f}% acc@5")
        return

    # ------------------------ training ------------------------
    logger.info(f">>> start training from {args.start_epoch} to {args.epochs} epoch")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_lr, train_loss = train_one_epoch(
            args, model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn, amp_autocast=amp_autocast, logger=logger
        )
        lr_scheduler.step(epoch)

        # save checkpoint
        if dist.get_rank() == 0:
            utils.save_checkpoint(
                args, os.path.join(output_dir, 'checkpoint-last.pth'),
                epoch, model_without_ddp, model_ema, loss_scaler, acc_max, optimizer, lr_scheduler, logger
            )
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_checkpoint(
                    args, os.path.join(output_dir, f'checkpoint-{epoch}.pth'),
                    epoch, model_without_ddp, model_ema, loss_scaler, acc_max, optimizer, lr_scheduler, logger
                )

        # evaluation
        logger.info(f'>>> start testing')
        acc1, acc5, test_loss = evaluate(
            args, data_loader_val, model, device, amp_autocast=amp_autocast, logger=logger
        )
        acc_max = max(acc_max, acc1)
        logger.info(
            f"    accuracy on the {len(dataset_val)} test images: {acc1:.3f}% acc@1 | {acc5:.3f}% acc@5 | max accuray: {acc_max:.2f}%"
        )
        # evaluation with ema
        logger.info(f'>>> start testing with ema model')
        acc1_ema, acc5_ema, test_loss_ema = evaluate(
            args, data_loader_val, model_ema.ema, device, amp_autocast=amp_autocast, logger=logger
        )
        logger.info(
            f"    [ema] accuracy on the {len(dataset_val)} test images: {acc1_ema:.3f}% acc@1 | {acc5_ema:.3f}% acc@5"
        )

        # print/save log
        if dist.get_rank() == 0:
            tb_logger.add_scalar('test_acc1', acc1, global_step=epoch)
            tb_logger.add_scalar('test_acc5', acc5, global_step=epoch)
            tb_logger.add_scalar('ema_acc1', acc1_ema, global_step=epoch)
            tb_logger.add_scalar('ema_acc5', acc5_ema, global_step=epoch)
            tb_logger.add_scalar('train_lr', train_lr, global_step=epoch)
            tb_logger.add_scalar('train_loss', train_loss, global_step=epoch)
            tb_logger.add_scalar('test_loss', test_loss, global_step=epoch)
            tb_logger.add_scalar('ema_loss', test_loss_ema, global_step=epoch)
            log_stats = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_lr': train_lr,
                'test_acc1': acc1,
                'test_acc5': acc5,
                'test_loss': test_loss,
                'ema_acc1': acc1_ema,
                'ema_acc5': acc5_ema,
                'ema_loss': test_loss_ema,
                'n_params': n_parameters
            }
            with open(os.path.join(output_dir, "log.txt"), 'a') as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('training time {}'.format(total_time_str))


def train_one_epoch(args, model, criterion, data_loader, optimizer, device, epoch, loss_scaler, clip_grad, model_ema,
                    mixup_fn, amp_autocast=None, logger=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    data_time, batch_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    start = end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        data_time.update(time.time() - end)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with amp_autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        if not math.isfinite(loss.item()):
            logger.info(f">>> loss is {loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=clip_grad, parameters=model.parameters(),
                        create_graph=is_second_order)
        else:
            loss.backward(create_graph=is_second_order)
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # update logger
        losses.update(loss.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0 or idx == num_steps - 1:
            lr = optimizer.param_groups[0]['lr']
            mem_used = torch.cuda.max_memory_allocated() / (1024. * 1024.)
            eta = batch_time.avg * (num_steps - idx)
            logger.info(
                f'train: [{epoch}/{args.epochs - 1}] [{idx:>{len(str(num_steps))}}/{num_steps}] | '
                f'eta {datetime.timedelta(seconds=int(eta))} | '
                f'lr {lr:.6f} | '
                f'time {batch_time.val:.3f}/{batch_time.avg:.3f} | '
                f'data {data_time.val:.3f}/{data_time.avg:.3f} | '
                f'loss {losses.val:.4f}/{losses.avg:.4f} | '
                f'mem {mem_used:.0f}M'
            )
    epoch_time = time.time() - start
    logger.info(f'>>> epoch {epoch} takes {datetime.timedelta(seconds=int(epoch_time))}')

    return lr, losses.avg


@torch.no_grad()
def evaluate(args, data_loader, model, device, amp_autocast=None, logger=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    num_steps = len(data_loader)
    data_time, batch_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    acc1s, acc5s = AverageMeter(), AverageMeter()

    start = end = time.time()

    for idx, (images, target) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        data_time.update(time.time() - end)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1, acc5, loss = utils.reduce_tensor(acc1), utils.reduce_tensor(acc5), utils.reduce_tensor(loss)

        acc1s.update(acc1.item(), target.size(0))
        acc5s.update(acc5.item(), target.size(0))
        losses.update(loss.item(), target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0 or idx == num_steps - 1:
            mem_used = torch.cuda.max_memory_allocated() / (1024. * 1024.)
            eta = batch_time.avg * (num_steps - idx)
            logger.info(
                f'test: [{idx:>{len(str(num_steps))}}/{num_steps}] | '
                f'eta {datetime.timedelta(seconds=int(eta))} | '
                f'time {batch_time.val:.3f}/{batch_time.avg:.3f} | '
                f'data {data_time.val:.3f}/{data_time.avg:.3f} | '
                f'loss {losses.val:.4f}/{losses.avg:.4f} | '
                f'acc@1 {acc1s.val:.3f}/{acc1s.avg:.3f} | '                
                f'acc@5 {acc5s.val:.3f}/{acc5s.avg:.3f} | '
                f'mem {mem_used:.0f}M'
            )
    epoch_time = time.time() - start
    logger.info(f'>>> test takes {datetime.timedelta(seconds=int(epoch_time))}')

    return acc1s.avg, acc5s.avg, losses.avg


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # configure ddp
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'>>> distributed init on rank {args.rank}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    torch.distributed.barrier()

    main(args)
