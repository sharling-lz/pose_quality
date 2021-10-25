# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.transforms import fliplr_joints
from utils.oks import computeOKS
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, flip_pairs, model, criterion_heatmap, criterion_score, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    h_losses = AverageMeter()
    s_losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs, scores = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss_weight = 2000

        if isinstance(outputs, list):
            loss_heatmap = criterion_heatmap(outputs[0], target, target_weight)*loss_weight
            for output in outputs[1:]:
                loss_heatmap += criterion_heatmap(output, target, target_weight)*loss_weight
        else:
            output = outputs
            loss_heatmap = criterion_heatmap(output, target, target_weight)*loss_weight

        # loss = criterion(output, target, target_weight)

        # lin: score loss
        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        a = meta['area'].numpy()
        r = meta['rotation'].numpy()
        f = meta['flip'].numpy()
        gt_joints = meta['gt_joints'].numpy()
        joints_vis = meta['joints_vis'].numpy()
        w = meta['image_width'].numpy()
        bbox = meta['bbox'].numpy()

        preds, _ = get_final_preds(
            config, output.clone().detach().cpu().numpy(), c, s, r)

        # lin: if train image is flipped, needs to flip back predicted joints
        tmp = np.zeros((preds.shape[1],1))
        for it in range(preds.shape[0]):
            if f[it]:
                tmp_preds = np.concatenate((preds[it,:,:],tmp), axis=-1)
                tmp_preds, _ = fliplr_joints(
                    tmp_preds, joints_vis[it,:,:], w[it], flip_pairs)
                preds[it,:,:] = tmp_preds[:,0:2]

        oks = computeOKS(preds, gt_joints, a, bbox)
        oks = torch.from_numpy(oks).cuda(non_blocking=True)

        if isinstance(scores, list):
            loss_score = criterion_score(scores[0], oks)
            for score in scores[1:]:
                loss_score += criterion_score(score, oks)
        else:
            score = scores
            loss_score = criterion_score(score, oks)

        loss = loss_heatmap + loss_score

        # compute gradient and do update step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        h_losses.update(loss_heatmap.item(), input.size(0))
        s_losses.update(loss_score.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'H_Loss {h_loss.val:.5f} ({h_loss.avg:.5f})\t' \
                  'S_Loss {s_loss.val:.5f} ({s_loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, h_loss=h_losses, s_loss=s_losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_heatmap', h_losses.val, global_steps)
            writer.add_scalar('train_loss_score', s_losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion_heatmap, criterion_score, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    h_losses = AverageMeter()
    s_losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs, scores = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if isinstance(scores, list):
                score = scores[-1]
            else:
                score = scores

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped, scores_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                if isinstance(scores_flipped, list):
                    score_flipped = scores_flipped[-1]
                else:
                    score_flipped = scores_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]


                output = (output + output_flipped) * 0.5
                score = (score + score_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss_heatmap = criterion_heatmap(output, target, target_weight)
        
            # lin: pose score loss
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            r = meta['rotation'].numpy()
            bbox_score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s, r)

            if config.TEST.USE_GT_BBOX:
                gt_joints = meta['gt_joints'].numpy()
                bbox = meta['bbox'].numpy()
                a = meta['area'].numpy()

                oks = computeOKS(preds, gt_joints, a, bbox)
                oks = torch.from_numpy(oks).cuda(non_blocking=True)

                loss_score = criterion_score(score, oks)
            else:
                loss_score = torch.Tensor([0])
            

            num_images = input.size(0)
            # measure accuracy and record loss
            h_losses.update(loss_heatmap.item(), num_images)
            s_losses.update(loss_score.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()         

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2] = score.cpu().numpy()
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = bbox_score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Heatmap_Loss {h_loss.val:.4f} ({h_loss.avg:.4f})\t' \
                      'Score_Loss {s_loss.val:.4f} ({s_loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          h_loss=h_losses, s_loss = s_losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_heatmaploss',
                h_losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_scoreloss',
                s_losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
