import os
import torch
from glob import glob
from tqdm import tqdm

from utils import AverageMeter
from utils.metrics import DiceCoef
from utils.transforms import decode_preds

def train(net, dataset_trn, optimizer, criterion, epoch, opt):
    print("Start Training...")
    net.train()

    losses, necro_dices, ce_dices, peri_dices, total_dices = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    for it, (img, mask) in enumerate(dataset_trn):
        # Optimizer
        optimizer.zero_grad()

        # Load Data
        img, mask = torch.Tensor(img).float(), torch.Tensor(mask).float()
        if opt.use_gpu:
            img, mask = img.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        # Predict
        pred = net(img)

        # Loss Calculation
        loss = criterion(pred, mask)

        # Backward and step
        loss.backward()
        optimizer.step()

        # Calculation Dice Coef Score
        pred_decoded = torch.stack(decode_preds(pred), 0)
        necro_dice, ce_dice, peri_dice = DiceCoef(return_score_per_channel=True)(pred_decoded, mask[:,1:])
        total_dice = (necro_dice + ce_dice + peri_dice) / 3
        necro_dices.update(necro_dice.item(), img.size(0))
        ce_dices.update(ce_dice.item(), img.size(0))
        peri_dices.update(peri_dice.item(), img.size(0))
        total_dices.update(total_dice.item(), img.size(0))

        # Stack Results
        losses.update(loss.item(), img.size(0))

        if (it==0) or (it+1) % 10 == 0:
            print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Dice : Necro %.4f CE %.4f Peri %.4f Total %.4f'
                % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg, necro_dices.avg, ce_dices.avg, peri_dices.avg, total_dices.avg))

    print(">>> Epoch[%3d/%3d] | Training Loss : %.4f | Dice : Necro %.4f CE %.4f Peri %.4f Total %.4f\n"
        % (epoch+1, opt.max_epoch, losses.avg, necro_dices.avg, ce_dices.avg, peri_dices.avg, total_dices.avg))


def validate(dataset_val, net, criterion, optimizer, epoch, opt, best_dice, best_epoch):
    print("Start Evaluation...")
    net.eval()

    losses, necro_dices, ce_dices, peri_dices, total_dices = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    for it, (img, masks_cropped, masks_org, meta) in enumerate(dataset_val):
        # Load Data
        img, masks_cropped, masks_org = [torch.Tensor(tensor).float() for tensor in [img, masks_cropped, masks_org]]
        if opt.use_gpu:
            img, masks_cropped, masks_org = [tensor.cuda(non_blocking=True) for tensor in [img, masks_cropped, masks_org]]

        # Predict
        with torch.no_grad():
            pred = net(img)

        # Loss Calculation
        loss = criterion(pred, masks_cropped)

        # Evaluation Metric Calcuation
        pred_decoded = decode_preds(pred, meta, refine=True)
        for pred, gt in zip(pred_decoded, masks_org):
            necro_dice, ce_dice, peri_dice = DiceCoef(return_score_per_channel=True)(pred[None, ...], gt[None, ...])
            total_dice = (necro_dice + ce_dice + peri_dice) / 3

            necro_dices.update(necro_dice.item(), 1)
            ce_dices.update(ce_dice.item(), 1)
            peri_dices.update(peri_dice.item(), 1)
            total_dices.update(total_dice.item(), 1)

        # Stack Results
        losses.update(loss.item(), img.size(0))
        

        print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Dice : Necro %.4f CE %.4f Peri %.4f Total %.4f'
            % (epoch+1, opt.max_epoch, it+1, len(dataset_val), losses.avg, necro_dices.avg, ce_dices.avg, peri_dices.avg, total_dices.avg))

    print(">>> Epoch[%3d/%3d] | Test Loss : %.4f | Dice : Necro %.4f CE %.4f Peri %.4f Total %.4f"
        % (epoch+1, opt.max_epoch, losses.avg, necro_dices.avg, ce_dices.avg, peri_dices.avg, total_dices.avg))

    # Update Result
    if total_dices.avg > best_dice:
        print('Best Score Updated...')
        best_dice = total_dices.avg
        best_epoch = epoch

        # Remove previous weights pth files
        for path in glob('%s/*.pth' % opt.exp):
            os.remove(path)

        model_filename = '%s/epoch_%04d_dice%.4f_loss%.8f.pth' % (opt.exp, epoch+1, best_dice, losses.avg)

        # Single GPU
        if opt.ngpu == 1:
            torch.save(net.state_dict(), model_filename)
        # Multi GPU
        else:
            torch.save(net.module.state_dict(), model_filename)

    print('>>> Current best: Dice: %.8f in %3d epoch\n' % (best_dice, best_epoch+1))
    
    return best_dice, best_epoch


def evaluate(dataset_val, net, opt):
    print("Start Evaluation...")
    net.eval()

    necro_dices, ce_dices, peri_dices, total_dices = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    for img, masks_cropped, masks_org, meta in tqdm(dataset_val):
        # Load Data
        img, masks_cropped, masks_org = [torch.Tensor(tensor).float() for tensor in [img, masks_cropped, masks_org]]
        if opt.use_gpu:
            img, masks_cropped, masks_org = [tensor.cuda(non_blocking=True) for tensor in [img, masks_cropped, masks_org]]

        # Predict
        with torch.no_grad():
            pred = net(img)


        # Evaluation Metric Calcuation
        pred_decoded = decode_preds(pred, meta, refine=True)
        for pred, gt in zip(pred_decoded, masks_org):
            necro_dice, ce_dice, peri_dice = DiceCoef(return_score_per_channel=True)(pred[None, ...], gt[None, ...])
            total_dice = (necro_dice + ce_dice + peri_dice) / 3

            necro_dices.update(necro_dice.item(), 1)
            ce_dices.update(ce_dice.item(), 1)
            peri_dices.update(peri_dice.item(), 1)
            total_dices.update(total_dice.item(), 1)

    print("Evaluate Result | Dice : Necro %.4f CE %.4f Peri %.4f Total %.4f"
        % (necro_dices.avg, ce_dices.avg, peri_dices.avg, total_dices.avg))