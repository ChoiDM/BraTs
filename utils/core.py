import os
import torch
from glob import glob

from utils import AverageMeter
from utils.metrics import DiceCoef

def train(net, dataset_trn, optimizer, criterion, epoch, opt):
    print("Start Training...")
    net.train()

    losses = AverageMeter()

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

        # Stack Results
        losses.update(loss.item(), img.size(0))

        if (it==0) or (it+1) % 10 == 0:
            print('Epoch[%3d/%3d] | Iteration[%3d/%3d] | Loss %.4f'
                % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg))

    print(">>> Epoch[%3d/%3d] Training Loss : %.8f\n" % (epoch+1, opt.max_epoch, losses.avg))


def validate(dataset_val, net, criterion, optimizer, epoch, opt, best_dice, best_epoch):
    print("Start Evaluation...")
    net.eval()

    losses, dice = AverageMeter(), AverageMeter()

    for it, (img, mask) in enumerate(dataset_val):
        # Load Data
        img, mask = torch.Tensor(img).float(), torch.Tensor(mask).float()
        if opt.use_gpu:
            img, mask = img.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        # Predict
        with torch.no_grad():
            pred = net(img)

        # Loss Calculation
        loss = criterion(pred, mask)

        # Evaluation Metric Calcuation
        dice_score = DiceCoef(sigmoid_normalization=False)(pred, mask)

        # Stack Results
        losses.update(loss.item(), img.size(0))
        dice.update(dice_score.item(), img.size(0))

        print('Epoch[%3d/%3d] | Iteration[%3d/%3d] | Loss %.4f | Dice %.4f'
            % (epoch+1, opt.max_epoch, it+1, len(dataset_val), losses.avg, dice.avg))

    print(">>> Epoch[%3d/%3d] Evalaution Loss : %.8f Dice %.4f" % (epoch+1, opt.max_epoch, losses.avg, dice.avg))

    # Update Result
    if dice.avg > best_dice:
        print('Best Score Updated...')
        best_dice = dice.avg
        best_epoch = epoch

        # Remove previous weights pth files
        for path in glob('%s/*.pth' % opt.exp):
            os.remove(path)

        model_filename = '%s/epoch_%04d_dice%.4f_loss%.8f.pth' % (opt.exp, epoch, best_dice, losses.avg)

        # Single GPU
        if opt.ngpu == 1:
            torch.save(net.state_dict(), model_filename)
        # Multi GPU
        else:
            torch.save(net.module.state_dict(), model_filename)

    print('>>> Current best: Dice: %.8f in %3d epoch\n' % (best_dice, best_epoch))
    
    return best_dice, best_epoch