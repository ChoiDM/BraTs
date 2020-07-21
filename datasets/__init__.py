from datasets.SevBrats import SevBraTsDataset2D, SevBraTsDataset3D
from torch.utils.data import DataLoader

def get_dataloader(opt):
    if opt.in_dim == 2:
        trn_dataset = SevBraTsDataset2D(opt.data_root, opt, is_Train=True, augmentation=opt.augmentation)
        val_dataset = SevBraTsDataset2D(opt.data_root, opt, is_Train=False, augmentation=False)
    
    elif opt.in_dim == 3:
        trn_dataset = SevBraTsDataset3D(opt.data_root, opt, is_Train=True, augmentation=opt.augmentation)
        val_dataset = SevBraTsDataset3D(opt.data_root, opt, is_Train=False, augmentation=False)
    
    else:
        raise ValueError("'in_dim' option must be 2(D) or 3(D)")

    train_dataloader = DataLoader(trn_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.workers)

    valid_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.workers)
    
    return train_dataloader, valid_dataloader