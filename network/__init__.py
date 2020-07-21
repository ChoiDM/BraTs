import torch
from network.modified_unet import Modified2DUNet, Modified3DUNet

def create_model(opt):
    # Load network
    if opt.in_dim == 2:
        net = Modified2DUNet(opt.in_channels, opt.n_classes, opt.base_n_filter)
    elif opt.in_dim == 3:
        net = Modified3DUNet(opt.in_channels, opt.n_classes, opt.base_n_filter)
    else:
        raise ValueError("Check input dimension option.")

    # GPU settings
    if opt.use_gpu:
        net.cuda()
        if opt.ngpu > 1:
            net = torch.nn.DataParallel(net)
    
    return net