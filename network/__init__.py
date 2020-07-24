import os
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
    
    if opt.resume:
        if os.path.isfile(opt.resume):
            pretrained_dict = torch.load(opt.resume, map_location=torch.device('cpu'))
            model_dict = net.state_dict()

            match_cnt = 0
            mismatch_cnt = 0
            pretrained_dict_matched = dict()
            for k, v in pretrained_dict.items():
                if k in model_dict and v.size() == model_dict[k].size():
                    pretrained_dict_matched[k] = v
                    match_cnt += 1
                else:
                    mismatch_cnt += 1
                    
            model_dict.update(pretrained_dict_matched) 
            net.load_state_dict(model_dict)

            print("=> Successfully loaded weights from %s (%d matched / %d mismatched)" % (opt.resume, match_cnt, mismatch_cnt))

        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))


    return net