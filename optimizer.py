"""
Pytorch Optimizer and Scheduler Related Task
"""
import math
import logging
import torch
from torch import optim
from config import cfg


def get_optimizer(args, net):
    """
    Decide Optimizer (Adam or SGD)
    """
    base_params = []

    max_iter = args.max_iter * args.num_domains if args.dsbn else args.max_iter

    for name, param in net.named_parameters(): # 网络层名字和参数迭代器
        base_params.append(param)

    if args.sgd and (args.arch != "network.rfnet.rfnet" and args.arch != "network.rfnet.rfnet_adapter"
                     and args.arch != "network.erfnet.erfnet" and args.arch != "network.erfnet.erfnet_adapter"): # 给上面获取的网络层参数添加优化方法
        optimizer = optim.SGD(base_params,
                            lr=args.lr,
                            weight_decay=5e-4, #args.weight_decay,
                            momentum=args.momentum,
                            nesterov=False)
    elif args.arch == "network.rfnet.rfnet" or args.arch == "network.rfnet.rfnet_adapter":
        train_params = [{'params': net.random_init_params()},
                        {'params': net.fine_tune_params(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
        optimizer = optim.Adam(train_params, lr=args.lr * 4,
                                    weight_decay=args.weight_decay * 4)
    elif args.arch == "network.erfnet.erfnet" or args.arch == "network.erfnet.erfnet_adapter":
        optimizer = optim.Adam(net.parameters(), args.lr, (0.9, 0.999),  eps=1e-08, weight_decay=args.weight_decay)
    else:
        raise ValueError('Not a valid optimizer')

    if args.lr_schedule == 'scl-poly':
        if cfg.REDUCE_BORDER_ITER == -1:
            raise ValueError('ERROR Cannot Do Scale Poly')

        rescale_thresh = cfg.REDUCE_BORDER_ITER
        scale_value = args.rescale
        lambda1 = lambda iteration: \
             math.pow(1 - iteration / max_iter,
                      args.poly_exp) if iteration < rescale_thresh else scale_value * math.pow(
                          1 - (iteration - rescale_thresh) / (max_iter - rescale_thresh),
                          args.repoly)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.lr_schedule == 'poly':
        lambda1 = lambda iteration: math.pow(1 - iteration / max_iter, args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    return optimizer, scheduler


#def load_weights(net, optimizer, scheduler, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file
    """
#    logging.info("Loading weights from model %s", snapshot_file)
#    net, optimizer, scheduler, epoch, mean_iu = restore_snapshot(net, optimizer, scheduler, snapshot_file,
#            restore_optimizer_bool) # checkpoints中除了结构和权重，还可能有optimizer和scheduler信息。
#    return epoch, mean_iu


def load_weights(net, optimizer, scheduler, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file
    """
    logging.info("Loading weights from model %s", snapshot_file)
    net, optimizer, scheduler, epoch = restore_snapshot(net, optimizer, scheduler, snapshot_file,
            restore_optimizer_bool) # checkpoints中除了结构和权重，还可能有optimizer和scheduler信息。
    return epoch, 1

def load_weights_adapter(net, optimizer, scheduler, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file with adapters
    """
    logging.info("Loading weights from model %s", snapshot_file)
    net, optimizer, scheduler, epoch = restore_snapshot_adapter(net, optimizer, scheduler, snapshot_file,
            restore_optimizer_bool) # checkpoints中除了结构和权重，还可能有optimizer和scheduler信息。
    return epoch, 1


#def restore_snapshot(net, optimizer, scheduler, snapshot, restore_optimizer_bool): # 权重的读取是先到一个checkpoints然后采用forgiving策略（partial loading）最终装载到我们定义的net模型中
    """
    #Restore weights and optimizer (if needed ) for resuming job.
    """
#    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
#    logging.info("Checkpoint Load Compelete")
#    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
#        optimizer.load_state_dict(checkpoint['optimizer'])
#    if scheduler is not None and 'scheduler' in checkpoint and restore_optimizer_bool:
#        scheduler.load_state_dict(checkpoint['scheduler'])

#    if 'state_dict' in checkpoint:
#        net = forgiving_state_restore(net, checkpoint['state_dict'])
#    else:
#        net = forgiving_state_restore(net, checkpoint)

#    return net, optimizer, scheduler, checkpoint['epoch'], checkpoint['mean_iu']


def restore_snapshot(net, optimizer, scheduler, snapshot, restore_optimizer_bool): # 权重的读取是先到一个checkpoints然后采用forgiving策略（partial loading）最终装载到我们定义的net模型中
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logging.info("Checkpoint Load Compelete")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint and restore_optimizer_bool:
        scheduler.load_state_dict(checkpoint['scheduler'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer, scheduler, checkpoint['epoch']

def restore_snapshot_adapter(net, optimizer, scheduler, snapshot, restore_optimizer_bool): # 权重的读取是先到一个checkpoints然后采用forgiving策略（partial loading）最终装载到我们定义的net模型中
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logging.info("Checkpoint Load Compelete")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint and restore_optimizer_bool:
        scheduler.load_state_dict(checkpoint['scheduler'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore_adapter(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer, scheduler, checkpoint['epoch']


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            print("Skipped loading parameter", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

def forgiving_state_restore_adapter(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        # adapter_restore_BN
        if ('layer2' in k or 'layer3' in k or 'layer4' in k or 'spp' in k or 'upsample' in k or 'logits' in k) and ('bns' in k):
            num = k.split(".")[-2]
            num_char = num + "." + k.split(".")[-1]
            bns_name = k.replace(num_char, num_char.replace(num, "0"))
            new_loaded_dict[k] = loaded_dict[bns_name]
            # print(bns_name)
            # print(k)
        # adapter_restore_conv
        elif ('layer2' in k or 'layer3' in k or 'layer4' in k or 'spp' in k or 'upsample' in k or 'logits' in k) and ('parallel_conv' in k):
            num = k.split(".")[-3]
            num_char = num + ".conv.weight"
            parallel_conv_name = k.replace(num_char, num_char.replace(num, "0"))
            new_loaded_dict[k] = loaded_dict[parallel_conv_name]
            # print(parallel_conv_name)
            # print(k)
        elif k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            print("Skipped loading parameter", k)
            logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

def forgiving_state_copy(target_net, source_net):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = target_net.state_dict()
    loaded_dict = source_net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
            print("Matched", k)
        else:
            print("Skipped loading parameter ", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    target_net.load_state_dict(net_state_dict)
    return target_net
