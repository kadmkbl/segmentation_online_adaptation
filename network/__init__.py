"""
Network Initializations
"""

import logging
import importlib
import torch
import datasets



def get_net(args, criterion, criterion_aux=None):
    """
    Get Network Architecture based on arguments provided
    """
    if "campuse1" == args.dataset[0]:
        num_classes = 10
    else:
        num_classes = datasets.num_classes
        
    if "new_campuse1" == args.dataset[0]:
        num_classes = 31
    else:
        num_classes = datasets.num_classes

    net = get_model(args=args, num_classes=num_classes,
                    criterion=criterion, criterion_aux=criterion_aux)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.3f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def warp_network_in_dataparallel(net, gpuid):
    """
    Wrap the network in Dataparallel
    """
    # torch.cuda.set_device(gpuid)
    # net.cuda(gpuid)    
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid], find_unused_parameters=True)
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid])#, find_unused_parameters=True)
    return net


def get_model(args, num_classes, criterion, criterion_aux=None):
    """
    Fetch Network Function Pointer
    """
    network = args.arch
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module) # 动态导入network文件夹下的对应模型(如network.deepv3.py)
    net_func = getattr(mod, model) # 读取对应模型的Class(如DeepR50V3PlusD)
    net = net_func(args=args, num_classes=num_classes, criterion=criterion, criterion_aux=criterion_aux) # 根据args最终确定模型（结构，输出类别，loss）
    return net
