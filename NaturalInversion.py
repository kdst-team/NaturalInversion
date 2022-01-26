from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import math
import numbers
import numpy as np
import os
import glob
import collections
from PIL import Image
import sys 
from tqdm import tqdm
from network import Generator, Feature_Decoder
from resnet import ResNet34
import vgg

sys.path.insert(0,os.path.abspath('..'))

NUM_CLASSES = 10
ALPHA=1.0
image_list=[]
target_list=[]

debug_output = False
debug_output = True

# To fix Seed
def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class NaturalInversionFeatureHook():
    def __init__(self, module, rs):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.rs = rs

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type())  - mean, 2)

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def make_grid(tensor, nrow, padding = 2, pad_value : int = 0):
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


def get_images(net, 
                num_classes=10,
                bs=256, 
                epochs=2000, 
                prefix=None, 
                global_iteration=0, 
                bn_reg_scale=10,
                g_lr=0.001,
                d_lr=0.0005,
                a_lr=0.05,
                var_scale=0.001, 
                l2_coeff=0.00001
            ):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    best_cost = 1e6
    
    generator = Generator(8, 1034, 3).to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=g_lr)

    #### Feature_Map Decoder
    feature = Feature_Decoder().to(device)
    optimizer_f = torch.optim.Adam(feature.parameters(), lr=d_lr)

    # Learnable Scale Parameter
    alpha = torch.empty((bs,3,1,1), requires_grad=True, device=device)
    torch.nn.init.normal_(alpha, 5.0, 1)
    optimizer_alpha = torch.optim.Adam([alpha], lr=a_lr)

    # set up criteria for optimization
    criterion = nn.CrossEntropyLoss()
    optimizer_g.state = collections.defaultdict(dict)
    optimizer_f.state = collections.defaultdict(dict)  # Reset state of optimizer
    optimizer_alpha.state = collections.defaultdict(dict)
    
    cls_idx_list = [i for i in range(num_classes)]
    rand_pick = 6 if num_classes==10 else 56
    rand_picked_cls = random.sample(cls_idx_list, rand_pick)

    picked = 250 if num_classes==10 else 200
    targets = [i % 10 for i in range(picked)] + rand_picked_cls
    
    targets = torch.LongTensor(targets).to('cuda')
    tf_targets = F.one_hot(targets, 10)

    z = torch.randn((bs, 1024)).to(device)
    z = torch.cat((z,tf_targets), dim = 1)

    ## Create hooks for feature statistics catching
    loss_r_feature_layers = []
    count = 0
    
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(NaturalInversionFeatureHook(module, 0))
    
    lim_0, lim_1 = 2, 2

    for epoch in tqdm(range(epochs), leave=False, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        # Concat Z
        
        ##### step1
        inputs_jit = generator(z)
        
        # apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs_jit, shifts=(off1,off2), dims=(2,3))
        
        # Apply random flip 
        flip = random.random() > 0.5
        if flip:
            inputs_jit = torch.flip(inputs_jit, dims = (3,))
        
        ##### step2
        input_for_f = inputs_jit.clone().detach()
        with torch.no_grad():
            _, f5, f4, f3, f2, f1 = net(input_for_f)
        
        inputs_jit, addition = feature(inputs_jit, f1, f2, f3, f4, f5)

        ##### step3
        inputs_jit = inputs_jit * alpha
        inputs_for_save = inputs_jit.data.clone()

        # apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs_jit, shifts=(off1,off2), dims=(2,3))
        
        # Apply random flip 
        flip = random.random() > 0.5
        if flip:
            inputs_jit = torch.flip(inputs_jit, dims = (3,))

        outputs, f5, f4, f3, f2, f1 = net(inputs_jit)
        
        loss_target = criterion(outputs, targets)
        loss = loss_target

        # apply total variation regularization
        diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
        diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
        diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
        diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss = loss + var_scale*loss_var

        # R_feature loss
        loss_distr = sum([mod.r_feature for idx, mod in enumerate(loss_r_feature_layers)])
        loss = loss + bn_reg_scale*loss_distr # best for noise before BN

        # l2 loss
        loss = loss + l2_coeff * torch.norm(inputs_jit, 2)

        if debug_output and epoch % 100==0:
            print("It {}\t Losses: total: {:.3f},\ttarget: {:.3f} \tR_feature_loss unscaled:\t {:.3f}\tstyle_loss : {:.3f}".format(epoch, loss.item(),loss_target,loss_distr.item(), 0))
            nchs = inputs_jit.shape[1]
            vutils.save_image(inputs_jit.data.clone(),
                './{}/generator_{}.png'.format(prefix, str(epoch//100).zfill(2)),
                normalize=True, scale_each=True, nrow=10)

        if best_cost > loss.item():
            best_cost = loss.item()
            with torch.no_grad():
                best_inputs = generator(z)
                _, f5, f4, f3, f2, f1 = net(best_inputs)
                best_inputs, addition = feature(best_inputs, f1, f2, f3, f4, f5)
                best_inputs *= alpha
        
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        optimizer_alpha.zero_grad()

        # backward pass
        loss.backward()

        optimizer_g.step()
        optimizer_f.step()
        optimizer_alpha.step()

    return best_inputs, targets


def save_finalimages(images, targets, num_generations, prefix, exp_descr):
    # method to store generated images locally
    local_rank = torch.cuda.current_device()
    images = images.data.clone()

    for id in range(images.shape[0]):
        class_id = str(targets[id].item()).zfill(2)
        image = images[id].reshape(3,32,32)
        image_np = images[id].data.cpu().numpy()
        pil_image = torch.from_numpy(image_np)

        save_pth = os.path.join(prefix, 'final_images/s{}'.format(class_id))
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        vutils.save_image(image, os.path.join(prefix, 'final_images/s{}/{}_output_{}_'.format(class_id, num_generations, id)) + exp_descr + '.png', normalize=True, scale_each=True, nrow=1)


def test(net):
    print('==> Teacher validation')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def main(args):
    print("loading pre-trained classifier")
    net_teacher = ResNet34()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net_teacher = net_teacher.to(device)
    criterion = nn.CrossEntropyLoss()

    # for reproducability
    random_seed(777)
    
    num_classes = 10 if args.dataset=='cifar10' else 100
    
    arch = {'vgg11' : vgg.__dict__['vgg11_bn'](num_classes=num_classes),
            'vgg16' : vgg.__dict__['vgg16_bn'](num_classes=num_classes),
            'resnet34' : ResNet34(num_classes=num_classes)
            }

    net_teacher = arch[args.arch].to(device)
    checkpoint = torch.load(args.teacher_weights)
    net_teacher.load_state_dict(checkpoint)
    net_teacher.eval()
    
    cudnn.benchmark = True
    
    prefix_ = args.exp_name
    prefix = os.path.join(prefix_, str(args.global_iter)+"/")

    for create_folder in [prefix, prefix+"/final_images/"]:
        if not os.path.exists(create_folder):
            os.makedirs(create_folder)

    if 0:
        # for check teacher accuracy
        transform_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(root='../data/CIFAR10', train=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=True, num_workers=6,
                                             drop_last=True)
        # Checking teacher accuracy
        print("Checking teacher accuracy")
        test(net_teacher)

    print("Starting model inversion")
    inputs, targets = get_images(net=net_teacher, 
                                num_classes=num_classes,
                                bs=args.bs, 
                                epochs=args.iters_mi, 
                                prefix=prefix, 
                                global_iteration=args.global_iter, 
                                bn_reg_scale=args.r_feature_weight,
                                g_lr=args.G_lr,
                                d_lr=args.D_lr,
                                a_lr=args.A_lr,
                                var_scale=args.var_scale, 
                                l2_coeff=args.l2_scale
                                )
    
    save_finalimages(inputs, targets, args.global_iter, prefix_, args.exp_descr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Inversion')
    parser.add_argument('--ngpu', type=str, default='0',
                        help='device number') 
    parser.add_argument('--dataset', type=str, 
                        choices=['cifar10', 'cifar100'], help='dataset to invert [cifar10/cifar100]')
    parser.add_argument('--arch', type=str, 
                        choices=['vgg11', 'vgg16', 'resnet34'], 
                        help='set the pre-trained teacher network architecture [vgg11, vgg16, resnet34]')
    parser.add_argument('--bs', default=256, type=int, \
                        help='batch size')
    parser.add_argument('--iters_mi', default=2000, type=int, 
                        help='number of iterations for model inversion')
    parser.add_argument('--G_lr', default=0.001, type=float, 
                        help='lr for deep inversion')
    parser.add_argument('--D_lr', default=0.0005, type=float, 
                        help='lr for deep inversion')
    parser.add_argument('--A_lr', default=0.05, type=float, 
                        help='lr for deep inversion')
    parser.add_argument('--var_scale', default=6.0e-3, type=float,
                        help='TV L2 regularization coefficient')
    parser.add_argument('--l2_scale', default=1.5e-5, type=float, 
                        help='L2 regularization coefficient')
    parser.add_argument('--r_feature_weight', default=10.0, type=float, 
                        help='weight for BN regularization statistic')
    parser.add_argument('--teacher_weights', default='./pretrained/resnet34.pt', type=str, 
                        help='path to load weights of the model')
    parser.add_argument('--exp_name', default='sample_image',type=str, 
                        help='path to save final inversion images')
    parser.add_argument('--global_iter', type=int, 
                        help='global itertation number')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

    main(args)
