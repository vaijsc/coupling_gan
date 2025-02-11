# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import numpy as np
import json
import os

import torchvision
from models.ncsnpp_generator_adagn_no_latent import NCSNpp
from pytorch_fid.fid_score import calculate_fid_given_paths


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def sample_from_model(generator, x_init, opt):
    noise = x_init
    images = generator(noise)
    images = (0.5*(images+1)).detach().cpu()
    return images

#%%
def sample_and_test(args, epoch_id):

    torch.manual_seed(123456)
    device = torch.device('cuda:{}'.format(0))
    # device = 'cuda:3'
    exp = args.exp
    # parent_dir = "./train_logs/{}".format(args.dataset)
    parent_dir = f"./train_logs/{args.dataset}/{args.datasetpair}"
    exp_path = os.path.join(parent_dir, exp)

    samples_dir = f'{exp_path}/generated_samples'
    result_dir = f'{exp_path}/stat'
    checkpoint = f'{exp_path}/netG_{epoch_id}.pth'
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'lsun':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    elif args.dataset == 'stl10':
        real_img_dir = 'pytorch_fid/stl10_stat.npy'
    elif args.dataset == 'sketch_64':
        real_img_dir = 'pytorch_fid/sketch_64_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celeba2_train_stat.npy'
    elif args.dataset == 'celebahq_256':
        real_img_dir = 'pytorch_fid/celebahq_train_stat.npy'
    else:
        real_img_dir = args.real_img_dir
    print(real_img_dir)
    
    # to_range_0_1 = lambda x: (x + 1.) / 2.

    json_filename = f'{result_dir}/fid_epoch_{epoch_id}.json'
    print(json_filename)
    
    netG = NCSNpp(args).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    print(checkpoint)
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()

    iters_needed = 50000 //args.batch_size
    
    image_index = 0
    if args.compute_fid:
        for iter in range(iters_needed):

            with torch.no_grad():

                noise = torch.randn(args.batch_size, args.num_channels,args.image_size, args.image_size).to(device)

                images = netG(noise)
                images = (0.5*(images+1)).detach().cpu()
                # torchvision.utils.save_image(images, os.path.join(exp_path, 'test_{}.png'.format(epoch_id)))

                for fake_sample in images: 
                    torchvision.utils.save_image(fake_sample, f'{samples_dir}/{image_index}.jpg')
                    image_index += 1
                print('generating batch ', iter)

        
        paths = [samples_dir, real_img_dir]
        
        kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
        # Specify the filename for the JSON file
        
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))

        data = {'fid': fid}


        # Write the dictionary to a JSON file
        with open(json_filename, 'w') as json_file:
            json.dump(data, json_file)
    else:
        for iter in range(iters_needed):

            with torch.no_grad():

                noise = torch.randn(args.batch_size, args.num_channels,args.image_size, args.image_size).to(device)

                images = netG(noise)
                images = (0.5*(images+1)).detach().cpu()
                # torchvision.utils.save_image(images, os.path.join(exp_path, 'test_{}.png'.format(epoch_id)))

                for fake_sample in images: 
                    torchvision.utils.save_image(fake_sample, f'{samples_dir}/{image_index}.jpg')
                    image_index += 1
                print('generating batch ', iter)

        # paths = [samples_dir, real_img_dir]
        # kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
        # # Specify the filename for the JSON file
        
        # fid = calculate_fid_given_paths(paths=paths, **kwargs)
        # print('FID = {}'.format(fid))

        # data = {'fid': fid}


        # # Write the dictionary to a JSON file
        # with open(json_filename, 'w') as json_file:
        #     json.dump(data, json_file)

    
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('UOT parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    
    # Generator configurations
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument('--num_channels_dae', type=int, default=128, help='number of initial channels in denoising model')
    parser.add_argument('--n_mlp', type=int, default=4, help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1,2,2,2], help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True, help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    # parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--resblock_type', default='biggan_no_latent', help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'], help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'], help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'], help='progressive combine method.')
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'], help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False, help='use tanh for last layer')
    parser.add_argument('--z_emb_dim', type=int, default=256, help='embedding dimension of z')
    parser.add_argument('--nz', type=int, default=100, help='latent dimension')
    parser.add_argument('--ngf', type=int, default=64, help='The default number of channels of model')
    
    #genrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'cifar10+mnist', 'lsun', 'celeba_256'], help='name of dataset')
    parser.add_argument('--datasetpair', default='uncond_cifar10_edm', help='name of dataset generated by a diffusion model')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')

    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')

    parser.add_argument('--master_port', type=str, default='6020',
                        help='master port number')
        
    parser.add_argument('--epoch_start', type=int,default=300)
    parser.add_argument('--epoch_end', type=int,default=600)
    parser.add_argument('--epoch_jump', type=int,default=20)









    # parser = argparse.ArgumentParser('UOT parameters')
    
    # # Experiment description
    # parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')
    # parser.add_argument('--exp', default='linear', help='name of experiment')
    # parser.add_argument('--resume', action='store_true',default=False, help='Resume training or not')
    # parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'cifar10+mnist', 'lsun', 'celeba_256'], help='name of dataset')
    # parser.add_argument('--datasetpair', default='uncond_cifar10_edm', help='name of dataset generated by a diffusion model')
    # parser.add_argument('--image_size', type=int, default=32, help='size of image')
    # parser.add_argument('--num_channels', type=int, default=3, help='channel of image')
    
    # # Generator configurations
    # parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    # parser.add_argument('--num_channels_dae', type=int, default=128, help='number of initial channels in denoising model')
    # parser.add_argument('--n_mlp', type=int, default=4, help='number of mlp layers for z')
    # parser.add_argument('--ch_mult', nargs='+', type=int, default=[1,2,2,2], help='channel multiplier')
    # parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    # parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
    # parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    # parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='always up/down sampling with conv')
    # parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
    # parser.add_argument('--fir', action='store_false', default=True, help='FIR')
    # parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    # parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    # # parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
    # parser.add_argument('--resblock_type', default='biggan_no_latent', help='tyle of resnet block, choice in biggan and ddpm')
    # parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'], help='progressive type for output')
    # parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'], help='progressive type for input')
    # parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'], help='progressive combine method.')
    # parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'], help='type of time embedding')
    # parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
    # parser.add_argument('--not_use_tanh', action='store_true', default=False, help='use tanh for last layer')
    # parser.add_argument('--z_emb_dim', type=int, default=256, help='embedding dimension of z')
    # parser.add_argument('--nz', type=int, default=100, help='latent dimension')
    # parser.add_argument('--ngf', type=int, default=64, help='The default number of channels of model')
    
    # # Training/Optimizer configurations
    # parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    # parser.add_argument('--num_epoch', type=int, default=600, help='the number of epochs')
    # parser.add_argument('--lr_g', type=float, default=1.6e-4, help='learning rate g')
    # parser.add_argument('--lr_d', type=float, default=1.0e-4, help='learning rate d')
    # parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    # parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    # parser.add_argument('--schedule', type=int, default=1800, help='cosine scheduler, learning rate 1e-5 until {schedule} epoch')
    # parser.add_argument('--use_ema', action='store_false', default=True, help='use EMA or not')
    # parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    # # Loss configurations
    # parser.add_argument('--phi1', type=str, default='kl', choices=['linear', 'kl', 'softplus', 'chi'])
    # parser.add_argument('--phi2', type=str, default='kl', choices=['linear', 'kl', 'softplus', 'chi'])
    # parser.add_argument('--tau', type=float, default=0.001, help='proportion of the cost c')
    # parser.add_argument('--r1_gamma', type=float, default=0.2, help='coef for r1 reg')
    # parser.add_argument('--alpha', type=float, default=0.5, help='weight of the real data')
    # parser.add_argument('--alpha_decay', type=float, default=1, help='decay factor of alpha')
    # parser.add_argument('--l1_weight', type=float, default=6, help='weight of the L1 loss')
    # parser.add_argument('--l1_amplification', type=float, default=1.001, help='amplification of l1 weight')
    # parser.add_argument('--dfeature_weight', type=float, default=0.5, help='weight of discriminator features')
        
    # # Visualize/Save configurations
    # parser.add_argument('--print_every', type=int, default=100, help='print current loss for every x iterations')
    # parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    # parser.add_argument('--save_ckpt_from_epoch', type=int, default=0, help='save ckpt from an epoch')
    # parser.add_argument('--save_ckpt_every', type=int, default=20, help='save ckpt every x epochs')
    # parser.add_argument('--save_image_every', type=int, default=10, help='save images every x epochs')

    # ###ddp
    # parser.add_argument('--num_proc_node', type=int, default=1,
    #                     help='The number of nodes in multi node env.')
    # parser.add_argument('--num_process_per_node', type=int, default=1,
    #                     help='number of gpus')
    # parser.add_argument('--node_rank', type=int, default=0,
    #                     help='The index of node.')
    # parser.add_argument('--local_rank', type=int, default=0,
    #                     help='rank of process in the node')
    # parser.add_argument('--master_address', type=str, default='127.0.0.1',
    #                     help='address for master')
    # parser.add_argument('--master_port', type=str, default='6020',
    #                     help='master port number')
    
    # parser.add_argument('--epoch_start', type=int,default=300)
    # parser.add_argument('--epoch_end', type=int,default=600)
    # parser.add_argument('--epoch_jump', type=int,default=20)

    # parser.add_argument('--compute_fid', action='store_true', default=False,
    #                         help='whether or not compute FID')

    args = parser.parse_args()
    
    # for epoch_id in range(args.epoch_start, args.epoch_end + 1, args.epoch_jump):
    for epoch_id in range(args.epoch_end, args.epoch_start - 1, -args.epoch_jump):
        sample_and_test(args=args, epoch_id=epoch_id)
