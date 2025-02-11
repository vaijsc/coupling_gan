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
from models.ncsnpp_generator_adagn import NCSNpp
from pytorch_fid.fid_score import calculate_fid_given_paths
from dataset_bias_noise import getCleanData


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

def get_noise_func(noise_type, distribution, image_shape=None):
    if noise_type in ['movemean', 'std_independence']:
        def sample(batch_size):
            return distribution.sample((batch_size,))
        return sample
    elif noise_type in ['std_dependence']:
        def sample(batch_size):
            # ipdb.set_trace()
            noise_flattened = distribution.sample((batch_size,))
            # Reshape the samples back to the original shape
            noise = noise_flattened.view(batch_size, *image_shape)
            return noise
        return sample
    else:
        raise ValueError('Only support movemean, std_independence, std_dependence')

#%%
def sample_and_test(args, epoch_id):

    torch.manual_seed(123456)
    device = torch.device('cuda:{}'.format(0))
    # device = 'cuda:3'
    batch_size = args.batch_size
    exp = args.exp
    parent_dir = "./train_logs/{}".format(args.dataset)
    if args.perturb_percent > 0:
        parent_dir += f'_{int(args.perturb_percent)}p_{args.perturb_dataset}'
    parent_dir += f'/{args.noise_type}/{args.phi1}_{args.phi2}/{args.image_distance}/tau_{args.tau}/std_{args.diffuse_std}'
    exp_path = os.path.join(parent_dir, exp)

    samples_dir = f'{exp_path}/generated_samples'
    result_dir = f'{exp_path}/stat'
    checkpoint = f'{exp_path}/netG/netG_{epoch_id}.pth'
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
    elif args.dataset == 'celeba_256_5p_fashion':
        real_img_dir = 'pytorch_fid/celeba2_train_stat.npy'
    elif args.dataset == 'mnist_1c':
        real_img_dir = 'pytorch_fid/mnist_1c_train_stat.npy'
    elif args.dataset == 'fashion_mnist_1c':
        real_img_dir = 'pytorch_fid/fashion_mnist_1c_train_stat.npy'
    elif args.dataset == 'mnist_cifar10':
        real_img_dir = 'pytorch_fid/mnist_3c_120_train_stat.npy'
    elif args.dataset == 'mnist12':
        real_img_dir = 'pytorch_fid/mnist_1c_only_120_train_stat.npy'
    else:
        real_img_dir = args.real_img_dir
    print(real_img_dir)
    
    # to_range_0_1 = lambda x: (x + 1.) / 2.

    json_filename = f'{result_dir}/fid.json'
    print(json_filename)
    
    netG = NCSNpp(args).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    print(checkpoint)EMA
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()

    inject_dataset = getCleanData(args.inject_dataset, image_size=args.image_size)
    len_inject_dataset = len(inject_dataset)
    inject_dataloader = torch.utils.data.DataLoader(inject_dataset, batch_size=len_inject_dataset, num_workers=4, pin_memory=True, drop_last = True)
    for batch in inject_dataloader:
        all_origin_images, _ = batch     
        all_origin_images = all_origin_images.to(device, non_blocking=True)
        break
    mean_image = torch.mean(all_origin_images, dim=0).to(device, non_blocking=True)
    image_shape = mean_image.shape
    batch_mean_images = mean_image.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    mean_new_noise = torch.zeros(image_shape).to(device, non_blocking=True)
    if args.noise_type == 'movemean':
        std_new_noise = torch.ones(image_shape).to(device, non_blocking=True)
        normal_distribution = torch.distributions.Normal(mean_new_noise, std_new_noise) * args.diffuse_std
    elif args.noise_type == 'std_independence':
        std_new_noise = torch.std(all_origin_images, dim=0).to(device, non_blocking=True) * args.diffuse_std
        small_number = 0.1
        std_new_noise[std_new_noise <= 0] = small_number
        normal_distribution = torch.distributions.Normal(mean_new_noise, std_new_noise)
    elif args.noise_type == 'std_dependence':
        # Assuming all_origin_images is a tensor of shape (num_images, height, width, channels)
        all_origin_images_flattened = all_origin_images.view(all_origin_images.size(0), -1)

        # Calculate mean and covariance matrix
        mean_dependence = torch.mean(all_origin_images_flattened, dim=0).to(device, non_blocking=True)
        # covariance_matrix = torch.cov(all_origin_images_flattened, rowvar=False).to(device, non_blocking=True)
        # covariance_matrix = torch.cov(all_origin_images_flattened.t()).to(device, non_blocking=True)   
        # ipdb.set_trace()
        covariance_matrix = torch.matmul((all_origin_images_flattened - mean_dependence).t(), (all_origin_images_flattened - mean_dependence)) / (all_origin_images_flattened.size(0) - 1)
        
        small_constant = 1e-5
        covariance_matrix = covariance_matrix + small_constant * torch.eye(covariance_matrix.size(0)).to(device, non_blocking=True)

        # # Compute the eigenvalues of the regularized covariance matrix
        # eigenvalues_regularized = torch.linalg.eigvals(covariance_matrix)

        # # Convert complex eigenvalues to real numbers
        # real_eigenvalues_regularized = eigenvalues_regularized.real

        # # Print the negative eigenvalues
        # negative_eigenvalues = real_eigenvalues_regularized[real_eigenvalues_regularized <= 0]
        # if negative_eigenvalues.numel() > 0:
        #     print("Negative Eigenvalues (after regularization):")
        #     print(negative_eigenvalues)

        # ipdb.set_trace()
        # Create a multivariate normal distribution
        normal_distribution = torch.distributions.MultivariateNormal(mean_dependence, covariance_matrix)
        # import ipdb; ipdb.set_trace()

    global get_noise_func
    get_noise = get_noise_func(args.noise_type, normal_distribution, image_shape)

    iters_needed = 50000 //args.batch_size
    
    image_index = 0
    if args.compute_fid:
        for iter in range(iters_needed):

            with torch.no_grad():

                noise = get_noise(batch_size)
                latent_z = torch.randn(args.batch_size, args.nz, device=device)
                images = netG(noise, latent_z)
                images += batch_mean_images
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


        # Write the dictionary to a JSON file
        

        if not os.path.exists(json_filename):
            data = {}
            with open(json_filename, 'w') as json_file:
                json.dump(data, json_file)

        with open(json_filename, 'r') as file:
            data = json.load(file)
        data[f'Epoch {epoch_id}'] = fid
        data = dict(sorted(data.items()))
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
    parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
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

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--perturb_dataset', default='none', help='name of data added to the dataset')
    parser.add_argument('--perturb_percent', type=float, default=0, help='percentage of perturb_data')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--inject_dataset', default='cifar10_pretrained', help='name of dataset')
    parser.add_argument('--diffuse_std', type=float, default=1, help='standard deviation of diffuse images')
    parser.add_argument('--noise_type', type=str, default='movemean', help='type of initial noise to generate')


    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')

    parser.add_argument('--master_port', type=str, default='6020',
                        help='master port number')
        
    parser.add_argument('--epoch_start', type=int,default=300)
    parser.add_argument('--epoch_end', type=int,default=600)
    parser.add_argument('--epoch_jump', type=int,default=20)

    # Loss configurations
    parser.add_argument('--phi1', type=str, default='kl', choices=['linear', 'kl', 'softplus', 'chi'])
    parser.add_argument('--phi2', type=str, default='kl', choices=['linear', 'kl', 'softplus', 'chi'])
    parser.add_argument('--tau', type=float, default=0.001, help='proportion of the cost c')
    parser.add_argument('--image_distance', type=str, default='l2', help='type of loss for image')



    args = parser.parse_args()
    
    # for epoch_id in range(args.epoch_start, args.epoch_end + 1, args.epoch_jump):
    for epoch_id in range(args.epoch_end, args.epoch_start - 1, -args.epoch_jump):
        sample_and_test(args=args, epoch_id=epoch_id)
