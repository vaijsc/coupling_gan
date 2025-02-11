import argparse
import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
from datetime import datetime
from utils import *
# from dataset_old_multi_gpu import get_dataset
from dataset_bias_noise import getCleanData, getMixedData
from torch.multiprocessing import Process
import torch.distributed as dist
import sys
import lpips
from DISTS_pytorch import DISTS

# ddp
def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def get_image_distance(image_distance):
    if image_distance == 'l1':
        def get_l1_loss(image1, image2):
            return torch.abs(image1 - image2).view(image2.size(0), -1)
        return get_l1_loss
    if image_distance == 'l2':
        def get_l2_loss(image1, image2):
            return torch.pow(image1 - image2, 2).view(image2.size(0), -1)
        return get_l2_loss
    elif image_distance == 'vgg_loss':
        return lpips.LPIPS(net='vgg')
    elif image_distance == 'alex_loss':
        return lpips.LPIPS(net='alex')
    elif image_distance == 'dists':
        D = DISTS()
        def get_dists(image1, image2):
            image1_rescale = (image1 + 1) / 2
            image2_rescale = (image2 + 1) / 2
            # return D(image1_rescale, image2_rescale, require_grad=True, batch_average=False).unsqueeze(1) 
            return D(image1_rescale, image2_rescale).unsqueeze(1)
        return get_dists
    else:
        raise ValueError('Only support l1, l2, vgg_loss, alex_loss, dists')


def train(rank, gpu, args):
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))


    batch_size = args.batch_size
    nz = args.nz
    
    
    # Set Generator
    from models.ncsnpp_generator_adagn import NCSNpp
    netG = NCSNpp(args).to(device)
    netNoise = NCSNpp(args).to(device)
    
    
    # Set potential
    if args.dataset in ['mnist','cifar10','cifar10+mnist', 'cifar10+3mnist', 'cifar10+5mnist', 'stl10', 'celeba_64_5p_fashion', 'cifar10_pretrained', 'mnist_pretrained', 'mnist']:
        from models.discriminator import Discriminator_small
        netD = Discriminator_small(nc = args.num_channels, ngf = args.ngf, act=nn.LeakyReLU(0.2)).to(device)
    else:
        from models.discriminator import Discriminator_large
        netD = Discriminator_large(nc = args.num_channels, ngf = args.ngf, act=nn.LeakyReLU(0.2)).to(device)

    # ddp
    broadcast_params(netG.parameters())
    broadcast_params(netNoise.parameters())
    broadcast_params(netD.parameters())

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    
    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
    
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.schedule, eta_min=1e-5)
    # netG = nn.DataParallel(netG)
    
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.schedule, eta_min=1e-5)
    # netD = nn.DataParallel(netD)

    #ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    netNoise = nn.parallel.DistributedDataParallel(netNoise, device_ids=[gpu])
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])


    # Create log path
    exp = args.exp
    parent_dir = "./train_logs/{}".format(args.dataset)
    if args.perturb_percent > 0:
        parent_dir += f'_{int(args.perturb_percent)}p_{args.perturb_dataset}'
    parent_dir += f'/biasnoise/{args.phi1}_{args.phi2}/{args.image_distance}/tau_{args.tau}'
    exp_path = os.path.join(parent_dir, exp)
    os.makedirs(exp_path, exist_ok=True)
    image_batches_path = f'{exp_path}/image_batches'
    content_path = f'{exp_path}/content'
    netG_path = f'{exp_path}/netG'
    netD_path = f'{exp_path}/netD'
    os.makedirs(image_batches_path, exist_ok=True)
    os.makedirs(content_path, exist_ok=True)
    os.makedirs(netG_path, exist_ok=True)
    os.makedirs(netD_path, exist_ok=True)
    
    
    # Get Data
    # data_loader = get_dataloader(args)
    # ddp
    # dataset = get_dataset(args)
    if args.perturb_dataset == 'none':
        dataset = getCleanData(args.dataset, image_size=args.image_size)
    else:
        dataset = getMixedData(args.dataset, args.perturb_dataset, percentage = args.perturb_percent, image_size=args.image_size)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True)
    
    
    # Resume
    if args.resume:
        checkpoint_file = os.path.join(content_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    
    netNoise.load_state_dict(torch.load(args.bias_model))
    netNoise.eval()

    # Make log file
    with open(os.path.join(exp_path, 'log.txt'), 'w') as f:
        f.write("Start Training")
        f.write('\n')

    command_line = ' '.join(sys.argv)
    command_line = 'python3 ' + ' '.join(sys.argv)
    output_file = f"{exp_path}/command_history.txt"
    # Append the command line to the output file
    with open(output_file, "w") as file:
        file.write(command_line + "\n")
    
    
    # get phi star
    phi_star1 = select_phi(args.phi1)
    phi_star2 = select_phi(args.phi2)
    global get_image_distance
    get_image_distance = get_image_distance(args.image_distance)


    # Start training
    start = datetime.now()

    for epoch in range(init_epoch, args.num_epoch+1):
        if rank == 0:
            print(f'Epoch {epoch}')
        train_sampler.set_epoch(epoch)
        
        for _, x in tqdm(enumerate(data_loader), total=len(data_loader)):
            try: x,_ = x
            except: pass
            
            #### Update potential ####
            for p in netD.parameters():  
                p.requires_grad = True

            real_data = x.float().to(device, non_blocking=True)
            real_data.requires_grad = True
                
            netD.zero_grad()

            # real D loss
            noise = torch.randn_like(real_data)            
            D_real = netD(real_data)
            errD_real = phi_star2(-D_real)
            errD_real = errD_real.mean()
            errD_real.backward(retain_graph=True)
            
            # R1 regularization
            grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=real_data, create_graph=True)[0]
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = args.r1_gamma / 2 * grad_penalty
            grad_penalty.backward()

            # fake D loss
            latent_z = torch.randn(batch_size, nz, device=device)
            with torch.no_grad():
                noise = netNoise(noise, latent_z)
            x_0_predict = netG(noise, latent_z)
            D_fake = netD(x_0_predict)
            
            # import ipdb; ipdb.set_trace()
            errD_fake = phi_star1(D_fake - args.tau * torch.sum(get_image_distance(x_0_predict, noise), dim=1))
            errD_fake = errD_fake.mean()
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()


            #### Update Generator ####
            for p in netD.parameters():
                p.requires_grad = False
            
            
            netG.zero_grad()

            # Generator loss
            noise = torch.randn_like(real_data)
            latent_z = torch.randn(batch_size, nz, device=device)
            with torch.no_grad():
                noise = netNoise(noise, latent_z)
            x_0_predict = netG(noise, latent_z)
            D_fake = netD(x_0_predict)
            
            # err = args.tau * torch.sum(((x_0_predict-noise).view(noise.size(0), -1))**2, dim=1) - D_fake
            # err = args.tau * torch.sum(((x_0_predict-noise).view(noise.size(0), -1))**2, dim=1) - D_fake
            err = args.tau * torch.sum(get_image_distance(x_0_predict, noise), dim=1) - D_fake
            # err2 = args.tau * torch.sum(((x_0_predict-noise)**2).view(noise.size(0), -1), dim=1) - D_fake
            # print(err - err2)
            err = err.mean()
            err.backward()
            optimizerG.step()
            global_step += 1
            
            ## save losses
            if rank == 0:
                if global_step % args.print_every == 0:
                    with open(os.path.join(exp_path, 'log.txt'), 'a') as f:
                        str_content = f'Epoch {epoch:04d} : G Loss {err.item():.4f}, D Loss {errD.item():.4f}, Elapsed {datetime.now() - start}'
                        print(str_content)
                        f.write(str_content)
                        f.write('\n')
        

        schedulerG.step()
        schedulerD.step()

        if rank == 0:
            # save content
            if epoch % args.save_content_every == 0:
                print('Saving content.')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                            'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                            'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                            'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                
                torch.save(content, os.path.join(content_path, 'content.pth'))
                if epoch % args.save_content_epoch_every == 0:
                    torch.save(content, os.path.join(content_path, f'content_{epoch}.pth'))
            
            # save checkpoint
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                torch.save(netG.state_dict(), os.path.join(netG_path, 'netG_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                
                torch.save(netD.state_dict(), os.path.join(netD_path, 'netD_{}.pth'.format(epoch)))

            # save generated images
            if epoch % args.save_image_every == 0:
                noise = torch.randn_like(real_data)
                latent_z = torch.randn(batch_size, nz, device=device)
                with torch.no_grad():
                    noise = netNoise(noise, latent_z)
                images = netG(noise, latent_z)
                images = (0.5*(images+1)).detach().cpu()
                torchvision.utils.save_image(images, os.path.join(image_batches_path, 'epoch_{}.png'.format(epoch)))

def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup() 

def cleanup():
    dist.destroy_process_group() 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('UOT parameters')
    
    # Experiment description
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')
    parser.add_argument('--exp', default='linear', help='name of experiment')
    parser.add_argument('--resume', action='store_true',default=False, help='Resume training or not')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32, help='size of image')
    parser.add_argument('--num_channels', type=int, default=3, help='channel of image')
    parser.add_argument('--perturb_dataset', default='none', help='name of data added to the dataset')
    parser.add_argument('--perturb_percent', type=float, default=0, help='percentage of perturb_data')
    parser.add_argument('--bias_model', type=str, help='location of bias_model')
    
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
    
    # Training/Optimizer configurations
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=600, help='the number of epochs')
    parser.add_argument('--lr_g', type=float, default=1.6e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1.0e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--schedule', type=int, default=1800, help='cosine scheduler, learning rate 1e-5 until {schedule} epoch')
    parser.add_argument('--use_ema', action='store_false', default=True, help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    # Loss configurations
    parser.add_argument('--phi1', type=str, default='kl', choices=['linear', 'kl', 'softplus', 'chi'])
    parser.add_argument('--phi2', type=str, default='kl', choices=['linear', 'kl', 'softplus', 'chi'])
    parser.add_argument('--tau', type=float, default=0.001, help='proportion of the cost c')
    parser.add_argument('--r1_gamma', type=float, default=0.2, help='coef for r1 reg')
    parser.add_argument('--image_distance', type=str, default='l2', help='type of loss for image')
        
    # Visualize/Save configurations
    parser.add_argument('--print_every', type=int, default=100, help='print current loss for every x iterations')
    parser.add_argument('--save_content_every', type=int, default=20, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=20, help='save ckpt every x epochs')
    parser.add_argument('--save_image_every', type=int, default=10, help='save images every x epochs')
    parser.add_argument('--save_content_epoch_every', type=int, default=50, help='save content of an epoch for resuming every x epochs')

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6020',
                        help='master port number')
    
    args = parser.parse_args()
    
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        print('starting in debug mode')
        
        init_processes(0, size, train, args)
