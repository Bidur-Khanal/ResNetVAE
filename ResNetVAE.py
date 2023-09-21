import os
import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from modules import *
from sklearn.model_selection import train_test_split
import pickle
import argparse
import utils 
from utils import custom_dermnet_faster, custom_FETAL_PLANE_faster, custom_COVID19_Xray_faster, custom_histopathology_faster, MURA_faster
import neptune as neptune
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from neptune.types import File

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)



def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def loss_function(recon_x, x, mu, logvar, args):
    #MSE = F.mse_loss(recon_x, x, reduction='mean')
    MSE = F.binary_cross_entropy(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + args.beta * KLD, MSE, KLD


def train(log_interval, model, device, train_loader, optimizer, epoch, args):
    # set model as training mode
    model.train()

    losses = []
    mse_losses = []
    kld_losses = []
    all_y, all_z, all_mu, all_logvar = [], [], [], []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)

        optimizer.zero_grad()
        X_reconst, z, mu, logvar = model(X)  # VAE
        loss, mse, kld = loss_function(X_reconst, X, mu, logvar, args)
        losses.append(loss.item())
        mse_losses.append(mse.item())
        kld_losses.append(kld.item())

        loss.backward()
        optimizer.step()

        print (X.shape,X_reconst.shape)

        # all_y.extend(y.data.cpu().numpy())
        # all_z.extend(z.data.cpu().numpy())
        # all_mu.extend(mu.data.cpu().numpy())
        # all_logvar.extend(logvar.data.cpu().numpy())

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))

    # all_y = np.stack(all_y, axis=0)
    # all_z = np.stack(all_z, axis=0)
    # all_mu = np.stack(all_mu, axis=0)
    # all_logvar = np.stack(all_logvar, axis=0)

    # save Pytorch models of best record
    # torch.save(model.state_dict(), os.path.join(save_model_path, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    # torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    # print("Epoch {} model saved!".format(epoch + 1))

    return X_reconst.data.cpu().numpy(), all_y, all_z, all_mu, all_logvar, np.mean(losses), np.mean(mse_losses), np.mean(kld_losses)


def validation(model, device, optimizer, test_loader, inverse_normalize, args):
    # set model as testing mode
    model.eval()

    test_loss = 0
    test_mse = 0.
    test_kld = 0.
    all_y, all_z, all_mu, all_logvar = [], [], [], []
    X_orgs = []
    X_recons = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            X_reconst, z, mu, logvar = model(X)

            loss, mse, kld = loss_function(X_reconst, X, mu, logvar, args)
            test_loss += loss.item()  # sum up batch loss
            test_mse += mse.item()
            test_kld += kld.item()

            # all_y.extend(y.data.cpu().numpy())
            # all_z.extend(z.data.cpu().numpy())
            # all_mu.extend(mu.data.cpu().numpy())
            # all_logvar.extend(logvar.data.cpu().numpy())

            X_recons.extend(inverse_normalize(X_reconst).data.cpu()[0:2])
            X_orgs.extend(inverse_normalize(X).data.cpu()[0:2])

    test_loss /= len(test_loader)
    test_mse /= len(test_loader)
    test_kld /= len(test_loader)
    # all_y = np.stack(all_y, axis=0)
    # all_z = np.stack(all_z, axis=0)
    # all_mu = np.stack(all_mu, axis=0)
    # all_logvar = np.stack(all_logvar, axis=0)

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}\n'.format(len(test_loader.dataset), test_loss))
    return X_reconst.data.cpu().numpy(), all_y, all_z, all_mu, all_logvar, test_loss, torch.stack(X_orgs), torch.stack(X_recons), test_mse, test_kld



class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
    

def main(args):

    
    # Preprocessings the data
    print('==> Preparing data..')
    if args.dataset == "covid19_xray":

        NUM_CLASSES = 3
        NUM_CHANNELS = 3

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        
        invnormalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        train_transform =  transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAdjustSharpness(3),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
                ])

            

        # load the data
        train_dataset = custom_COVID19_Xray_faster(root = args.root, train=True, transform = train_transform)
        test_dataset = custom_COVID19_Xray_faster(root = args.root,train = False, transform= val_transform)


        # encapsulate data into dataloader form
        trainloader = FastDataLoader(train_dataset,batch_size=args.batch_size, shuffle = True, num_workers=2, persistent_workers= True)
        testloader = FastDataLoader(test_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)


    elif args.dataset == "histopathology":

        NUM_CLASSES = 9
        NUM_CHANNELS = 3

        train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.RandomAdjustSharpness(3),
                    transforms.RandomAutocontrast(),
                    transforms.RandomEqualize(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
            
        invnormalize = NormalizeInverse(mean=[.5], std=[.5])
        
        # load the data
        train_dataset = custom_histopathology_faster(root = args.root,train=True, transform = train_transform)
        test_dataset = custom_histopathology_faster(root = args.root,train = False, transform= val_transform)



        # encapsulate data into dataloader form
        trainloader = FastDataLoader(train_dataset,batch_size=args.batch_size, shuffle = True, num_workers=2, persistent_workers= True)
        testloader = FastDataLoader(test_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)


    elif args.dataset == "fetal":

        NUM_CLASSES = 6
        NUM_CHANNELS = 3

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        train_transform =  transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAdjustSharpness(3),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
                ])
            
        invnormalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        # load the data
        train_dataset = custom_FETAL_PLANE_faster(root = args.root,train=True, transform = train_transform)
        test_dataset = custom_FETAL_PLANE_faster(root = args.root,train = False, transform= val_transform)


        # encapsulate data into dataloader form
        trainloader = FastDataLoader(train_dataset,batch_size=args.batch_size, shuffle = True, num_workers=2, persistent_workers= True)
        testloader = FastDataLoader(test_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)


    elif args.dataset == "dermnet":

        NUM_CLASSES = 23
        NUM_CHANNELS = 3

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        train_transform =  transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAdjustSharpness(3),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
                ])
        
        invnormalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        # load the data
        train_dataset = custom_dermnet_faster(root = args.root,train=True, transform = train_transform)
        test_dataset = custom_dermnet_faster(root = args.root,train = False, transform= val_transform)


        # encapsulate data into dataloader form
        trainloader = FastDataLoader(train_dataset,batch_size=args.batch_size, shuffle = True, num_workers=2, persistent_workers= True)
        testloader = FastDataLoader(test_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)

    elif args.dataset == "mura":

        NUM_CLASSES = 7
        NUM_CHANNELS = 3

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        train_transform =  transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAdjustSharpness(3),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
                ])
        
        invnormalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            

        # load the data
        train_dataset = MURA_faster(root = args.root,train=True, transform = train_transform)
        test_dataset = MURA_faster(root = args.root,train = False, transform= val_transform)


        # encapsulate data into dataloader form
        trainloader = FastDataLoader(train_dataset,batch_size=args.batch_size, shuffle = True, num_workers=2, persistent_workers= True)
        testloader = FastDataLoader(test_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)



    # EncoderCNN architecture
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
    CNN_embed_dim = 256     # latent dim extracted by 2D CNN
    res_size = 224        # ResNet image size
    dropout_p = 0.2       # dropout probability


    # training parameters
    epochs = args.epochs        # training epochs
    learning_rate = args.lr
    log_interval = args.print_freq

    
    # Create model
    resnet_vae = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)

    print(resnet_vae)
    model_params = list(resnet_vae.parameters())
    optimizer = torch.optim.Adam(model_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


    # record training process
    epoch_train_losses = []
    epoch_test_losses = []

    best_loss = 100000.
    
    
    # start training
    for epoch in range(epochs):
        # train, test model
        X_reconst_train, y_train, z_train, mu_train, logvar_train, train_losses, train_mse_loss, train_kld_loss = train(log_interval, resnet_vae, device, trainloader, optimizer, epoch, args)
        X_reconst_test, y_test, z_test, mu_test, logvar_test, epoch_test_loss, X_org, X_recons, test_mse_loss, test_kld_loss = validation(resnet_vae, device, optimizer, testloader,invnormalize, args)
        scheduler.step()

        # save results
        epoch_train_losses.append(train_losses)
        epoch_test_losses.append(epoch_test_loss)

        

        # save all train test results
        # A = np.array(epoch_train_losses)
        # C = np.array(epoch_test_losses)
        
        # np.save(os.path.join(save_model_path, 'ResNet_VAE_training_loss.npy'), A)
        # np.save(os.path.join(save_model_path, 'y_cifar10_train_epoch{}.npy'.format(epoch + 1)), y_train)
        # np.save(os.path.join(save_model_path, 'z_cifar10_train_epoch{}.npy'.format(epoch + 1)), z_train)

        
       
        if epoch_test_loss < best_loss:
            ### save model checkpoints
            best_checkpoint_name = "ResNetVAE_"+'lr_'+str(args.lr)+'_batch_size_'+str(args.batch_size)+'_version_'+args.version+'_beta_'+str(args.beta)+'_mean_loss_best_checkpoint.pth.tar'
            if not os.path.exists(os.path.join(args.save_dir,args.dataset)):
                os.makedirs(os.path.join(os.path.join(args.save_dir,args.dataset)))

            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': resnet_vae.resnet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(args.save_dir,args.dataset,best_checkpoint_name))


        if ((epoch+1)%50) == 0:

            ### save model checkpoints
            checkpoint_name = "ResNetVAE_"+'lr_'+str(args.lr)+'_batch_size_'+str(args.batch_size)+'_epoch_'+str(epoch)+'_version_'+args.version+'_beta_'+str(args.beta)+'_mean_loss_checkpoint.pth.tar'
            if not os.path.exists(os.path.join(args.save_dir,args.dataset)):
                os.makedirs(os.path.join(os.path.join(args.save_dir,args.dataset)))

            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': resnet_vae.resnet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(args.save_dir,args.dataset,checkpoint_name))


        run["train/loss"].log(train_losses)
        run["test/loss"].log(epoch_test_loss)
        run["train/mse_loss"].log(train_mse_loss)
        run["test/mse_loss"].log(test_mse_loss)
        run["train/kld_loss"].log(train_kld_loss)
        run["test/kld_loss"].log(test_kld_loss)
        n_row = int(np.sqrt(X_org.shape[0]))

        
        org_img = torchvision.utils.make_grid(X_org, nrow=n_row)
        recons_img = torchvision.utils.make_grid(X_recons, nrow=n_row)
        fig, ax = plt.subplots()
        plt.imshow(org_img.permute(1, 2, 0))
        run["test/org"].log(fig)
        plt.close()

        fig, ax = plt.subplots()
        plt.imshow(recons_img.permute(1, 2, 0))
        run["test/reconstruction"].log(fig)
        plt.close()
    

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



if __name__ == "__main__":

    run = neptune.init_run(
    project="bidur/ResNetVAE",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxM2NkY2I5MC01OGUzLTQzZWEtODYzYi01YTZiYmFjZmM4NmIifQ==",
)  # your credentials
    

    parser = argparse.ArgumentParser(description='PyTorch CNN Training')
    parser.add_argument('--root', metavar='DIR', default='data/',
                        help='path to dataset (default: data)')


    parser.add_argument('--save_dir', default='checkpoints', type= str)
    parser.add_argument('-a', '--arch', type=str, default='resnet18',
                        help='model architecture you want ti use')
    parser.add_argument('--dataset', default = "fetal", type = str, choices = ["covid19_xray","histopathology","fetal","dermnet","mura"])
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default= 50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=5, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for various purposes (model initialization, subset selection). Here it is used for subset selection')
    parser.add_argument('--use_gpu', default=True, type= bool,
                        help='use the default GPU')
    parser.add_argument('--gpu_id', type =str , default= '0',
                        help='select a gpu id')
    parser.add_argument('--version', default = "1", type = str)
    parser.add_argument('--beta', default=0.1, type=float, metavar='M',
                        help='beta term that weights KLD loss term')

        

    args = parser.parse_args()
    params = vars(args)
    run["parameters"] = params
    run["all files"].upload_files("*.py")
        
    if args.use_gpu:
        device = 'cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu'

    main(args)