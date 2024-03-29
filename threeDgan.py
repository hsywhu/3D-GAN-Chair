import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from datasets import ChairDataset
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy

parser = argparse.ArgumentParser()
parser.add_argument( '--n_epochs',
                     type=int,
                     default=200,
                     help='number of epochs of training' )
parser.add_argument( '--batch_size',
                     type=int,
                     default=80,
                     help='size of the batches' )
parser.add_argument( '--lr_G',
                     type=float,
                     default=0.0025,
                     help='adam: learning rate' )
parser.add_argument( '--lr_D',
                     type=float,
                     default=0.0001,
                     help='adam: learning rate' )
parser.add_argument( '--b1',
                     type=float,
                     default=0.5,
                     help='adam: decay of first order momentum of gradient' )
parser.add_argument( '--b2',
                     type=float,
                     default=0.999,
                     help='adam: decay of first order momentum of gradient' )
parser.add_argument( '--n_cpu',
                     type=int,
                     default=8,
                     help='number of cpu threads to use during batch generation' )
parser.add_argument( '--latent_dim',
                     type=int,
                     default=200,
                     help='dimensionality of the latent space' )
parser.add_argument( '--sample_interval',
                     type=int,
                     default=400,
                     help='interval between image sampling' )
parser.add_argument( '--train_csv',
                     type=str,
                     default='./dataset/3d/train.csv',
                     help='path to the training csv file' )
parser.add_argument( '--train_root',
                     type=str,
                     default='./dataset/chairs/',
                     help='path to the training root' )
parser.add_argument( '--unrolled_steps',
                     type=int,
                     default=10,
                     help='how many iteration for every unrolled step' )
opt = parser.parse_args()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # ConvTranspose3d(in_channels, out_channels, kernel_size,
        #                 stride=1, padding=0, output_padding=0, 
        #                 groups=1, bias=True, dilation=1)
        self.deconv1 = nn.ConvTranspose3d(opt.latent_dim, 512, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm3d(512)
        self.deconv2 = nn.ConvTranspose3d(512, 256, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm3d(256)
        self.deconv3 = nn.ConvTranspose3d(256, 128, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm3d(128)
        self.deconv4 = nn.ConvTranspose3d(128, 64, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm3d(64)
        self.deconv5 = nn.ConvTranspose3d(64, 1, 4, 2, 1)

    # weight_init
    def weight_init( self, mean, std ):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = input.view(-1, opt.latent_dim, 1, 1, 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.sigmoid(self.deconv5(x))
        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, 4, 2, 1)
        self.conv2 = nn.Conv3d(64, 128, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 512, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm3d(512)
        self.conv5 = nn.Conv3d(512, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        input = input.view(-1, 1, 64, 64, 64)
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))
        return x

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def main():
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    # Initialize weights
    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)
    # Configure data loader
    chair_dataset = ChairDataset(opt.train_root, 6778)
    dataloader = torch.utils.data.DataLoader( chair_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=True )
    # Optimizers
    optimizer_G = torch.optim.Adam( generator.parameters(),
                                    lr=opt.lr_G,
                                    betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam( discriminator.parameters(),
                                    lr=opt.lr_D,
                                    betas=(opt.b1, opt.b2))
    # ----------
    #  Training
    # ----------
    os.makedirs('models', exist_ok=True)
    os.makedirs('mesh', exist_ok=True)
    d_acc_last = 0

    for epoch in range(opt.n_epochs):
        # learning rate decay
        # if (epoch + 1) == 11:
        #     optimizer_G.param_groups[0]['lr'] /= 10
        #     optimizer_D.param_groups[0]['lr'] /= 10
        #     print('learning rate change!')
        # if (epoch + 1) == 16:
        #     optimizer_G.param_groups[0]['lr'] /= 10
        #     optimizer_D.param_groups[0]['lr'] /= 10
        #     print('learning rate change!')
        for i, mesh in enumerate(dataloader):
            # Adversarial ground truths
            valid = Variable(Tensor(mesh.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(mesh.shape[0], 1).fill_(0.0), requires_grad=False)
            # Configure input
            mesh = mesh.unsqueeze(1)
            real_mesh = Variable(mesh.type(Tensor))
            if cuda:
                valid = valid.cuda()
                fake = fake.cuda()
                real_mesh = real_mesh.cuda()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            label_real = discriminator(real_mesh)
            z = Variable(Tensor(np.random.normal(0, 1, (mesh.shape[0], opt.latent_dim))))
            if cuda:
                z = z.cuda()
            with torch.no_grad():
                gen_mesh = generator(z)
            label_gen = discriminator(gen_mesh.detach())
            real_loss = adversarial_loss(label_real, valid)
            fake_loss = adversarial_loss(label_gen, fake)
            d_loss = (real_loss + fake_loss) / 2
            real_acc = (label_real > 0.5).float().sum() / real_mesh.shape[ 0 ]
            gen_acc = (label_gen < 0.5).float().sum() / gen_mesh.shape[ 0 ]
            d_acc = (real_acc + gen_acc) / 2
            if d_acc_last < 0.8:
                # print("d_loss_step")
                d_loss.backward()
                optimizer_D.step()
            d_acc_last = d_acc

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            if opt.unrolled_steps > 0:
                backup = copy.deepcopy(discriminator.state_dict())
                for unrolled_step in range(opt.unrolled_steps):
                    optimizer_D.zero_grad()

                    # train D on real
                    d_real_decision = discriminator(real_mesh)
                    d_real_error = adversarial_loss(d_real_decision, valid)

                    # train D on fake
                    z = Variable(Tensor(np.random.normal(0, 1, (mesh.shape[0], opt.latent_dim))))
                    if cuda:
                        z = z.cuda()
                    with torch.no_grad():
                        d_fake_data = generator(z)
                    d_fake_decision = discriminator(d_fake_data)
                    d_fake_error = adversarial_loss(d_fake_decision, fake)

                    d_loss = d_real_error + d_fake_error
                    d_loss.backward(create_graph = True)
                    optimizer_D.step()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (mesh.shape[0], opt.latent_dim))))
            if cuda:
                z = z.cuda()
            g_fake_data = generator(z)
            d_g_fake_decision = discriminator(g_fake_data)
            g_loss = adversarial_loss(d_g_fake_decision, valid)
            g_loss.backward()
            optimizer_G.step()

            if opt.unrolled_steps > 0:
                discriminator.load_state_dict(backup)
                del backup

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % \
                    (epoch,
                     opt.n_epochs,
                     i,
                     len(dataloader),
                     d_loss.item(),
                     d_acc * 100,
                     g_loss.item()))
            batches_done = epoch * len(dataloader) + i
            print("batches_done: ", batches_done)
            if batches_done % opt.sample_interval == 0:
                np.save('mesh/%d.npy' % batches_done, gen_mesh.detach().cpu())
                torch.save( generator, 'models/gen_%d.pt' % batches_done )
                torch.save( discriminator, 'models/dis_%d.pt' % batches_done )
if __name__ == '__main__':
    main()
