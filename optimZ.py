from PIL import Image
import argparse
import os
import numpy as np
import torch
# from datasets import MaskFaceDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from glob import glob
import pdb
import random
from threeDgan import Generator, Discriminator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--generator',
                         type=str,
                         help='Pretrained generator',
                         default='E:/PyCharmProject/GAN-inpainting/models/gen_28000_TA.pt' )
    parser.add_argument( '--discriminator',
                         type=str,
                         help='Pretrained discriminator',
                         default='E:/PyCharmProject/GAN-inpainting/models/dis_28000_TA.pt' )
    parser.add_argument( '--imgSize',
                         type=int,
                         default=64 )
    parser.add_argument( '--batch_size',
                         type=int,
                         default=64 )
    parser.add_argument( '--n_size',
                         type=int,
                         default=7,
                         help='size of neighborhood' )
    parser.add_argument( '--blend',
                         action='store_true',
                         default=True,
                         help="Blend predicted image to original image" )
    # These files are already on the VC server. Not sure if students have access to them yet.
    parser.add_argument( '--mask_csv',
                         type=str,
                         default='E:/celebA/mask.csv',
                         help='path to the masked csv file' )
    parser.add_argument( '--mask_root',
                         type=str,
                         default='E:/celebA',
                         help='path to the masked root' )
    parser.add_argument( '--per_iter_step',
                         type=int,
                         default=1500,
                         help='number of steps per iteration' )
    parser.add_argument('--pic',
                        type=str,
                        default='./selfie.jpg',
                        help='path to the input image')
    args = parser.parse_args()
    return args

def saveimages( corrupted, completed, blended, index ):
    os.makedirs( 'completion', exist_ok=True )
    save_image( corrupted,
                'completion/%d_corrupted.png' % index,
                nrow=corrupted.shape[ 0 ] // 5,
                normalize=True )
    save_image( completed,
                'completion/%d_completed.png' % index,
                nrow=completed.shape[ 0 ] // 5,
                normalize=True )
    save_image( blended,
                'completion/%d_blended.png' % index,
                nrow=corrupted.shape[ 0 ] // 5,
                normalize=True )

def main():
    args = parse_args()

    if args.pic == "":
        # Configure data loader
        celebA_dataset = MaskFaceDataset( args.mask_csv,
                                          args.mask_root,
                                          transform=transforms.Compose( [
                               transforms.Resize( args.imgSize ),
                               transforms.ToTensor(),
                               transforms.Normalize( ( 0.5, 0.5, 0.5 ), ( 0.5, 0.5, 0.5 ) )
                           ] ) )
        dataloader = torch.utils.data.DataLoader( celebA_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False )
        m = ModelInpaint( args )
        for i, ( imgs, masks ) in enumerate( dataloader ):
            # pdb.set_trace()
            print(imgs.shape)
            masks = np.stack( ( masks, ) * 3, axis=1 )
            corrupted = imgs * torch.tensor( masks )
            completed, blended = m.inpaint( corrupted, masks )
            print(masks.size)
            saveimages( corrupted, completed, blended, i )
    else:
        image = Image.open(args.pic)
        resize = transforms.Resize(args.imgSize)
        toTensor = transforms.ToTensor()
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = normalize(toTensor(resize(image)))

        maskSize = (15, 15) # (w, h)
        mask = torch.ones(image.shape)
        start_w = random.randrange(0, image.shape[2] - maskSize[0] - 1)
        start_h = random.randrange(0, image.shape[1] - maskSize[1] - 1)
        mask[:, start_h: start_h+maskSize[1], start_w: start_w+maskSize[0]] = 0
        corrupted = image * mask
        args.batch_size = 1
        m = ModelInpaint(args)
        corrupted = torch.unsqueeze(corrupted, 0)
        print(corrupted.shape)
        mask = torch.unsqueeze(mask, 0)
        mask = mask.numpy()
        completed, blended = m.inpaint(corrupted, mask)
        saveimages(corrupted, completed, blended, 0)

if __name__ == '__main__':
    main()
