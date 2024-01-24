import torch
import torch.optim
import numpy as np
import argparse
import dcam_model
import torch.nn as nn
import os
import cv2

def mkdir_ifnotexist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

fpath='./'
ap='./alpha646/'
mkdir_ifnotexist(ap)

if __name__ == '__main__':
    segmodel = dcam_model.D646AUTO()
    segmodel.load_state_dict(torch.load('d646.ckpt',map_location='cpu'))
    segmodel.eval()
    segmodel=segmodel.cpu()
    files=os.listdir(fpath)
    for file in files:
        print(file)
        im=cv2.imread(fpath+file)
        h,w,c=im.shape
        nonph,nonpw,_=im.shape
        newh= (((h-1)//32)+1)*32
        neww= (((w-1)//32)+1)*32
        padh=newh-h
        padh1=int(padh/2)
        padh2=padh-padh1
        padw=neww-w
        padw1=int(padw/2)
        padw2=padw-padw1
        rawimg_pad=cv2.copyMakeBorder(im,padh1,padh2,padw1,padw2,cv2.BORDER_REFLECT)
        im=rawimg_pad[:,:,::-1]
        im=im/255.
        im = np.transpose(im, [2, 0, 1])
        im=im[np.newaxis,:,:,:].astype(np.float32)
        im=torch.from_numpy(im).cpu()
        Tm = im

        import time
        a=time.time()
        with torch.no_grad():
            seg,o1,o2, trio ,alpha= segmodel(im)
            alpha=torch.clamp(alpha,0.,1.)
            trio=torch.argmax(trio,dim=1,keepdim=True)
            alpha=alpha*(trio==1).float()+(trio==2).float()
            alpha=alpha.cpu().numpy()
            alpha=alpha[0,0,:,:]
        alpha=np.array(alpha*255,dtype=np.uint8)
        alpha=alpha[padh1:padh1+nonph,padw1:padw1+nonpw]
        cv2.imwrite(ap+file[:-3]+'png',alpha)
