import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--img_dir', default='',
                        help='path where to save')
    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='./weights/best_mae.pth',
                        help='path where the trained weights saved')
    parser.add_argument('--type', default='RGB',
                        help='path where to save')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    parser.add_argument('--type', default='RGB',
                        help='path where to save')
    return parser
import glob
def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    npy_path= args.img_dir+"/train/*%s.jpg"%args.type
    gt_list =sorted( glob.glob(npy_path))  # change to npy for gt_list
    for img_path in gt_list:
    # set your image path here
        # load the images
        img_raw = Image.open(img_path).convert('RGB')
        # round the size
        width, height = img_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        # pre-proccessing
        img = transform(img_raw)

        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)
        # run inference
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        
        outputs_points = outputs['pred_points'][0]

        # threshold = 0.75
        # # filter the predictions
        # points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        # imgid=img_path.split('/')[-1].replace("_T.jpg","")
        # print(imgid,",",float(len(points)))

        threshold = 0.7
        # filter the predictions
        addata=[]
        for td in [0.6,0.64,0.68,0.73,0.75]:

            points = outputs_points[outputs_scores > td].detach().cpu().numpy().tolist()
            addata.append(float(len(points)))
        imgid=img_path.split('/')[-1].replace("_%s.jpg"%args.type,"")
        print("%s,%d,%d,%d,%d,%d"%(imgid,addata[0],addata[1],addata[2],addata[3],addata[4]))
        # predict_cnt = int((outputs_scores > threshold).sum())

        # outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        # outputs_points = outputs['pred_points'][0]
        # # draw the predictions
        # size = 2
        # img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        # for p in points:
        #     img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        # # save the visualized image
        # cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)