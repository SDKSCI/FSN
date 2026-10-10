import argparse
import os

import numpy as np
import torch.backends.cudnn as cudnn
import scipy.io as io
import models
import torchvision.transforms as transforms
import flow_transforms
from imageio import imread
from imageio import imsave
from tqdm import tqdm
from loss import *
import time
import random
import math
from datetime import datetime
from En import merge_superpxiel





@torch.no_grad()
def test(args, model, img_paths, save_path, idx,num):
      # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    img_file = img_paths[idx]
    load_path = img_file
    imgId = os.path.basename(img_file)[:-4]




    img_ = imread(load_path)
    # RCF_mask=RCF_get.RCF_bd(img_)
    # img_  = cv2.GaussianBlur(img_ , (3, 3), 0)


    tic = time.time()
    H, W, _ = img_.shape

    scale = math.sqrt((1*num) / (np.ceil(H / 16.) * np.ceil(W / 16.)))
    H_, W_ = int(np.ceil(H *scale/ 16.)*16 ), int(np.ceil(W *scale/ 16.)*16)
    img = cv2.resize(img_, (int( W_), int(H_)), interpolation=cv2.INTER_CUBIC)

      # get spixel id
    n_spixl_h = int(np.ceil(H_ / args.downsize))
    n_spixl_w = int(np.ceil( W_ / args.downsize))
    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor = np.repeat(
      np.repeat(spix_idx_tensor_, args.downsize, axis=1), args.downsize, axis=2)
    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()
    n_spixel = int(n_spixl_h * n_spixl_w)


    img1 = input_transform(img)

    ori_img = input_transform(img_)

    # compute output
    tic = time.time()
    output = model(img1.cuda().unsqueeze(0))


    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)
    #print(curr_spixl_map.shape)
    # The orignal sz of nyu test set 448*608
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H, W), mode='nearest').type(torch.int)

    label_np = ori_sz_spixel_map.squeeze().cpu().numpy().astype(np.int32).ravel()
    img_rgb = cv2.cvtColor(img_.astype(np.uint8), cv2.COLOR_BGR2RGB)  # 确保 RGB
    merged_label_map = merge_superpxiel.getsuperpixle(img_rgb, label_np, n_spixel)
    merged_label_map = merged_label_map.reshape(H, W)
    # ✅ ========================================

    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.dtype).view(3, 1, 1)
    spixel_viz, spixel_label_map = get_spixel_image(
      (ori_img + mean_values).clamp(0, 1),
      torch.from_numpy(merged_label_map).int(),  # ✅ 用合并后的标签
      n_spixels=n_spixel,
      b_enforce_connect=False
    )


    # mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1)
    #
    # spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(),
    #                                                   n_spixels= n_spixel, b_enforce_connect=True)


    toc = time.time() - tic
    if not os.path.isdir(os.path.join(save_path, 'Get_image')):
      os.makedirs(os.path.join(save_path, 'Get_image'))
    spixl_save_name = os.path.join(save_path, 'Get_image', imgId + '.jpg')
    imsave(spixl_save_name, (spixel_viz.transpose(1, 2, 0)).astype(np.uint8))
    #
    # save the unique maps as csv for eval
    if not os.path.isdir(os.path.join(save_path, 'Label')):
      os.makedirs(os.path.join(save_path, 'Label'))
    output_path = os.path.join(save_path, 'Label', imgId + '.mat')
    io.savemat(output_path, {'labels': (spixel_label_map + 1).astype(int)}, do_compression=True)

    return toc

def main():
    global args, save_path
    data_dir = args.data_dir
    # print("=> fetching img pairs in '{}'".format(data_dir))

    input_img_height = args.input_img_height
    input_img_width = args.input_img_width
    #print( input_img_height, input_img_width)
    mean_time_list = []
    # for scale in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
    for scale in range(1200,1500,300):

    # for scale in [0.7]:
    #     assert (input_img_height * scale % 16 == 0 and input_img_width * scale % 16 == 0)
        # num = round(int(input_img_height / 16 * scale * input_img_width / 16 * scale) / 100) * 100

        save_path =os.path.join(args.output,"FSN", str(scale))


        # print(args.timeoutput)
        if not os.path.isdir(args.timeoutput):
            os.makedirs(args.timeoutput)

        # print('=> will save everything to {}'.format(save_path))
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        tst_lst = []
        with open(args.test_list, 'r') as tf:
            img_path = tf.readlines()
            for path in img_path:
                # print(os.path.basename(path))
                file_extension = os.path.basename(path).split('.')[-1]
                file_extension = file_extension.strip('\n')
                # if file_extension == "jpg" and os.path.basename(path).split('.')[0] == "224":
                if file_extension == "jpg" or file_extension == "png":
                    tst_lst.append(path[:-1])
                else:

                    continue
        # print("=> fetching img pairs in '{}' {} samples found, num is {}".format(data_dir,len(tst_lst), num))


        # create model
        network_data = torch.load(args.pretrained)
        # print("=> using pre-trained model '{}'".format(network_data['arch']))
        model = models.__dict__[network_data['arch']]( data = network_data).cuda()
        model.eval()
        args.arch = network_data['arch']
        cudnn.benchmark = True


        mean_time = 0
        # tst_lst = tst_lst[::10]
        for n in tqdm(range(len(tst_lst)),desc=f"输入路径：{data_dir},输出路径：{save_path}"):

            time = test(args, model, tst_lst, save_path, n,scale)
            mean_time += time
        mean_time /= len(tst_lst)
        mean_time_list.append(round(mean_time, 3))


    import csv
    with open(args.timeoutput + r'\Time.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(["FSN"] + mean_time_list)


if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__"))

    parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of imgs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', metavar='DIR', default=r"G:\Data\NYUv2\Clean\images",
                        help='path to images folder')
    parser.add_argument('--output', metavar='DIR', default=r'G:\Result\demo', help='path to output folder')
    parser.add_argument('--timeoutput', metavar='DIR', default=r'G:\Result\demo', help='path to output folder')

    parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model',
                        default=r"/root/Code/FSN/weights/BSD500/FSNet_adam_3000000epochs_epochSize6000_b8_lr0.001_posW0.003_25_10_22_10_00/model_best.tar")

    parser.add_argument('--downsize', default=16, type=float,
                        help='superpixel grid cell, must be same as training setting')

    parser.add_argument('-nw', '--num_threads', default=1, type=int, help='num_threads')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')

    # nyu only has one type
    parser.add_argument('--input_img_height', '-v_imgH', default=480, type=int, help='img height_must be 16*n')
    parser.add_argument('--input_img_width', '-v_imgW', default=640, type=int, help='img width must be 16*n')

    args = parser.parse_args()

    for data in ["VOC2012"]:
        inpath = os.path.join(r"/root/Data", data)
        outpath = os.path.join(r"/root/Result", data)
        file_list = sorted(os.listdir(inpath))
        for file in file_list:
            if file == "Label" or file == "groundturth" or file == "instance" or file == "semantic" or file == "semantic_rgb" or file == "groundturth" or file == "GA" or file == "GM" or file == "SP" or file == "SegmentationClass":
                # if file == "Label" or file == "groundturth" or file == "GA" or file == "GM" or file == "SP":
                print(1)
                continue

            sub_file_list = sorted(os.listdir(os.path.join(inpath, file)))
            sub_file_list = [f for f in sub_file_list if f != '.ipynb_checkpoints']
            for sub_file in sub_file_list:
                file_path = os.path.join(os.path.join(inpath, file), sub_file)
                outputdir = os.path.join(os.path.join(outpath, file), sub_file)

                timeputdir = os.path.join(os.path.join(outpath, file), sub_file, "FSN")

                args = parser.parse_args()
                args.data_dir = file_path
                args.output = outputdir
                args.timeoutput = timeputdir
                print("目录：", args.data_dir, args.output, args.timeoutput)
                args.test_list = args.data_dir + '/test.txt'
                random.seed(100)
                main()

