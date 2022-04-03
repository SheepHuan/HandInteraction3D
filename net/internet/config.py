# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import sys
import math
import numpy as np


class Config:
    ## dataset
    dataset = 'InterHand2.6M'  # InterHand2.6M, RHD, STB
    dataset_root_path = 'InterHand2.6M'
    ## input, output
    input_img_shape = (256, 256)
    output_hm_shape = (64, 64, 64)  # (depth, height, width)
    sigma = 2.5
    bbox_3d_size = 400  # depth axis
    bbox_3d_size_root = 400  # depth axis
    output_root_hm_shape = 64  # depth axis
    joint_nums = 21

    ## model
    resnet_type = 50  # 18, 34, 50, 101, 152

    ## training config
    lr_dec_epoch = [10,20,35,50,70,100] if dataset == 'InterHand2.6M' else [45, 47]
    end_epoch = 231 if dataset == 'InterHand2.6M' else 50
    lr = 0.001
    lr_dec_factor = 2
    train_batch_size = 6

    ## testing config
    test_batch_size = 64
    trans_test = 'rootnet'  # gt, rootnet

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '../..')
    data_dir = osp.join(root_dir, 'data')
    # output_dir = osp.join(root_dir, '')
    model_dir = osp.join(root_dir, 'weights/internet')
    # vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(root_dir, 'log')
    # result_dir = osp.join(output_dir, 'result')

    ## others
    num_thread = 8
    gpu_ids = '0,1,2,3'
    num_gpus = 4
    continue_train = True

    pretrained_weight='weights/internet/snapshot_90.pdparams'

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))


cfg = Config()

# sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
# from utils.dir import add_pypath, make_folder
#
# add_pypath(osp.join(cfg.data_dir))
# add_pypath(osp.join(cfg.data_dir, cfg.dataset))
# make_folder(cfg.model_dir)
# make_folder(cfg.vis_dir)
# make_folder(cfg.log_dir)
# make_folder(cfg.result_dir)



