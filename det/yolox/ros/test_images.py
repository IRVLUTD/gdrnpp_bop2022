#!/usr/bin/env python

"""Test a YOLO on images"""

import tf
import rosnode
import message_filters
import threading
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import time
import rospy

from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from geometry_msgs.msg import PoseStamped

import logging
from loguru import logger as loguru_logger
import os.path as osp
from setproctitle import setproctitle
from detectron2.engine import (
    default_argument_parser,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.config import LazyConfig, instantiate

import cv2
import torch

cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../"))

from omegaconf import OmegaConf
from lib.utils.time_utils import get_time_str
import core.utils.my_comm as comm
from core.utils.my_checkpoint import MyCheckpointer
from det.yolox.engine.yolox_setup import default_yolox_setup
from det.yolox.engine.yolox_trainer import YOLOX_DefaultTrainer
from det.yolox.engine.yolox_inference import yolox_inference_on_image
from det.yolox.utils import fuse_model
from det.yolox.data.datasets.dataset_factory import register_datasets_in_cfg


try:
   import Queue as queue
except ImportError:
   import queue


lock = threading.Lock()

class ImageListener:

    def __init__(self, cfg, model):

        self.cfg = cfg
        self.model = model
        self.cv_bridge = CvBridge()
        self.renders = dict()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None

        suffix = '_%02d' % (cfg.instance_id)
        prefix = '%02d_' % (cfg.instance_id)
        self.suffix = suffix
        self.prefix = prefix
        fusion_type = ''

        # initialize a node
        rospy.init_node("yolox_rgb")
        self.br = tf.TransformBroadcaster()
        self.label_pub = rospy.Publisher('yolo_label' + fusion_type + suffix, Image, queue_size=10)

        if cfg.ROS_CAMERA == 'D435':
            # use RealSense D435
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.base_frame = 'measured/camera_color_optical_frame'
        elif cfg.ROS_CAMERA == 'Azure':             
            rgb_sub = message_filters.Subscriber('/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/depth_to_rgb/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/rgb/camera_info', CameraInfo)
            self.base_frame = 'rgb_camera_link'
        elif cfg.ROS_CAMERA == 'Fetch':
            rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
            self.base_frame = 'head_camera_rgb_optical_frame'
        else:
            # use kinect
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (cfg.TEST.ROS_CAMERA), Image, queue_size=2)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (cfg.TEST.ROS_CAMERA), Image, queue_size=2)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (cfg.TEST.ROS_CAMERA), CameraInfo)
            self.base_frame = '%s_depth_optical_frame' % (cfg.TEST.ROS_CAMERA)

        # update camera intrinsics
        K = np.array(msg.K).reshape(3, 3)
        self.intrinsic_matrix = K
        print(self.intrinsic_matrix)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)
        

    def callback_rgbd(self, rgb, depth):

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
        
        
    def run_network(self):

        with lock:
            if listener.im is None:
                return
            im = self.im.copy()
            depth_cv = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id

        fusion_type = ''
        start_time = time.time()
        
        loader_cfgs = self.cfg.dataloader.test
        test_loader = instantiate(loader_cfgs[0])
        test_dset_name = loader_cfgs[0].dataset.lst.names
        if not isinstance(test_dset_name, str):
            test_dset_name = ",".join(test_dset_name)
        
        evaluator_cfgs = cfg.dataloader.evaluator
        eval_cfg = evaluator_cfgs[0]
        if OmegaConf.is_readonly(eval_cfg):
            OmegaConf.set_readonly(eval_cfg, False)
        eval_cfg.output_dir = osp.join(cfg.train.output_dir, "inference", test_dset_name)
        OmegaConf.set_readonly(eval_cfg, True)        
        evaluator = instantiate(eval_cfg)
        class_names=evaluator._metadata.get("thing_classes")
        num_classes = len(class_names)
        print(class_names, len(class_names))
        
        # change image to torch tensor
        im_tensor = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)
        print(im_tensor.shape)

        # run yolo network        
        ret = yolox_inference_on_image(
            self.model,
            im_tensor,
            amp_test=self.cfg.test.amp_test,
            half_test=self.cfg.test.half_test,
            test_cfg=self.cfg.test,
            val_cfg=self.cfg.val,
        )        
        print(ret)
        print("--- %s seconds ---" % (time.time() - start_time))        
        
        det_preds = ret['det_preds'][0].cpu().numpy()
        num = det_preds.shape[0]
        print('%d object detected' % num)

        # publish box image
        # Blue color in BGR
        color = (0, 255, 0)
        # Line thickness of 2 px
        thickness = 2
  
        im_label = im.copy()
        for i in range(num):
            bbox = det_preds[i, 0:4].astype(np.int32)
            im_label = cv2.rectangle(im_label, bbox[:2], bbox[2:], color, thickness)        
        
        label_msg = self.cv_bridge.cv2_to_imgmsg(im_label)
        label_msg.header.stamp = rospy.Time.now()
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'bgr8'
        self.label_pub.publish(label_msg)

        if num == 0:
            return

        indexes = np.zeros((num_classes, ), dtype=np.int32)
        for i in range(num):
            cls = int(det_preds[i, -1])

            if class_names[cls][3] == '_':
                name = self.prefix + class_names[cls][4:]
            else:
                name = self.prefix + class_names[cls]
            name = name + fusion_type
            indexes[cls] += 1
            name = name + '_%02d' % (indexes[cls])
            tf_name = os.path.join("yolo", name)

            # send transformation as bounding box (mis-used)
            n = np.linalg.norm(det_preds[i, 0:4])
            x1 = det_preds[i, 0] / n
            y1 = det_preds[i, 1] / n
            x2 = det_preds[i, 2] / n
            y2 = det_preds[i, 3] / n
            now = rospy.Time.now()
            self.br.sendTransform([n, now.secs, 0], [x1, y1, x2, y2], now, tf_name + '_roi', self.base_frame)
    
    
def setup(args):
    """Create configs and perform basic setups."""
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_yolox_setup(cfg, args)
    register_datasets_in_cfg(cfg)
    setproctitle("{}.{}".format(cfg.train.exp_name, get_time_str()))
    cfg.instance_id = 0  # set instance id to zero
    cfg.ROS_CAMERA = 'Fetch'
    cfg.test.conf_thr = 0.5
    cfg.test.nms_thr = 0.5
    return cfg


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print('Called with args:')
    print(args)
    
    cfg = setup(args)
    print('Using config:')
    pprint.pprint(cfg)
    
    # prepare network
    Trainer = YOLOX_DefaultTrainer    
    model = Trainer.build_model(cfg)
    MyCheckpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
        cfg.train.init_checkpoint, resume=args.resume
    )
    if cfg.test.fuse_conv_bn:
        model = fuse_model(model)

    # image listener
    listener = ImageListener(cfg, model)
    print("Going to run the listener using the network------------------------")
    while not rospy.is_shutdown():       
       listener.run_network()
