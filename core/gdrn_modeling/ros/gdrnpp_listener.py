import time
import rospy
import tf
import message_filters
import cv2
import numpy as np
import torch
import torch.nn as nn
import threading
import sys
import os.path as osp
import scipy.io
import random
import datetime
import tf.transformations as tra
import matplotlib.pyplot as plt
import ros_numpy
import ref

# from queue import queue
from random import shuffle
from cv_bridge import CvBridge, CvBridgeError
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import JointState
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from geometry_msgs.msg import PoseStamped, PoseArray
from detectron2.data import MetadataCatalog

from core.utils.data_utils import crop_resize_by_warp_affine, get_2d_coord_np, read_image_mmcv, xyz_to_region
from lib.utils.config_utils import try_get_key
from core.gdrn_modeling.engine.engine_utils import batch_data, get_out_coor, get_out_mask, batch_data_inference_roi

lock = threading.Lock()
lock_tf = threading.Lock()

def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans
    return obj_T


def get_relative_pose_from_tf(listener, source_frame, target_frame):
    first_time = True
    while True:
        try:
            stamp = rospy.Time.now()
            init_trans, init_rot = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            break
        except Exception as e:
            if first_time:
                print(str(e))
                first_time = False
            continue
    return ros_qt_to_rt(init_rot, init_trans), stamp


def rotation_matrix_from_vectors(A, B):
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    v = np.cross(A, B)
    ssc = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + ssc + np.matmul(ssc, ssc) * (1 - np.dot(A, B)) / (np.linalg.norm(v))**2
    return R


class ImageListener:

    def __init__(self, cfg, dataset_name, model, extents):

        print(' *** Initializing GDRNPP ROS Node ... ')

        # variables
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.model = model
        self.extents = extents
        self._cpu_device = torch.device("cpu")
        
        self.cv_bridge = CvBridge()
        self.count = 0
        self.objects = []
        self.frame_names = []
        self.frame_lost = []
        self.queue_size = 10

        self.camera_type = cfg.TEST.ROS_CAMERA
        self.suffix = '_%02d' % (cfg.instance_id)
        self.prefix = '%02d_' % (cfg.instance_id)

        self.init_failure_steps = 0
        self.input_rgb = None
        self.input_depth = None
        self.input_rois = None
        self.input_stamp = None
        self.input_frame_id = None
        self.input_joint_states = None
        self.input_robot_joint_states = None
        self.main_thread_free = True
        self.kf_time_stamp = None

        # thread for publish poses
        self.tf_thread = None
        self.stop_event = None

        # initialize a node
        rospy.init_node('gdrnpp_image_listener' + self.suffix)
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        rospy.sleep(3.0)
        self.pose_pub = rospy.Publisher('gdrnpp_image' + self.suffix, Image, queue_size=1)

        # subscriber for camera information
        # self.base_frame = 'measured/base_link'
        if cfg.TEST.ROS_CAMERA == 'D415':
            self.base_frame = 'measured/camera_color_optical_frame'        
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=1)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.target_frame = self.base_frame
            self.camera_frame = 'measured/camera_color_optical_frame'
            self.forward_kinematics = True

            '''
            self.T_delta = np.array([[0.99911077, 0.04145749, -0.00767817, 0.003222],  # -0.003222, -0.013222 (left plus and right minus)
                                     [-0.04163608, 0.99882554, -0.02477858, 0.01589],  # -0.00289, 0.01089 (close plus and far minus)
                                     [0.0066419, 0.02507623, 0.99966348, 0.003118],
                                     [0., 0., 0., 1.]], dtype=np.float32)
            '''
            self.T_delta = np.eye(4, dtype=np.float32)
            
        elif cfg.TEST.ROS_CAMERA  == 'Fetch':
            self.base_frame = 'base_link'
            rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
            self.camera_frame = 'head_camera_rgb_optical_frame'
            self.target_frame = self.base_frame
            self.forward_kinematics = True

        elif cfg.TEST.ROS_CAMERA == 'Azure':
            self.base_frame = 'measured/base_link'        
            rgb_sub = message_filters.Subscriber('/k4a/rgb/image_raw', Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', Image, queue_size=1)
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)
            self.target_frame = self.base_frame
            self.camera_frame = 'rgb_camera_link'
            self.forward_kinematics = False
        elif cfg.TEST.ROS_CAMERA == 'ISAAC_SIM':
            rgb_sub = message_filters.Subscriber('/sim/left_color_camera/image', Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/sim/left_depth_camera/image', Image, queue_size=1)
            msg = rospy.wait_for_message('/sim/left_color_camera/camera_info', CameraInfo)
            self.target_frame = self.base_frame
            self.forward_kinematics = True
        else:
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (cfg.TEST.ROS_CAMERA), Image, queue_size=1)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (cfg.TEST.ROS_CAMERA), Image, queue_size=1)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (cfg.TEST.ROS_CAMERA), CameraInfo)
            self.forward_kinematics = False

        # camera to base transformation
        self.Tbc_now = np.eye(4, dtype=np.float32)
        self.Tbc_prev = np.eye(4, dtype=np.float32)
        self.camera_distance = 0

        K = np.array(msg.K).reshape(3, 3)
        self.intrinsic_matrix = K
        print('Intrinsics matrix : ')
        print(self.intrinsic_matrix)

        # set up ros service
        print(' GDRNPP ROS Node is Initialized ! *** ')
        self.is_keyframe = False

        # subscriber for posecnn label
        queue_size = 10
        slop_seconds = 0.2
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)

        self.Tbr_kf = np.eye(4, dtype=np.float32)  # keyframe which is used for refine the object pose
        self.Tbr_kf_list = []
        self.Tco_kf_list = []
        self.record = False
        self.Tbr_save = np.eye(4, dtype=np.float32)
        self.image_disp = None
        
        # main results
        self.rois_est = None
        self.yolo_names = None
        self.poses = None
        
        if cfg.TEST.USE_DEPTH_REFINE:
            from lib.render_vispy.model3d import load_models
            from lib.render_vispy.renderer import Renderer

            net_cfg = cfg.MODEL.POSE_NET
            width = net_cfg.OUTPUT_RES
            height = width
            self.depth_refine_threshold = cfg.TEST.DEPTH_REFINE_THRESHOLD

            self._metadata = MetadataCatalog.get(dataset_name)
            self.data_ref = ref.__dict__[self._metadata.ref_key]

            self.ren = Renderer(size=(width, height), cam=self.data_ref.camera_matrix)
            self.ren_models = load_models(
                model_paths=self.data_ref.model_paths,
                scale_to_meter=0.001,
                cache_dir=".cache",
                texture_paths=self.data_ref.texture_paths if cfg.DEBUG else None,
                center=False,
                use_cache=True,
            )

        # start pose thread
        self.start_publishing_tf()


    def start_publishing_tf(self):
        self.stop_event = threading.Event()
        self.tf_thread = threading.Thread(target=self.tf_thread_func)
        self.tf_thread.start()


    def stop_publishing_tf(self):
        if self.tf_thread is None:
            return False
        self.stop_event.set()
        self.tf_thread.join()
        return True


    # publish poses
    def tf_thread_func(self):
        rate = rospy.Rate(30.)
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            # publish pose
            with lock_tf:
                if self.rois_est is not None and self.yolo_names is not None and self.poses is not None:
                    num = self.rois_est.shape[0]
                    for i in range(num):
                        name = self.yolo_names[i].replace('yolo', 'gdrnpp')
                        print(name)
                        # publish tf
                        pose = self.poses[i]
                        t_bo = pose[:, 3]
                        q_bo = mat2quat(pose[:, :3])
                        self.br.sendTransform(t_bo, [q_bo[1], q_bo[2], q_bo[3], q_bo[0]], rospy.Time.now(), name, self.camera_frame)
            rate.sleep()


    # callback function to get images
    def callback(self, rgb, depth):

        self.Tbc_now, self.Tbc_stamp = get_relative_pose_from_tf(self.listener, self.camera_frame, self.base_frame)

        # decode image
        if depth is not None:
            if depth.encoding == '32FC1':
                depth_cv = ros_numpy.numpify(depth)
            elif depth.encoding == '16UC1':
                depth_cv = ros_numpy.numpify(depth)
                depth_cv = depth.copy().astype(np.float32)
                depth_cv /= 1000.0
            else:
                rospy.logerr_throttle(1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(depth.encoding))
                return
        else:
            depth_cv = None

        with lock:
            self.input_depth = depth_cv
            # rgb image used for posecnn detection
            self.input_rgb = ros_numpy.numpify(rgb)
            # other information
            self.input_stamp = rgb.header.stamp
            self.input_frame_id = rgb.header.frame_id


    def process_data(self):
        # callback data
        with lock:
            if self.input_rgb is None:
                return
            input_stamp = self.input_stamp
            input_rgb = self.input_rgb.copy()
            input_depth = self.input_depth.copy()
            input_Tbc = self.Tbc_now.copy()
            input_Tbc_stamp = self.Tbc_stamp

        # detection information of the target object
        rois_est = np.zeros((0, 7), dtype=np.float32)
        yolo_names = []
        # TODO look for multiple object instances
        max_objects = 5
        class_names = self.cfg.class_names
        for i in range(len(class_names)):

            for object_id in range(max_objects):
                suffix_frame = '_%02d_roi' % (object_id)

                # check posecnn frame
                if class_names[i][3] == '_':
                    source_frame = 'yolo/' + self.prefix + class_names[i][4:] + suffix_frame
                else:
                    source_frame = 'yolo/' + self.prefix + class_names[i] + suffix_frame

                try:
                    # print('look for yolo detection ' + source_frame)
                    trans, rot = self.listener.lookupTransform(self.camera_frame, source_frame, rospy.Time(0))
                    n = trans[0]
                    secs = trans[1]
                    now = rospy.Time.now()
                    if abs(now.secs - secs) > 1.0:
                        print('yolo detection for %s time out %f %f' % (source_frame, now.secs, secs))
                        continue
                    roi = np.zeros((1, 7), dtype=np.float32)
                    roi[0, 0] = 0
                    roi[0, 1] = i
                    roi[0, 2] = rot[0] * n
                    roi[0, 3] = rot[1] * n
                    roi[0, 4] = rot[2] * n
                    roi[0, 5] = rot[3] * n
                    roi[0, 6] = trans[2]
                    rois_est = np.concatenate((rois_est, roi), axis=0)
                    yolo_names.append(source_frame)
                    print('find yolo detection ' + source_frame)
                    print(rois_est)
                except:
                    continue

        # call pose estimation function
        poses = self.process_image_multi_obj(input_rgb, input_depth, rois_est)
        self.rois_est = rois_est
        self.yolo_names = yolo_names
        self.poses = poses     
        
        
    def normalize_image(self, cfg, image):
        """
        cfg: upper format, the whole cfg; lower format, the input_cfg
        image: CHW format
        """
        pixel_mean = np.array(try_get_key(cfg, "MODEL.PIXEL_MEAN", "pixel_mean")).reshape(-1, 1, 1)
        pixel_std = np.array(try_get_key(cfg, "MODEL.PIXEL_STD", "pixel_std")).reshape(-1, 1, 1)
        return (image - pixel_mean) / pixel_std
        
        
    def read_data_test(self, image, depth, rois):
        """load image and annos random shift & scale bbox; crop, rescale."""

        im_H, im_W = image.shape[:2]

        net_cfg = self.cfg.MODEL.POSE_NET
        input_res = net_cfg.INPUT_RES
        out_res = net_cfg.OUTPUT_RES

        # CHW -> HWC
        coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)

        # don't load annotations at test time
        test_bbox_type = self.cfg.TEST.TEST_BBOX_TYPE
        if test_bbox_type == "gt":
            bbox_key = "bbox"
        else:
            bbox_key = f"bbox_{test_bbox_type}"

        # here get batched rois
        roi_infos = {}
        # yapf: disable
        roi_keys = ["scene_im_id", "file_name", "cam", "im_H", "im_W",
                    "roi_img", "inst_id", "roi_coord_2d", "roi_coord_2d_rel",
                    "roi_cls", "score", "time", "roi_extent",
                    bbox_key, "bbox_mode", "bbox_center", "roi_wh",
                    "scale", "resize_ratio", "model_info",
        ]
        depth_ch = 1
        roi_keys.append("roi_depth")
        for _key in roi_keys:
            roi_infos[_key] = []
        # yapf: enable
        # TODO: how to handle image without detections
        #   filter those when load annotations or detections, implement a function for this
        # "annotations" means detections
        
        num = rois.shape[0]
        for i in range(num):
            roi = rois[i]
        
            # inherent image-level infos
            roi_infos["im_H"].append(im_H)
            roi_infos["im_W"].append(im_W)
            roi_infos["cam"].append(self.intrinsic_matrix.astype("float32"))

            # roi-level infos
            roi_cls = int(roi[1])
            score = roi[-1]
            roi_infos["roi_cls"].append(roi_cls)
            roi_infos["score"].append(score)

            # extent
            roi_extent = self.extents[roi_cls]
            roi_infos["roi_extent"].append(roi_extent.astype("float32"))

            # TODO: adjust amodal bbox here
            x1, y1, x2, y2 = roi[2:6]
            bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
            bw = max(x2 - x1, 1)
            bh = max(y2 - y1, 1)
            scale = max(bh, bw) * self.cfg.INPUT.DZI_PAD_SCALE
            scale = min(scale, max(im_H, im_W)) * 1.0

            roi_infos["bbox_center"].append(bbox_center.astype("float32"))
            roi_infos["scale"].append(scale)
            roi_wh = np.array([bw, bh], dtype=np.float32)
            roi_infos["roi_wh"].append(roi_wh)
            roi_infos["resize_ratio"].append((out_res / scale).astype("float32"))

            # CHW, float32 tensor
            # roi_image
            roi_img = crop_resize_by_warp_affine(
                image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)
            
            # rgb -> bgr
            # print(roi_img.shape)
            # import matplotlib.pyplot as plt
            # plt.imshow(roi_img.transpose(1, 2, 0))
            # plt.show()              

            roi_img = self.normalize_image(self.cfg, roi_img)
            roi_infos["roi_img"].append(roi_img.astype("float32"))

            # roi_depth
            roi_depth = crop_resize_by_warp_affine(
                depth, bbox_center, scale, input_res, interpolation=cv2.INTER_NEAREST
            )
            if depth_ch == 1:
                roi_depth = roi_depth.reshape(1, input_res, input_res)
            else:
                roi_depth = roi_depth.transpose(2, 0, 1)
            roi_infos["roi_depth"].append(roi_depth.astype("float32"))

            # roi_coord_2d
            roi_coord_2d = crop_resize_by_warp_affine(
                coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
            ).transpose(
                2, 0, 1
            )  # HWC -> CHW
            roi_infos["roi_coord_2d"].append(roi_coord_2d.astype("float32"))

            if net_cfg.PNP_NET.COORD_2D_TYPE == "rel":
                # roi_coord_2d_rel
                roi_coord_2d_rel = (
                    bbox_center.reshape(2, 1, 1) - roi_coord_2d * np.array([im_W, im_H]).reshape(2, 1, 1)
                ) / scale
                roi_infos["roi_coord_2d_rel"].append(roi_coord_2d_rel.astype("float32"))
                
        dataset_dict = dict()
        for _key in roi_keys:
            if _key in ["roi_img", "roi_coord_2d", "roi_coord_2d_rel", "roi_depth"]:
                dataset_dict[_key] = torch.as_tensor(np.array(roi_infos[_key])).contiguous().cuda()
            elif _key in ["model_info", "scene_im_id", "file_name"]:
                # can not convert to tensor
                dataset_dict[_key] = roi_infos[_key]
            else:
                if isinstance(roi_infos[_key], list):
                    dataset_dict[_key] = torch.as_tensor(np.array(roi_infos[_key])).cuda()
                else:
                    dataset_dict[_key] = torch.as_tensor(roi_infos[_key]).cuda()

        return dataset_dict


    # function for pose etimation and tracking
    def process_image_multi_obj(self, rgb, depth, rois):    
    
        # prepare data, rgb -> bgr
        batch = self.read_data_test(rgb, depth, rois)
        
        # run network
        if self.cfg.INPUT.WITH_DEPTH and "depth" in self.cfg.MODEL.POSE_NET.NAME.lower():
            inp = torch.cat([batch["roi_img"], batch["roi_depth"]], dim=1)
            print('use depth')
        else:
            inp = batch["roi_img"]
            print('not use depth')
        
        '''
        print(inp.shape)
        print('roi_cls', batch["roi_cls"], batch["roi_cls"].shape)        
        print('cam', batch["cam"], batch["cam"].shape)           
        print('roi_wh', batch["roi_wh"], batch["roi_wh"].shape)           
        print('bbox_center', batch["bbox_center"], batch["bbox_center"].shape)
        print('resize_ratio', batch["resize_ratio"], batch["resize_ratio"].shape)        
        print('roi_coord_2d', batch["roi_coord_2d"], batch["roi_coord_2d"].shape)          
        print('roi_coord_2d_rel', batch["roi_coord_2d_rel"], batch["roi_coord_2d_rel"].shape)           
        print('roi_extent', batch["roi_extent"], batch["roi_extent"].shape)
        '''
        
        out_dict = self.model(
                    inp,
                    roi_classes=batch["roi_cls"],
                    roi_cams=batch["cam"],
                    roi_whs=batch["roi_wh"],
                    roi_centers=batch["bbox_center"],
                    resize_ratios=batch["resize_ratio"],
                    roi_coord_2d=batch.get("roi_coord_2d", None),
                    roi_coord_2d_rel=batch.get("roi_coord_2d_rel", None),
                    roi_extents=batch.get("roi_extent", None),
        )
        
        # depth refine
        inputs = [batch]
        poses = self.process_depth_refine(inputs, out_dict)        
        return poses
        
        
    def process_depth_refine(self, inputs, out_dict):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id", "scene_id".
            outputs:
        """
        cfg = self.cfg
        net_cfg = cfg.MODEL.POSE_NET
        out_res = net_cfg.OUTPUT_RES        
        out_coor_x = out_dict["coor_x"].detach()
        out_coor_y = out_dict["coor_y"].detach()
        out_coor_z = out_dict["coor_z"].detach()
        out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)
        out_xyz = out_xyz.to(self._cpu_device) #.numpy()

        out_mask = get_out_mask(cfg, out_dict["mask"].detach())
        out_mask = out_mask.to(self._cpu_device) #.numpy()
        out_rots = out_dict["rot"].detach().to(self._cpu_device).numpy()
        out_transes = out_dict["trans"].detach().to(self._cpu_device).numpy()

        zoom_K = batch_data_inference_roi(cfg, inputs)['roi_zoom_K']
        print('zoom_K', zoom_K)

        out_i = -1
        poses = []
        for i, _input, in enumerate(inputs):
            for inst_i in range(len(_input["roi_img"])):
                out_i += 1

                K = _input["cam"][inst_i].cpu().numpy().copy()
                # print('K', K)

                K_crop = zoom_K[inst_i].cpu().numpy().copy()
                # print('K_crop', K_crop)

                roi_label = _input["roi_cls"][inst_i]  # 0-based label
                score = _input["score"][inst_i]
                cls_name = cfg.class_names[roi_label]

                # get pose
                xyz_i = out_xyz[out_i].permute(1, 2, 0)
                mask_i = np.squeeze(out_mask[out_i])

                rot_est = out_rots[out_i]
                trans_est = out_transes[out_i]
                pose_est = np.hstack([rot_est, trans_est.reshape(3, 1)])
                depth_sensor_crop = cv2.resize(_input['roi_depth'][inst_i][-1].cpu().numpy().copy().squeeze(), (out_res, out_res))
                depth_sensor_mask_crop = depth_sensor_crop > 0

                net_cfg = cfg.MODEL.POSE_NET
                crop_res = net_cfg.OUTPUT_RES

                for _ in range(cfg.TEST.DEPTH_REFINE_ITER):
                    self.ren.clear()
                    self.ren.set_cam(K_crop)
                    self.ren.draw_model(self.ren_models[self.data_ref.objects.index(cls_name)], pose_est)
                    ren_im, ren_dp = self.ren.finish()
                    
                    # import matplotlib.pyplot as plt
                    # plt.imshow(ren_im)
                    # plt.show()  
                    
                    ren_mask = ren_dp > 0

                    if self.cfg.TEST.USE_COOR_Z_REFINE:
                        coor_np = xyz_i.numpy()
                        coor_np_t = coor_np.reshape(-1, 3)
                        coor_np_t = coor_np_t.T
                        coor_np_r = rot_est @ coor_np_t
                        coor_np_r = coor_np_r.T
                        coor_np_r = coor_np_r.reshape(crop_res, crop_res, 3)
                        query_img_norm = coor_np_r[:, :, -1] * mask_i.numpy()
                        query_img_norm = query_img_norm * ren_mask * depth_sensor_mask_crop
                    else:
                        query_img = xyz_i

                        query_img_norm = torch.norm(query_img, dim=-1) * mask_i
                        query_img_norm = query_img_norm.numpy() * ren_mask * depth_sensor_mask_crop
                    norm_sum = query_img_norm.sum()
                    if norm_sum == 0:
                        continue
                    query_img_norm /= norm_sum
                    norm_mask = query_img_norm > (query_img_norm.max() * self.depth_refine_threshold)
                    yy, xx = np.argwhere(norm_mask).T  # 2 x (N,)
                    depth_diff = depth_sensor_crop[yy, xx] - ren_dp[yy, xx]
                    depth_adjustment = np.median(depth_diff)

                    yx_coords = np.meshgrid(np.arange(crop_res), np.arange(crop_res))
                    yx_coords = np.stack(yx_coords[::-1], axis=-1)  # (crop_res, crop_res, 2yx)
                    yx_ray_2d = (yx_coords * query_img_norm[..., None]).sum(axis=(0, 1))  # y, x
                    ray_3d = np.linalg.inv(K_crop) @ (*yx_ray_2d[::-1], 1)
                    ray_3d /= ray_3d[2]

                    trans_delta = ray_3d[:, None] * depth_adjustment
                    trans_est = trans_est + trans_delta.reshape(3)
                    pose_est = np.hstack([rot_est, trans_est.reshape(3, 1)])
                print('pose_est', pose_est)
                poses.append(pose_est)
        return poses
