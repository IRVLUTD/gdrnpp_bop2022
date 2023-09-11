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
import scipy.io
import random
import datetime
import tf.transformations as tra
import matplotlib.pyplot as plt
import ros_numpy

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

from core.utils.data_utils import crop_resize_by_warp_affine, get_2d_coord_np, read_image_mmcv, xyz_to_region
from lib.utils.config_utils import try_get_key
from core.gdrn_modeling.engine.engine_utils import batch_data

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

    def __init__(self, cfg, model, extents):

        print(' *** Initializing GDRNPP ROS Node ... ')

        # variables
        self.cfg = cfg
        self.model = model
        self.extents = extents
        
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

        # start pose thread
        # self.start_publishing_tf()


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

                Tbo = self.pose_rbpf.rbpfs[i].T_in_base

                # publish tf
                t_bo = Tbo[:3, 3]
                q_bo = mat2quat(Tbo[:3, :3])
                self.br.sendTransform(t_bo, [q_bo[1], q_bo[2], q_bo[3], q_bo[0]], rospy.Time.now(), name, self.target_frame)
                    
            rate.sleep()
            # self.stop_event.wait(timeout=0.1)


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
                    print('find yolo detection ' + source_frame)
                    print(rois_est)
                except:
                    continue


        # call pose estimation function
        image_disp, image_disp_1 = self.process_image_multi_obj(input_rgb, input_depth, rois_est)

        # visualization
        # '''
        if image_disp is not None:
            pose_msg = self.cv_bridge.cv2_to_imgmsg(image_disp)
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = self.input_frame_id
            pose_msg.encoding = 'rgb8'
            self.pose_pub.publish(pose_msg)
            self.image_disp = image_disp
            
            pose_msg = self.cv_bridge.cv2_to_imgmsg(image_disp_1)
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = self.input_frame_id
            pose_msg.encoding = 'rgb8'
            self.pose_pub_1.publish(pose_msg)
            
        # '''
        
        
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
            print(roi_img.shape)
            import matplotlib.pyplot as plt
            plt.imshow(roi_img.transpose(1, 2, 0))
            plt.show()              

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
        batch = self.read_data_test(rgb[:, :, (2, 1, 0)], depth, rois)
        print(batch)
        
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
        print(out_dict)
        sys.exit(1)

        image_rgb = rgb.astype(np.float32) / 255.0
        image_bgr = image_rgb[:, :, (2, 1, 0)]
        image_bgr = torch.from_numpy(image_bgr).cuda()
        im_label = torch.from_numpy(im_label).cuda()

        # backproject depth
        depth = torch.from_numpy(depth).cuda()
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        px = self.intrinsic_matrix[0, 2]
        py = self.intrinsic_matrix[1, 2]
        # dpoints = backproject(depth, self.intrinsic_matrix)
        im_pcloud = posecnn_cuda.backproject_forward(fx, fy, px, py, depth)[0]

        # collect rois from rbpfs
        num_rbpfs = self.pose_rbpf.num_rbpfs
        rois_rbpf = np.zeros((num_rbpfs, 7), dtype=np.float32)
        for i in range(num_rbpfs):
            rois_rbpf[i, :] = self.pose_rbpf.rbpfs[i].roi
            self.pose_rbpf.rbpfs[i].roi_assign = None

        # data association based on bounding box overlap
        num_rois = rois.shape[0]
        assigned_rois = np.zeros((num_rois, ), dtype=np.int32)
        if num_rbpfs > 0 and num_rois > 0:
            # overlaps: (rois x gt_boxes) (batch_id, x1, y1, x2, y2)
            overlaps = bbox_overlaps(np.ascontiguousarray(rois_rbpf[:, (1, 2, 3, 4, 5)], dtype=np.float),
                np.ascontiguousarray(rois[:, (1, 2, 3, 4, 5)], dtype=np.float))

            # assign rois to rbpfs
            assignment = overlaps.argmax(axis=1)
            max_overlaps = overlaps.max(axis=1)
            unassigned = []
            for i in range(num_rbpfs):
                if max_overlaps[i] > 0.2:
                    self.pose_rbpf.rbpfs[i].roi_assign = rois[assignment[i]]
                    assigned_rois[assignment[i]] = 1
                else:
                    unassigned.append(i)

            # check if there are un-assigned rois
            index = np.where(assigned_rois == 0)[0]

            # if there is un-assigned rbpfs
            if len(unassigned) > 0 and len(index) > 0:
                for i in range(len(unassigned)):
                    for j in range(len(index)):
                        if assigned_rois[index[j]] == 0 and self.pose_rbpf.rbpfs[unassigned[i]].roi[1] == rois[index[j], 1]:
                            self.pose_rbpf.rbpfs[unassigned[i]].roi_assign = rois[index[j]]
                            assigned_rois[index[j]] = 1

        elif num_rbpfs == 0 and num_rois == 0:
            return False, None, None

        # initialize new object
        if num_rois > 0:
            good_initial = True
        else:
            good_initial = False

        start_time = rospy.Time.now()
        for i in range(num_rois):
            if assigned_rois[i]:
                continue

            print('Initializing detection {} ... '.format(i))
            roi = rois[i].copy()
            print(roi)
            self.pose_rbpf.estimation_poserbpf(roi, self.intrinsic_matrix, image_bgr, depth, im_pcloud, im_label, self.grasp_mode, self.grasp_cls)

            # pose evaluation
            image_tensor, pcloud_tensor = self.pose_rbpf.render_image_all(self.intrinsic_matrix, self.grasp_mode, self.grasp_cls)
            cls = cfg.TEST.CLASSES[int(roi[1])]
            sim, depth_error, vis_ratio = self.pose_rbpf.evaluate_6d_pose(self.pose_rbpf.rbpfs[-1].roi, self.pose_rbpf.rbpfs[-1].pose, cls, \
                image_bgr, image_tensor, pcloud_tensor, depth, self.intrinsic_matrix, im_label)
            print('Initialization : Object: {}, Sim obs: {}, Depth Err: {:.3}, Vis Ratio: {:.2}'.format(i, sim, depth_error, vis_ratio))

            if sim < cfg.PF.THRESHOLD_SIM or torch.isnan(depth_error) or depth_error > cfg.PF.THRESHOLD_DEPTH or vis_ratio < cfg.PF.THRESHOLD_RATIO:
                print('===================is NOT initialized!=================')
                self.pose_rbpf.num_objects_per_class[self.pose_rbpf.rbpfs[-1].cls_id, self.pose_rbpf.rbpfs[-1].object_id] = 0
                with lock_tf:
                    del self.pose_rbpf.rbpfs[-1]
                good_initial = False
            else:
                print('===================is initialized!======================')
                self.pose_rbpf.rbpfs[-1].roi_assign = roi
                if self.grasp_mode:
                    if not (sim < cfg.PF.THRESHOLD_SIM_GRASPING or depth_error > cfg.PF.THRESHOLD_DEPTH_GRASPING or vis_ratio < cfg.PF.THRESHOLD_RATIO_GRASPING):
                        self.pose_rbpf.rbpfs[-1].graspable = True
                        self.pose_rbpf.rbpfs[-1].status = True
                        self.pose_rbpf.rbpfs[-1].need_filter = False
        print('initialization time %.6f' % (rospy.Time.now() - start_time).to_sec())

        # filter all the objects
        print('Filtering objects')
        save, image_tensor = self.pose_rbpf.filtering_poserbpf(self.intrinsic_matrix, image_bgr, depth, im_pcloud, im_label, self.grasp_mode, self.grasp_cls)
        print('*********full time %.6f' % (rospy.Time.now() - start_time).to_sec())

        # non-maximum suppression within class
        num = self.pose_rbpf.num_rbpfs
        status = np.zeros((num, ), dtype=np.int32)
        rois = np.zeros((num, 7), dtype=np.float32)
        for i in range(num):
            rois[i, :6] = self.pose_rbpf.rbpfs[i].roi[:6]
            rois[i, 6] = self.pose_rbpf.rbpfs[i].num_frame
        keep = nms(rois, 0.5)
        status[keep] = 1

        # remove untracked objects
        for i in range(num):
            if status[i] == 0 or self.pose_rbpf.rbpfs[i].num_lost >= cfg.TEST.NUM_LOST:
                print('###############remove rbpf#################')
                self.pose_rbpf.num_objects_per_class[self.pose_rbpf.rbpfs[i].cls_id, self.pose_rbpf.rbpfs[i].object_id] = 0
                status[i] = 0
                save = False
        with lock_tf:
            self.pose_rbpf.rbpfs = [self.pose_rbpf.rbpfs[i] for i in range(num) if status[i] > 0]

        if self.pose_rbpf.num_rbpfs == 0:
            save = False

        # image to publish for visualization
        # '''
        if image_tensor is not None:
            image_disp = (0.4 * image_bgr[:, :, (2, 1, 0)] + 0.6 * image_tensor) * 255
            image_disp_1 = image_tensor * 255
        else:
            image_disp = 0.4 * image_bgr[:, :, (2, 1, 0)] * 255
            image_disp_1 = image_disp
        image_disp = torch.clamp(image_disp, 0, 255).byte().cpu().numpy()
        image_disp_1 = torch.clamp(image_disp_1, 0, 255).byte().cpu().numpy()        
        # '''
        # image_disp = None
        
        return save & good_initial, image_disp, image_disp_1
