#!/usr/bin/env python

import torch
import torch.distributed as dist

import logging
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
os.environ["PYOPENGL_PLATFORM"] = "egl"

import rospy
import glob
import copy


cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../"))
from pytorch_lightning import seed_everything
from detectron2.data import MetadataCatalog
from mmcv import Config
from setproctitle import setproctitle

from core.utils.my_setup import setup_for_distributed
from core.utils.my_checkpoint import MyCheckpointer
from lib.utils.time_utils import get_time_str
from lib.utils.utils import iprint
from gdrnpp_listener import ImageListener
from core.utils.default_args_setup import my_default_argument_parser, my_default_setup
from core.gdrn_modeling.engine.engine import GDRN_Lite
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.models import (
    GDRN,
    GDRN_no_region,
    GDRN_cls,
    GDRN_cls2reg,
    GDRN_double_mask,
    GDRN_Dstream_double_mask,
)  # noqa


logger = logging.getLogger("detectron2")

def setup(args):
    """Create configs and perform basic setups."""
    cfg = Config.fromfile(args.config_file)
    if args.opts is not None:
        cfg.merge_from_dict(args.opts)
    ############## pre-process some cfg options ######################
    # NOTE: check if need to set OUTPUT_DIR automatically
    if cfg.OUTPUT_DIR.lower() == "auto":
        cfg.OUTPUT_DIR = osp.join(
            cfg.OUTPUT_ROOT,
            osp.splitext(args.config_file)[0].split("configs/")[1],
        )
        iprint(f"OUTPUT_DIR was automatically set to: {cfg.OUTPUT_DIR}")

    if cfg.get("EXP_NAME", "") == "":
        setproctitle("{}.{}".format(osp.splitext(osp.basename(args.config_file))[0], get_time_str()))
    else:
        setproctitle("{}.{}".format(cfg.EXP_NAME, get_time_str()))

    if cfg.SOLVER.AMP.ENABLED:
        if torch.cuda.get_device_capability() <= (6, 1):
            iprint("Disable AMP for older GPUs")
            cfg.SOLVER.AMP.ENABLED = False

    # NOTE: pop some unwanted configs in detectron2
    # ---------------------------------------------------------
    cfg.SOLVER.pop("STEPS", None)
    cfg.SOLVER.pop("MAX_ITER", None)
    bs_ref = cfg.SOLVER.get("REFERENCE_BS", cfg.SOLVER.IMS_PER_BATCH)  # nominal batch size
    if bs_ref <= cfg.SOLVER.IMS_PER_BATCH:
        bs_ref = cfg.SOLVER.REFERENCE_BS = cfg.SOLVER.IMS_PER_BATCH
        # default DDP implementation is slow for accumulation according to: https://pytorch.org/docs/stable/notes/ddp.html
        # all-reduce operation is carried out during loss.backward().
        # Thus, there would be redundant all-reduce communications in a accumulation procedure,
        # which means, the result is still right but the training speed gets slower.
        # TODO: If acceleration is needed, there is an implementation of allreduce_post_accumulation
        # in https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/run_pretraining.py
        accumulate_iter = max(round(bs_ref / cfg.SOLVER.IMS_PER_BATCH), 1)  # accumulate loss before optimizing
    else:
        accumulate_iter = 1
    # NOTE: get optimizer from string cfg dict
    if cfg.SOLVER.OPTIMIZER_CFG != "":
        if isinstance(cfg.SOLVER.OPTIMIZER_CFG, str):
            optim_cfg = eval(cfg.SOLVER.OPTIMIZER_CFG)
            cfg.SOLVER.OPTIMIZER_CFG = optim_cfg
        else:
            optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
        iprint("optimizer_cfg:", optim_cfg)
        cfg.SOLVER.OPTIMIZER_NAME = optim_cfg["type"]
        cfg.SOLVER.BASE_LR = optim_cfg["lr"]
        cfg.SOLVER.MOMENTUM = optim_cfg.get("momentum", 0.9)
        cfg.SOLVER.WEIGHT_DECAY = optim_cfg.get("weight_decay", 1e-4)
        if accumulate_iter > 1:
            if "weight_decay" in cfg.SOLVER.OPTIMIZER_CFG:
                cfg.SOLVER.OPTIMIZER_CFG["weight_decay"] *= (
                    cfg.SOLVER.IMS_PER_BATCH * accumulate_iter / bs_ref
                )  # scale weight_decay
    if accumulate_iter > 1:
        cfg.SOLVER.WEIGHT_DECAY *= cfg.SOLVER.IMS_PER_BATCH * accumulate_iter / bs_ref
    # -------------------------------------------------------------------------
    if cfg.get("DEBUG", False):
        iprint("DEBUG")
        args.num_gpus = 1
        args.num_machines = 1
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.TRAIN.PRINT_FREQ = 1
    # register datasets
    register_datasets_in_cfg(cfg)

    exp_id = "{}".format(osp.splitext(osp.basename(args.config_file))[0])

    if args.eval_only:
        if cfg.TEST.USE_PNP:
            # NOTE: need to keep _test at last
            exp_id += "{}_test".format(cfg.TEST.PNP_TYPE.upper())
        else:
            exp_id += "_test"
    cfg.EXP_ID = exp_id
    cfg.RESUME = args.resume
    
    cfg.instance_id = 0
    cfg.TEST.ROS_CAMERA = 'Fetch'
    
    ####################################
    # cfg.freeze()
    return cfg
    
    
class Lite(GDRN_Lite):
    def set_my_env(self, args, cfg):
        my_default_setup(cfg, args)  # will set os.environ["PYTHONHASHSEED"]
        seed_everything(int(os.environ["PYTHONHASHSEED"]), workers=True)
        setup_for_distributed(is_master=self.is_global_zero)

    def run(self, args, cfg):
        self.set_my_env(args, cfg)

        # get renderer ----------------------
        if args.eval_only or cfg.TEST.SAVE_RESULTS_ONLY or (not cfg.MODEL.POSE_NET.XYZ_ONLINE):
            renderer = None
        else:
            train_dset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            data_ref = ref.__dict__[train_dset_meta.ref_key]
            train_obj_names = train_dset_meta.objs
            render_gpu_id = self.local_rank
            renderer = get_renderer(cfg, data_ref, obj_names=train_obj_names, gpu_id=render_gpu_id)

        logger.info(f"Used GDRN module name: {cfg.MODEL.POSE_NET.NAME}")
        model, optimizer = eval(cfg.MODEL.POSE_NET.NAME).build_model_optimizer(cfg, is_test=args.eval_only)
        logger.info("Model:\n{}".format(model))

        # don't forget to call `setup` to prepare for model / optimizer for distributed training.
        # the model is moved automatically to the right device.
        if optimizer is not None:
            model, optimizer = self.setup(model, optimizer)
        else:
            model = self.setup(model)

        if True:
            # sum(p.numel() for p in model.parameters() if p.requires_grad)
            params = sum(p.numel() for p in model.parameters()) / 1e6
            logger.info("{}M params".format(params))

        MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR, prefix_to_remove="_module.").resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        return cfg, model


if __name__ == '__main__':
    parser = my_default_argument_parser()
    parser.add_argument(
        "--strategy",
        default=None,
        type=str,
        help="the strategy for parallel training: dp | ddp | ddp_spawn | deepspeed | ddp_sharded",
    )
    args = parser.parse_args()

    print('Called with args:')
    print(args)

    print('Using config:')
    cfg = setup(args)
    pprint.pprint(cfg)

    # set up model    
    cfg, model = Lite(
        accelerator="gpu",
        strategy=args.strategy,
        devices=args.num_gpus,
        num_nodes=args.num_machines,
        precision=16 if cfg.SOLVER.AMP.ENABLED else 32,
    ).run(args, cfg)
    
    # get class names
    dataset_name = cfg.DATASETS.TRAIN[0]
    dataset_meta = MetadataCatalog.get(dataset_name)
    print(dataset_name, dataset_meta)
    cfg.class_names = dataset_meta.objs
    print(cfg.class_names)
    
    # load extents
    extents = np.loadtxt('extents.txt', delimiter=',')
    print(extents, extents.shape)

    # image listener
    listener = ImageListener(cfg, dataset_name, model, extents)
    while not rospy.is_shutdown():
        listener.process_data()
    listener.stop_publishing_tf()
