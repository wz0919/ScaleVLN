''' Batched REVERIE navigation environment '''

import json
import os
import numpy as np
import math
import random
import networkx as nx
import copy
import h5py
import jsonlines
import collections

from utils.data import load_nav_graphs, new_simulator
from utils.data import angle_feature, get_all_point_angle_feature

from .env import EnvBatch, ReverieObjectNavBatch
from utils.data import ImageFeaturesDB
from .data_utils import ObjectFeatureDB

def construct_instrs(instr_files, max_instr_len=512):
    data = []
    for instr_file in instr_files:
        with jsonlines.open(instr_file) as f:
            for item in f:
                newitem = {
                    'instr_id': item['instr_id'], 
                    'objId': item['objid'],
                    'scan': item['scan'],
                    'path': item['path'],
                    'end_vps': item['pos_vps'],
                    'instruction': item['instruction'],
                    'instr_encoding': item['instr_encoding'][:max_instr_len],
                    'heading': np.random.rand() * np.pi * 2,
                }
                data.append(newitem)
    return data


class HM3DReverieObjectNavBatch(ReverieObjectNavBatch):
    ''' Implements the REVERIE navigation task, using discretized viewpoints and pretrained features '''

    def __init__(
        self, view_db_file, obj_db_file, instr_files, connectivity_dir,
        multi_endpoints=False, multi_startpoints=False,
        image_feat_size=768, obj_feat_size=768,
        batch_size=64, angle_feat_size=4, max_objects=None, 
        seed=0, name=None, sel_data_idxs=None, scan_ranges=None
    ):
        view_db = ImageFeaturesDB(view_db_file, image_feat_size)
        obj_db = ObjectFeatureDB(obj_db_file, obj_feat_size, im_width=224, im_height=224)
        instr_data = construct_instrs(instr_files, max_instr_len=100)
        if scan_ranges is not None:
            scan_idxs = set(list(range(scan_ranges[0], scan_ranges[1])))
            new_instr_data = []
            for item in instr_data:
                if int(item['scan'].split('-')[0]) in scan_idxs:
                    new_instr_data.append(item)
            instr_data = new_instr_data
        #print(connectivity_dir)
        #exit()
        self.env = EnvBatch(connectivity_dir, feat_db=view_db, batch_size=batch_size)
        self.obj_db = obj_db
        self.data = instr_data
        self.scans = set([x['scan'] for x in self.data])
        self.multi_endpoints = multi_endpoints
        self.multi_startpoints = multi_startpoints
        self.connectivity_dir = connectivity_dir
        self.batch_size = batch_size
        self.angle_feat_size = angle_feat_size
        self.max_objects = max_objects
        self.name = name

        self.gt_trajs = self._get_gt_trajs(self.data) # for evaluation

        # in validation, we would split the data
        if sel_data_idxs is not None:
            t_split, n_splits = sel_data_idxs
            ndata_per_split = len(self.data) // n_splits 
            start_idx = ndata_per_split * t_split
            if t_split == n_splits - 1:
                end_idx = None
            else:
                end_idx = start_idx + ndata_per_split
            self.data = self.data[start_idx: end_idx]

        # use different seeds in different processes to shuffle data
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.obj2vps = collections.defaultdict(list)  # {scan_objid: vp_list} (objects can be viewed at the viewpoints)
        for item in self.data:
            self.obj2vps['%s_%s'%(item['scan'], item['objId'])].extend(item['end_vps'])

        self.ix = 0
        self._load_nav_graphs()

        self.sim = new_simulator(self.connectivity_dir)
        self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size)
        
        self.buffered_state_dict = {}
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))
