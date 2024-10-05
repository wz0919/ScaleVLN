import os
import json
import jsonlines
import networkx as nx
import math
import numpy as np

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import h5py
import random

class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}
        if ".hdf5" not in self.img_ft_file:
            self.env = lmdb.open(self.img_ft_file, readonly=True)
        else:
            print('pass!')
            with h5py.File(self.img_ft_file, 'r') as f:
                for key in list(f.keys()):
                    self._feature_store[key] = f[key][...].astype(np.float32)

    def __del__(self):
        if ".hdf5" not in self.img_ft_file:
            self.env.close()

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            if ".hdf5" in self.img_ft_file:
                with h5py.File(self.img_ft_file, 'r') as f:
                    ft = f[key][...].astype(np.float32)
            else:
                with self.env.begin() as txn:
                    ft = msgpack.unpackb(txn.get(key.encode('ascii')))
            ft = ft[:, :self.image_feat_size].astype(np.float32)
            self._feature_store[key] = ft
        return ft

class ImageFeaturesDB2(object):
    def __init__(self, img_ft_files, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_files
        self._feature_stores = {}
        for name in img_ft_files:
            self._feature_stores[name] = {}
            with h5py.File(name, 'r') as f:
                for key in f.keys():
                    ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                    self._feature_stores[name][key] = ft 
        self.env_names = list(self._feature_stores.keys())
        print(self.env_names)
        

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        env_name = random.choice(self.env_names)
        if key in self._feature_stores[env_name]:
            ft = self._feature_stores[env_name][key]
        else:
            with h5py.File(env_name, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_stores[env_name][key] = ft
        return ft

def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def new_simulator(connectivity_dir, scan_data_dir=None, width=640, height=480, vfov=60):
    import MatterSim

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(width, height)
    sim.setCameraVFOV(math.radians(vfov))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()
    #sim.init()

    return sim

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    for ix in range(36):
        # if ix == 0:
        #     sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        # elif ix % 12 == 0:
        #     sim.makeAction([0], [1.0], [1.0])
        # else:
        #     sim.makeAction([0], [1.0], [0])

        # state = sim.getState()[0]
        # assert state.viewIndex == ix

        # heading = state.heading - base_heading
        # elevation = state.elevation - base_elevation
        heading = (ix % 12) * math.radians(30) - base_heading
        elevation = (ix // 12 - 1) * math.radians(30) - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]

