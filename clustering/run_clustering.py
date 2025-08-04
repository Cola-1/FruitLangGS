# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import copy
from typing import Union
import open3d as o3d
import numpy as np
from pathlib import Path
import alphashape
import trimesh
from shapely.geometry import GeometryCollection

from clustering_base import (load_obj_file, FruitClustering)


class Clustering(FruitClustering):
    def __init__(self,
                 template_path: Union[str, Path] = './clustering/apple_template.ply',
                 voxel_size_down_sample: float = 0.00005,
                 remove_outliers_nb_points: int = 800,
                 remove_outliers_radius: float = 0.02,
                 min_samples: int = 60,
                 apple_template_size: float = 0.8,
                 cluster_merge_distance: float = 0.04,
                 gt_cluster=None,
                 gt_count: int = None):
        super().__init__(voxel_size_down_sample=voxel_size_down_sample,
                         remove_outliers_nb_points=remove_outliers_nb_points,
                         remove_outliers_radius=remove_outliers_radius,
                         cluster_merge_distance=cluster_merge_distance)
        self.template_path = template_path
        self.min_samples = min_samples

        self.fruit_template = o3d.io.read_point_cloud(self.template_path)
        self.fruit_template = self.fruit_template.scale(apple_template_size, center=(0, 0, 0))
        self.fruit_template = self.fruit_template.translate(-self.fruit_template.get_center())
        self.fruit_alpha_shape_ = alphashape.alphashape(np.asarray(self.fruit_template.points), 10)

        if isinstance(self.fruit_alpha_shape_, GeometryCollection):
            raise ValueError(f"Alpha shape construction failed; check template: {self.template_path}")
        elif isinstance(self.fruit_alpha_shape_, trimesh.Trimesh):
            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(self.fruit_alpha_shape_.vertices),
                triangles=o3d.utility.Vector3iVector(self.fruit_alpha_shape_.faces)
            )
            self.fruit_alpha_shape = mesh.sample_points_uniformly(number_of_points=1000)
        elif hasattr(self.fruit_alpha_shape_, "as_open3d"):
            self.fruit_alpha_shape = self.fruit_alpha_shape_.as_open3d.sample_points_uniformly(1000)
        else:
            raise TypeError(f"Unsupported alpha shape type: {type(self.fruit_alpha_shape_)}")

        self.gt_cluster = gt_cluster
        if self.gt_cluster:
            if "obj" in self.gt_cluster:
                self.gt_mesh, self.gt_cluster_center, self.gt_cluster_pcd = load_obj_file(gt_cluster)
                self.gt_position = o3d.geometry.PointCloud()
                self.gt_position.points = o3d.utility.Vector3dVector(np.vstack(self.gt_cluster_center))
            else:
                self.gt_position = o3d.io.read_line_set(self.gt_cluster)
        self.gt_count = gt_count

        self.true_positive = None
        self.precision = None
        self.recall = None
        self.F1 = None


if __name__ == '__main__':
   

    from config_real import (Baum_01_unet, Baum_01_unet_Big, Baum_01_SAM, Baum_01_SAM_Big)
    from config_real import Fuji_unet, Fuji_unet_big, Fuji_sam, Fuji_sam_big

    Baums = [Fuji_sam]

    results = {}

    for Baum in Baums:
        clustering = Clustering(remove_outliers_nb_points=Baum['remove_outliers_nb_points'],
                                  remove_outliers_radius=Baum['remove_outliers_radius'],
                                  voxel_size_down_sample=Baum['down_sample'],
                                  template_path=Baum['template_path'],
                                  min_samples=Baum['min_samples'],
                                  apple_template_size=Baum['apple_template_size'],
                                  gt_cluster=Baum['gt_cluster'],
                                  cluster_merge_distance=Baum['cluster_merge_distance'],
                                  gt_count=Baum['gt_count'])
        count = clustering.count(pcd=Baum["path"], eps=Baum['eps'])

        entry = {
            'count': count,
            'gt': clustering.gt_count,
        }
        if Baum['gt_cluster'] and hasattr(clustering, 'true_positive'):
            entry.update({
                'TP': clustering.true_positive,
                'precision': clustering.precision,
                'recall': clustering.recall,
                'F1': clustering.F1,
            })
        results[Baum['path']] = entry

        print(results)
        print("\n --------------------------------- \n")
    print(results)

    import json
    with open('results_synthetic.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4, default=str)
