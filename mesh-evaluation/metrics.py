# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
from pycg import vis, exp
from pykdtree.kdtree import KDTree


NAN_METRIC = float('nan')


def distance_p2p(points_src, points_tgt):
    kdtree = KDTree(points_tgt)
    dist, _ = kdtree.query(points_src)

    return dist

def get_threshold_percentage(dist, thresholds):
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold

class MeshEvaluator:

    ESSENTIAL_METRICS = [
        'chamfer-L1', 'f-score',
    ]
    ALL_METRICS = [
        'completeness', 'accuracy',
        'completeness2', 'accuracy2', 'chamfer-L2',
        'chamfer-L1', 'f-precision', 'f-recall', 'f-score', 'f-score-15', 'f-score-20'
    ]

    """
    Mesh evaluation class that handles the mesh evaluation process. Returned dict has meaning:
        - completeness:             mean distance from all gt to pd.
        - accuracy:                 mean distance from all pd to gt.
        - chamfer-l1/l2:            average of the above two. [Chamfer distance]
        - f-score(/-15/-20):        [F-score], computed at the threshold of 0.01, 0.015, 0.02.
    Args:
        n_points (int): number of points to be used for evaluation
    """

    def __init__(self, n_points=100000, metric_names=ALL_METRICS):
        self.n_points = n_points
        self.thresholds = np.array([0.01, 0.015, 0.02, 0.002, 0.1])
        self.fidx = [0, 1, 2, 3, 4]
        self.metric_names = metric_names

    def eval_mesh(self, pointcloud,  pointcloud_tgt):
        """
        Evaluates a mesh.
        :param pointcloud: np (Nx3) predicted xyz
        :param pointcloud_tgt: np (Nx3) ground-truth xyz
        :return: metric-dict
        """
        if isinstance(pointcloud_tgt, torch.Tensor):
            pointcloud_tgt = pointcloud_tgt.detach().cpu().numpy().astype(float)

        out_dict = self._evaluate(
            pointcloud, pointcloud_tgt)

        return out_dict

    def _evaluate(self, pointcloud, pointcloud_tgt):
        """
        Evaluates a point cloud.
        :param pointcloud: np (Mx3) predicted xyz
        :param pointcloud_tgt:  np (Nx3) ground-truth xyz
        :return: metric-dict
        """
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            exp.logger.warning('Empty pointcloud / mesh detected! Return NaN metric!')
            return {k: NAN_METRIC for k in self.metric_names}

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness = distance_p2p(
            pointcloud_tgt,  pointcloud
        )
        recall = get_threshold_percentage(completeness, self.thresholds)
        completeness2 = completeness ** 2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy = distance_p2p(
            pointcloud,pointcloud_tgt
        )
        precision = get_threshold_percentage(accuracy, self.thresholds)
        accuracy2 = accuracy ** 2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()

        # Chamfer distance
        chamfer_l2 = 0.5 * (completeness2 + accuracy2)
        chamfer_l1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            for i in range(len(precision))
        ]

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamfer_l2,
            'chamfer-L1': chamfer_l1,
            'f-precision': precision[self.fidx[0]],
            'f-recall': recall[self.fidx[0]],
            'f-score': F[self.fidx[0]],  # threshold = 1.0%
            'f-score-15': F[self.fidx[1]],  # threshold = 1.5%
            'f-score-20': F[self.fidx[2]],  # threshold = 2.0%
            # -- F-outdoor
            'f-precision-outdoor': precision[self.fidx[4]],
            'f-recall-outdoor': recall[self.fidx[4]],
            'f-score-outdoor': F[self.fidx[4]]
        }

        return {
            k: out_dict[k] for k in self.metric_names
        }
