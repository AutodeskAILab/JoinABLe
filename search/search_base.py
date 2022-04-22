
from random import random
import time
import json
import random
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from joint.joint_environment import JointEnvironment


class SearchBase:

    def __init__(
        self,
        search_method=None,
        eval_method="default",
        budget=100,
        prediction_limit=50,
        random_state=None
    ):
        """
        - search_method: Method to use for search
        - eval_method: Evaluation method passed to the cost function to determine the best solution
        - budget: Budget used to constrain the search
        - prediction_limit: Limit search to the top k predictions
        - random_state: Numpy random state to use
        """
        self.env = JointEnvironment()
        self.cache = []
        self.search_method = search_method
        self.eval_method = eval_method
        self.budget = budget
        self.prediction_limit = prediction_limit

        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

    def search(self, jps):
        """
        Search for the best solution using the given:
        - jps: Joint Prediction Set

        Returns a dict with the following fields:
        - prediction_index: Index of the prediction
        - offset: Offset parameter
        - rotation: Rotation parameter
        - flip: Flip parameter
        - transform: Transform created from the axis and parameters
        - evaluation: Evaluation score from the cost function where lower is better
        - overlap: Overlap between the parts
        - contact: Contact between the parts
        """
        self.load_cache(jps, self.prediction_limit)

    def load_cache(self, jps, prediction_limit):
        """
        Precalculate and cache some values for each prediction
        To avoid doing it multiple times for large budgets
        """
        self.pred_indices = jps.get_joint_prediction_indices(prediction_limit)
        self.pred_probs = jps.get_joint_prediction_probabilities(prediction_limit)
        self.cache = []

        for prediction_index in self.pred_indices:
            # Cache the alignment matrix, origin, and direction
            align_mat, origin2, direction2 = self.env.get_joint_alignment_matrix(jps, prediction_index)
            # Precalculate the offset limit so we search within a sensible range
            offset_limit = self.get_offset_limit(jps, prediction_index)
            # Check if body 1 has rotational symmetry
            # allowing us to skip rotation
            is_symmetric = jps.is_joint_body_rotationally_symmetric(joint_axis_direction=direction2)

            self.cache.append({
                "offset_limit": offset_limit,
                "align_mat": align_mat,
                "origin2": origin2,
                "direction2": direction2,
                "skip_rotation": is_symmetric
            })

    def get_offset_limit(self, jps, prediction_index, use_convex_hull=False):
        """Get the offset limit to constraint the search with"""
        if use_convex_hull:
            # We first try and calculate the distance of the axis through a convex hull
            # of each body
            ch_locs1 = jps.get_joint_prediction_axis_convex_hull_intersections(1, prediction_index)
            ch_locs2 = jps.get_joint_prediction_axis_convex_hull_intersections(2, prediction_index)
            if ch_locs1 is not None and ch_locs2 is not None:
                ch_dist1 = np.linalg.norm(ch_locs1[0] - ch_locs1[1])
                ch_dist2 = np.linalg.norm(ch_locs2[0] - ch_locs2[1])
                # Use the combined length to limit the range
                # of random sampling of the offset parameter
                return abs(ch_dist1) + abs(ch_dist2)
            else:
                use_convex_hull = False
        if not use_convex_hull:
            # Here we fall back to calculating the length of the part of the axis
            # that is passing through the axis aligned bounding box of each body
            tmin1, tmax1 = jps.get_joint_prediction_axis_aabb_intersections(1, prediction_index)
            tmin2, tmax2 = jps.get_joint_prediction_axis_aabb_intersections(2, prediction_index)

            # If the ray is missing the box
            # Use the length of the longest dimension
            if (tmin1 is None and tmax1 is None) or (math.isinf(tmin1) and math.isinf(tmax1)):
                length1 = np.max(jps.body_one_mesh.extents)
            else:
                length1 = abs(tmax1 - tmin1)
            if (tmin2 is None and tmax2 is None) or (math.isinf(tmin2) and math.isinf(tmax2)):
                length2 = np.max(jps.body_two_mesh.extents)
            else:
                length2 = abs(tmax2 - tmin2)
            # Use the combined length to limit the range
            # of random sampling of the offset parameter
            return length1 + length2
