
import sys
import time
import json
import random
from pathlib import Path
import numpy as np
from utils import util
from search.search_base import SearchBase


class SearchRandom(SearchBase):

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
        super().search(jps)

        k = self.prediction_limit

        best_result = sys.float_info.max
        best_params = None
        probabilities = jps.get_joint_prediction_probabilities(k)
        for i in range(self.budget):
            # Choose a random prediction with probabilities
            # limited to the k top predictions
            prediction_index = self.random_state.choice(self.pred_indices[:k], p=probabilities)

            # If this is the first time, try the default values first
            if i == 0:
                offset = 0
                rotation = 0
                flip = False
            else:
                offset_limit = self.cache[prediction_index]["offset_limit"]
                offset = self.random_state.uniform(-offset_limit, offset_limit)
                # offset = util.round_to_nearest(offset, 0.05)
                if self.cache[prediction_index]["skip_rotation"]:
                    rotation = 0
                else:
                    rotation = self.random_state.randint(0, 360)
                    # rotation = util.round_to_nearest(rotation, 5)
                flip = bool(self.random_state.randint(0, 2))
            # Get a transform from the random parameters
            transform = self.env.get_transform_from_parameters(
                jps,
                prediction_index=prediction_index,
                offset=offset,
                rotation_in_degrees=rotation,
                flip=flip,
                align_mat=self.cache[prediction_index]["align_mat"],
                origin2=self.cache[prediction_index]["origin2"],
                direction2=self.cache[prediction_index]["direction2"]
            )
            # Evaluate how optimal the parameters are by passing the transform
            # returns a value in the 0-1 range where lower is better
            result, overlap, contact = self.env.evaluate(jps, transform, eval_method=self.eval_method)
            if result < best_result:
                best_result = result
                best_params = {
                    "prediction_index": prediction_index,
                    "offset": offset,
                    "rotation": rotation,
                    "flip": flip,
                    "transform": transform,
                    "evaluation": result,
                    "overlap": overlap,
                    "contact": contact,
                }
        return best_params
