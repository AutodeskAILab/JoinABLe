
import sys
import time
import json
import random
from pathlib import Path
import scipy
import numpy as np
from tqdm import tqdm
from utils import util
from search.search_base import SearchBase


class SearchSimplex(SearchBase):

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
        # Make sure we don't go over the number of predictions we have
        k = min(k, len(self.pred_indices) - 1)
        best_result = sys.float_info.max
        best_params = None

        # Iterate over all prediction indices
        for prediction_index in range(k):
            # We optimize with flip on and off
            optimize_result = self.optimize(jps, prediction_index, False)
            optimize_flip_result = self.optimize(jps, prediction_index, True)
            # Then take the best result
            final_optimize_result = optimize_result
            flip = False
            if optimize_flip_result.fun < optimize_result.fun:
                final_optimize_result = optimize_flip_result
                flip = True

            if final_optimize_result.fun < best_result:
                offset = final_optimize_result.x[0]
                if self.cache[prediction_index]["skip_rotation"]:
                    rotation = 0
                else:
                    rotation = final_optimize_result.x[1]
                best_result = final_optimize_result.fun
                x = [offset, rotation]
                # Compute the transform again
                # from the parameters to keep track of it
                transform = self.get_transform_from_x(
                    x,
                    jps,
                    prediction_index,
                    flip,
                    self.cache[prediction_index]["align_mat"],
                    self.cache[prediction_index]["origin2"],
                    self.cache[prediction_index]["direction2"]
                )
                _, overlap, contact = self.env.evaluate(jps, transform, eval_method=self.eval_method)
                best_params = {
                    "prediction_index": prediction_index,
                    "offset": final_optimize_result.x[0],
                    "rotation": rotation,
                    "flip": flip,
                    "transform": transform,
                    "evaluation": best_result,
                    "overlap": overlap,
                    "contact": contact,
                }
        return best_params

    def optimize(self, jps, prediction_index, flip):
        """Run the optimization"""
        args = (
            jps,
            prediction_index,
            flip,
            self.cache[prediction_index]["align_mat"],
            self.cache[prediction_index]["origin2"],
            self.cache[prediction_index]["direction2"]
        )
        offset_limit = self.cache[prediction_index]["offset_limit"]
        if self.cache[prediction_index]["skip_rotation"]:
            bounds = scipy.optimize.Bounds([-offset_limit], [offset_limit])
            initial_guess = np.array([0])
        else:
            bounds = scipy.optimize.Bounds([-offset_limit, 0], [offset_limit, 360])
            initial_guess = np.array([0, 0])
        return scipy.optimize.minimize(
            self.cost_function,
            initial_guess,
            args,
            method="Nelder-Mead",
            # Note: Need scipy 1.7 otherwise:
            # Method Nelder-Mead cannot handle constraints nor bounds
            bounds=bounds,
            options={
                "disp": False,
                "maxiter": self.budget,
            }
        )

    def cost_function(self, x, jps, prediction_index, flip, align_mat, origin2, direction2):
        """Cost function for use with scipy.optimize"""
        transform = self.get_transform_from_x(x, jps, prediction_index, flip, align_mat, origin2, direction2)
        # Evaluate how optimal the parameters are by passing the transform
        # returns a value in the 0-1 range where lower is better
        result, overlap, contact = self.env.evaluate(jps, transform, eval_method=self.eval_method)
        return result

    def get_transform_from_x(self, x, jps, prediction_index, flip, align_mat, origin2, direction2):
        """Get the transform when given x containing the optimization parameters"""
        offset_limit = self.cache[prediction_index]["offset_limit"]
        # x increments by very tiny amounts
        # scale it to something sensible
        scale = 1500
        offset = x[0] * offset_limit * scale
        if len(x) == 2:
            rotation = np.rad2deg(x[1]) * scale
        else:
            rotation = 0
        return self.env.get_transform_from_parameters(
            jps,
            prediction_index=prediction_index,
            offset=offset,
            rotation_in_degrees=rotation,
            flip=flip,
            align_mat=align_mat,
            origin2=origin2,
            direction2=direction2
        )
