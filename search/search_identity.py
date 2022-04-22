
import sys
import time
import json
import random
from pathlib import Path
import numpy as np
from utils import util
from search.search_base import SearchBase


class SearchIdentity(SearchBase):

    def search(self, jps):
        """
        'Identity Search' that uses only the axis prediction
        without any actual search:
        - jps: Joint Prediction Set

        Returns a dict with the following fields:
        - prediction_index: Index of the prediction
        - offset: Offset parameter
        - rotation: Rotation parameter
        - flip: Flip parameter
        - transform: Transform created from the axis and parameters
        - evaluation: Evaluation score between 0-1, where lower is better
        - overlap: Overlap between the parts
        - contact: Contact between the parts
        """
        super().search(jps)
        prediction_index = 0
        offset = 0
        rotation = 0
        flip = False
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
        result, overlap, contact = self.env.evaluate(jps, transform, eval_method=self.eval_method)
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
