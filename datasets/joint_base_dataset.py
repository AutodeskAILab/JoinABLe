import pickle
import json
import random
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils import util


class JointBaseDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        random_rotate=False,
        delete_cache=False,
        limit=0,
        threads=1,
        shuffle_split=False,
        seed=42
    ):
        """
        Base class to load different versions of the Fusion 360 Gallery joints dataset
        :param root_dir: Root path to the dataset
        :param split: string Either train, val, test, mix_test, or all set
        :param random_rotate: bool Randomly rotate the point features
        :param delete_cache: bool Delete the cached pickle files
        :param limit: int Limit the number of joints to load to this number
        :param threads: Number of threads to use for data loading
        :param shuffle_split: Shuffle the files within a split when loading from json data
        :param seed: Random seed to use
        """
        if isinstance(root_dir, Path):
            self.root_dir = root_dir
        else:
            self.root_dir = Path(root_dir)
        assert split in ("train", "val", "validation", "test", "mix_test", "all")
        self.split = split
        self.shuffle_split = shuffle_split
        self.random_rotate = random_rotate
        self.threads = threads
        self.seed = seed
        # Limit the number of files to load
        self.limit = limit
        # Delete the cache
        self.delete_cache = delete_cache
        # Limit the number of files to load from cache
        self.cache_limit = limit
        # The cache file
        self.cache_file = self.root_dir / f"{self.split}.pickle"

        # The number of files in the original dataset split
        self.original_file_count = 0
        # The joint set files
        self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # To be implemented by the child class
        pass

    def setup_cache(self):
        """Setup the cache, either deleting or loading it"""
        cache_loaded = False
        # Delete the cache if requested
        if self.delete_cache:
            self.delete_data_cache()
        else:
            # Try to load from the cache to save time
            cache_loaded = self.load_data_cache()
            if cache_loaded:
                cache_loaded = True
        return cache_loaded

    def save_data_cache(self, data=None):
        """Save a pickle of the data"""
        if not data:
            return
        data["files"] = self.files
        data["original_file_count"] = self.original_file_count
        with open(self.cache_file, "wb") as f:
            pickle.dump(data, f)
        if self.cache_file.exists():
            print(f"Data cache written to: {self.cache_file}")
        else:
            print(f"Data cache failed to be written: {self.cache_file}")

    def load_data_cache(self):
        """Load a pickle of the data"""
        if not self.cache_file.exists():
            print("No data cache available")
            return False
        with open(self.cache_file, "rb") as f:
            data = pickle.load(f)

        if self.limit > 0:
            self.cache_limit = self.limit
        else:
            self.cache_limit = len(data["files"])
        self.files = data["files"][:self.cache_limit]
        if "original_file_count" in data:
            self.original_file_count = data["original_file_count"]
        # Return the data directly so the child class
        # can further process it
        return data

    def delete_data_cache(self):
        """Delete the cache pickle file"""
        cache_file = self.root_dir / f"{self.split}.pickle"
        if cache_file.exists():
            cache_file.unlink()
            print(f"Data cache deleted from: {cache_file}")
        else:
            print(f"No data cache to delete from: {cache_file}")

    def get_joint_files(self):
        """Get the joint files to load"""
        all_joint_files = self.get_all_joint_files()
        # Create the train test split
        joint_files = self.get_split(all_joint_files)
        # Using only a subset of files
        if self.limit > 0:
            joint_files = joint_files[:self.limit]
        # Store the original file count
        # to keep track of the number of files we filter
        # from the official train/test split
        self.original_file_count = len(joint_files)
        print(f"Loading {len(joint_files)} {self.split} data")
        return joint_files

    def get_all_joint_files(self):
        """Get all the json joint files that look like joint_set_00025.json"""
        pattern = "joint_set_[0-9][0-9][0-9][0-9][0-9].json"
        return [f.name for f in Path(self.root_dir).glob(pattern)]

    def get_split(self, all_joint_files):
        """Get the train/test split"""
        # First check if we have the official split in the dataset dir
        split_file = self.root_dir / "train_test.json"
        # Look in the parent directory too if we can't find it
        if not split_file.exists():
            split_file = self.root_dir.parent / "train_test.json"
        if split_file.exists():
            print("Using official train test split")
            train_joints = []
            val_joints = []
            test_joints = []
            with open(split_file, encoding="utf8") as f:
                official_split = json.load(f)
            if self.split == "train":
                joint_files = official_split["train"]
            elif self.split == "val" or self.split == "validation":
                joint_files = official_split["validation"]
            elif self.split == "test":
                joint_files = official_split["test"]
            elif self.split == "mix_test":
                if "mix_test" not in official_split:
                    raise Exception("Mix test split missing")
                else:
                    joint_files = official_split["mix_test"]
            elif self.split == "all":
                joint_files = []
                for split_files in official_split.values():
                    joint_files.extend(split_files)
            else:
                raise Exception("Unknown split name")
            joint_files = [f"{f}.json" for f in joint_files]
            if self.shuffle_split:
                random.Random(self.seed).shuffle(joint_files)
            return joint_files
        else:
            # We don't have an official split, so we make one
            print("Using new train test split")
            if self.split != "all":
                trainval_joints, test_joints = train_test_split(
                    all_joint_files, test_size=0.2, random_state=self.seed,
                )
                train_joints, val_joints = train_test_split(
                    trainval_joints, test_size=0.25, random_state=self.seed + self.seed,
                )
            if self.split == "train":
                joint_files = train_joints
            elif self.split == "val" or self.split == "validation":
                joint_files = val_joints
            elif self.split == "test":
                joint_files = test_joints
            elif self.split == "all":
                joint_files = all_joint_files
            else:
                raise Exception("Unknown split name")
            return joint_files

    def get_random_rotation(self):
        """Get a random rotation in 90 degree increments along the canonical axes"""
        axes = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        axis = random.choice(axes)
        angle_radians = np.radians(random.choice(angles))
        return Rotation.from_rotvec(angle_radians * axis)

    @staticmethod
    def get_joint_transforms(joint_data):
        """Return a list of transforms for each joint body"""
        transforms = []
        joints = joint_data["joints"]
        for joint in joints:
            # Transforms as 4x4 affine matrix
            aff_mat1 = util.transform_to_np(joint["geometry_or_origin_one"]["transform"])
            aff_mat2 = util.transform_to_np(joint["geometry_or_origin_two"]["transform"])
            transforms.append((aff_mat1, aff_mat2))
        return transforms

    @staticmethod
    def transforms_to_trans_rots(transforms):
        """Convert transforms to a translation point and rotation quaternion"""
        # Shape is the number of joint transforms (n)
        # by the size of a point (3) and quaternion (4)
        # (n, 3) and (n, 4)
        trans = torch.zeros((len(transforms), 3), dtype=torch.float)
        rots = torch.zeros((len(transforms), 4), dtype=torch.float)
        for i, aff_mat1 in enumerate(transforms):
            t1, q1 = util.matrix_to_trans_rot(aff_mat1)
            trans[i] = t1
            rots[i] = q1
        return trans, rots

    @staticmethod
    def get_joint_parameters(joint_data, scale):
        """Get the parameters for each joint
            Returns a tensor of shape (num_joints, 3)
            with (offset, rotation, flip) as floats
        """
        num_joints = len(joint_data["joints"])
        params = torch.zeros(num_joints, 3, dtype=torch.float)
        for joint_index, joint in enumerate(joint_data["joints"]):
            params[joint_index][0] = joint["offset"]["value"] * scale
            params[joint_index][1] = joint["angle"]["value"]
            params[joint_index][2] = float(joint["is_flipped"])
        return params

    @staticmethod
    def get_center_from_bounding_box(bbox):
        """Get the center from the bounding box
            to be used with data normalization"""
        max_pt = bbox["max_point"]
        min_pt = bbox["min_point"]
        span_x = max_pt["x"] - min_pt["x"]
        span_y = max_pt["y"] - min_pt["y"]
        span_z = max_pt["z"] - min_pt["z"]
        center = np.array(
            [
                max_pt["x"] - (span_x * 0.5),
                max_pt["y"] - (span_y * 0.5),
                max_pt["z"] - (span_z * 0.5),
            ]
        )
        return center
