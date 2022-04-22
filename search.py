"""

JoinABLe Joint Pose Search

"""

import time
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
import torch
import numpy as np

from args import args_search
from train import JointPrediction
from utils.metrics import SearchMetrics
from search.search_random import SearchRandom
from search.search_simplex import SearchSimplex
from search.search_identity import SearchIdentity
from joint.joint_prediction_set import JointPredictionSet
from datasets.joint_graph_dataset import JointGraphDataset


def get_joint_files(input_dir):
    """Get json joint files that look like joint_set_00025.json"""
    assert input_dir.exists()
    pattern = "joint_set_[0-9][0-9][0-9][0-9][0-9].json"
    joint_files = [f for f in input_dir.glob(pattern)]
    return joint_files


def get_search_method(args):
    """Initialize the search method to use"""
    init_args = {
        "search_method": args.search_method,
        "eval_method": args.eval_method,
        "budget": args.budget,
        "prediction_limit": args.prediction_limit,
        "random_state": np.random.RandomState(args.seed)
    }
    if args.search_method == "random":
        # Random picking of parameters using the network probabilities to pick an axis
        return SearchRandom(**init_args)
    elif args.search_method == "simplex":
        # Nelder-Mead optimization to find the parameters (Ours + Search)
        return SearchSimplex(**init_args)
    elif args.search_method == "identity":
        # Use the default position without search (Ours)
        return SearchIdentity(**init_args)
    else:
        raise Exception("Invalid search method")


def search_joint_file(search, model, index, g1, g2, joint_graph, joint_file, args):
    """Perform search on a single joint file"""
    metrics = set(args.metrics.split(","))
    # Load the joint prediction set
    jps = JointPredictionSet(
        joint_file, g1, g2, joint_graph, model,
        load_bodies=False,
        seed=args.seed,
        num_samples=args.num_samples,
        prediction_limit=args.prediction_limit
    )
    holes = jps.joint_data.get("holes", [])
    has_holes = len(holes) > 0
    best_result = search.search(jps)

    # Check the Joint Axis Hits
    search_hit = None
    no_search_hit = None
    if "axis_hit" in metrics:
        # Get the brep entities from the best search prediction
        # and the top network prediction
        search_prediction_brep_indices = jps.get_joint_prediction_brep_index(best_result["prediction_index"])
        no_search_prediction_brep_indices = jps.get_joint_prediction_brep_index(0)
        # Get the ground truth brep entities including equivalents
        joint_brep_indices = jps.get_joint_brep_indices()
        # Check if the predictied indices are in the gt
        search_hit = search_prediction_brep_indices in joint_brep_indices
        no_search_hit = no_search_prediction_brep_indices in joint_brep_indices

    # Compare with Ground Truth
    iou = None
    cd = None
    if "iou" in metrics or "cd" in metrics:
        iou, cd = search.env.evaluate_vs_gt(
            jps,
            best_result["transform"],
            iou="iou" in metrics,
            cd="cd" in metrics,
            num_samples=args.num_samples
        )
    overlap = None
    if "overlap" in metrics and "overlap" in best_result:
        overlap = best_result["overlap"]
    contact = None
    if "contact" in metrics and "contact" in best_result:
        contact = best_result["contact"]
    return index, joint_file, search_hit, no_search_hit, iou, cd, overlap, contact, has_holes, best_result


def experiment_name(args):
    """Generate a name for the experiment based on given args"""
    tokens = [
        "JointPoseSearch",
        args.search_method,
        args.eval_method,
    ]
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tokens.append(timestamp)
    return "-".join(map(str, tokens))


def load_network(args):
    """Load the network using the command line args"""
    # First try to load the checkpoint file provided
    exp_dir = Path(args.exp_dir)
    exp_name_dir = exp_dir / args.exp_name
    checkpoint_file = exp_name_dir / f"{args.checkpoint}.ckpt"
    if not checkpoint_file.exists():
        # Fallback to the default checkpoint
        checkpoint_file = Path("pretrained/paper/last_run_0.ckpt")
        print(f"Using default checkpoint: {checkpoint_file}")
    else:
        print(f"Using checkpoint: {checkpoint_file}")
    if not checkpoint_file.exists():
        print("Checkpoint file does not exist")
        exit()
    model = JointPrediction.load_from_checkpoint(
        checkpoint_file,
        map_location=torch.device("cpu")
    )
    return model


def load_dataset(args):
    return JointGraphDataset(
        root_dir=args.dataset,
        split=args.test_split,
        limit=args.limit,
        threads=args.threads,
        label_scheme=args.test_label_scheme,
        max_node_count=0,  # Don't filter large graphs
        input_features=args.input_features
    )


def run_search(args, dataset, dataset_dir):
    """ Run Joint Pose Search by iterating over all joints """
    num_joint_files = len(dataset)
    search = get_search_method(args)
    # Class to keep track of the metrics
    sm = SearchMetrics(args, num_joint_files)
    # Load the model from the checkpoint
    model = load_network(args)

    print(f"Searching {num_joint_files} joints using {args.search_method} search with a budget of {args.budget} using up to {args.prediction_limit} (k) predictions...")
    for index in range(num_joint_files):
        g1, g2, joint_graph = dataset[index]
        joint_file = dataset_dir / dataset.files[index]
        search_result = search_joint_file(
            search, model, index, g1, g2, joint_graph, joint_file, args
        )
        sm.update(*search_result)
    return sm.summarize()


if __name__ == "__main__":
    args = args_search.get_args()
    start_time = time.time()
    # Check if we have the full data, rather than the precomputed data
    # i.e. all of the json and obj files
    dataset_dir = Path(args.dataset).resolve()
    joint_files = get_joint_files(dataset_dir)
    if len(joint_files) == 0:
        print(f"No joint json files found in {dataset_dir}")
        print("Please check you have downloaded the full dataset")
        exit()

    dataset = load_dataset(args)
    search_result = run_search(args, dataset, dataset_dir)
    print(f"Completed search in {time.time() - start_time:.2f} secs")
