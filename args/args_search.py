"""

JoinABLe search args

"""

from args import args_train


def get_parser():
    """Return the search args parser"""
    parser = args_train.get_parser()
    parser.add_argument(
        "--search_method",
        type=str,
        default="random",
        choices=["random", "simplex", "identity"],
        help="Search method to use [default: simplex] can be one of:\
             - random: Random picking of parameters using the network probabilities to pick an axis\
             - simplex: Nelder-Mead optimization to find the parameters (Ours + Search)\
             - identity: Use the default position without search (Ours)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="The number of steps to search, when using random search [default: 100]"
    )
    parser.add_argument(
        "--prediction_limit",
        type=int,
        default=50,
        help="Limit searching to the top k predictions [default: 50]"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4096,
        help="The number of point cloud samples to use [default: 4096]"
    )
    parser.add_argument(
        "--eval_method",
        type=str,
        default="default",
        choices=["default", "smooth"],
        help="Method used for evaluation"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="axis_hit,cd",
        help="Metrics to log results for as a comma separated string containing one of:\
             (iou, cd, axis_hit, overlap, contact)"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="File to save search results to"
    )
    return parser


def get_args():
    """Return the search args"""
    parser = get_parser()
    args = parser.parse_args()
    return args