import json
import numpy as np
from pathlib import Path
import torch
from scipy.spatial.distance import cdist


class SearchMetrics:
    """
    Store search related metrics
    """
    def __init__(self, args, num_joint_files):
        self.args = args
        self.metrics = set(args.metrics.split(","))
        self.num_joint_files = num_joint_files
        # Joint axis hits after search
        self.search_hit_count = 0
        # Joint axis misses after search
        self.no_search_hit_count = 0
        # Intersection over Union
        self.ious = []
        # IoU of samples with a joint axis hit
        self.hit_ious = []
        # Chamfer distance
        self.cds = []
        # Chamfer distance of samples with holes
        self.cds_holes = []
        # Chamfer distance of samples without holes
        self.cds_no_holes = []
        # Chamfer distance of samples with a joint axis hit
        self.hit_cds = []
        # Samples where the pair of parts overlap
        self.overlaps = []
        # Samples where the pair of parts are in contact
        self.contacts = []
        # Results data to save to file
        self.results_data = {}
        self.results_file = None
        if args.results_file:
            self.results_file = Path(args.results_file)

    def mean_str(self, array):
        """Get a string of the numpy mean result"""
        if len(array) == 0:
            return "--"
        else:
            return f"{np.mean(array):.4f}"

    def median_str(self, array):
        """Get a string of the numpy median result"""
        if len(array) == 0:
            return "--"
        else:
            return f"{np.median(array):.4f}"

    def std_str(self, array):
        """Get a string of the numpy std result"""
        if len(array) == 0:
            return "--"
        else:
            return f"{np.std(array):.4f}"

    def update(self, index, joint_file, search_hit, no_search_hit, iou, cd, overlap, contact, has_holes, best_result):
        """Update the search metrics"""
        log_axis_hit = ""
        log_iou = ""
        log_iou_hit = ""
        log_cd = ""
        log_cd_hit = ""
        log_overlap = ""
        log_has_overlap = ""
        log_contact = ""
        log_has_contact = ""
        if "axis_hit" in self.metrics:
            self.search_hit_count += search_hit
            self.no_search_hit_count += no_search_hit
            hit_improvement = (self.search_hit_count / (index + 1) - self.no_search_hit_count / (index + 1)) * 100.0
            log_axis_hit = f"Search Hits: {self.search_hit_count} vs {self.no_search_hit_count} ({hit_improvement:+.2f}%) |"
            if "iou" in self.metrics:
                if search_hit:
                    self.hit_ious.append(iou)
                avg_hit_iou = self.mean_str(self.hit_ious)
            if "cd" in self.metrics:
                if search_hit:
                    self.hit_cds.append(cd)
        if "iou" in self.metrics:
            self.ious.append(iou)
            log_iou = f"Avg IoU: {self.mean_str(self.ious)} Median IoU: {self.median_str(self.ious)}"
        if "cd" in self.metrics:
            self.cds.append(cd)
            log_cd = f"Avg CD: {self.mean_str(self.cds)} Median CD: {self.median_str(self.cds)}"
            if has_holes:
                self.cds_holes.append(cd)
            else:
                self.cds_no_holes.append(cd)
        if "overlap" in self.metrics:
            self.overlaps.append(overlap)
            has_overlap = (np.array(self.overlaps) > 0.0).sum()
            has_overlap_percent = has_overlap / len(self.overlaps) * 100.0
            log_overlap = f"Avg Overlap: {self.mean_str(self.overlaps)} Median Overlap: {self.median_str(self.overlaps)}"
            log_has_overlap = f"Has Overlap: {has_overlap_percent:.4f}%"
        if "contact" in self.metrics:
            self.contacts.append(contact)
            has_contact = (np.array(self.contacts) > 0.0).sum()
            has_contact_percent = has_contact / len(self.contacts) * 100.0
            log_contact = f"Avg Contact: {self.mean_str(self.contacts)} Median Contact: {self.median_str(self.contacts)}"
            log_has_contact = f"Has Contact: {has_contact_percent:.4f}%"

        log_progress = f"[{index + 1}/{self.num_joint_files}]"
        log_output = f"{log_progress} {joint_file.stem} | {log_axis_hit} {log_iou} {log_cd} {log_overlap} {log_has_overlap} {log_contact} {log_has_contact}"
        print(log_output)

        # Save the results to file if requested
        if self.results_file is not None:
            self.results_data[joint_file.stem] = {}
            self.results_data[joint_file.stem]["prediction_index"] = int(best_result["prediction_index"])
            self.results_data[joint_file.stem]["offset"] = float(best_result["offset"])
            self.results_data[joint_file.stem]["rotation"] = float(best_result["rotation"])
            self.results_data[joint_file.stem]["flip"] = bool(best_result["flip"])
            self.results_data[joint_file.stem]["transform"] = best_result["transform"].tolist()
            self.results_data[joint_file.stem]["evaluation"] = float(best_result["evaluation"])
            self.results_data[joint_file.stem]["overlap"] = float(best_result["overlap"])
            self.results_data[joint_file.stem]["contact"] = float(best_result["contact"])
            self.results_data[joint_file.stem]["has_holes"] = has_holes
            if "axis_hit" in self.metrics:
                self.results_data[joint_file.stem]["search_hit"] = search_hit
                self.results_data[joint_file.stem]["no_search_hit"] = no_search_hit
            if "iou" in self.metrics:
                self.results_data[joint_file.stem]["iou"] = iou
            if "cd" in self.metrics:
                self.results_data[joint_file.stem]["cd"] = cd
            if "overlap" in self.metrics:
                self.results_data[joint_file.stem]["overlap"] = overlap
            with open(self.results_file, "w", encoding="utf8") as f:
                json.dump(self.results_data, f, indent=4)

    def summarize(self):
        """Summarize the final search metrics"""
        print("\nJOINT POSE SEARCH RESULTS")
        print("-----------------")
        results = {
            "num_joint_files": self.num_joint_files
        }
        if "axis_hit" in self.metrics:
            search_hit_percent = (self.search_hit_count / self.num_joint_files)
            no_search_hit_percent = (self.no_search_hit_count / self.num_joint_files)
            print(f"Top-1 with Search: {self.search_hit_count}/{self.num_joint_files} ({search_hit_percent * 100.0:.4f}%)")
            print(f"Top-1 without Search: {self.no_search_hit_count}/{self.num_joint_files} ({no_search_hit_percent * 100.0:.4f}%)")
            results["search_hit_count"] = self.search_hit_count
            results["no_search_hit_count"] = self.search_hit_count
            results["search_hit_percent"] = search_hit_percent
            results["no_search_hit_percent"] = no_search_hit_percent
        if "iou" in self.metrics:
            avg_iou = self.mean_str(self.ious)
            median_iou = self.median_str(self.ious)
            print(f"Average IoU: {avg_iou}")
            print(f"Median IoU: {median_iou}")
            results["avg_iou"] = avg_iou
            results["median_iou"] = median_iou
            if "axis_hit" in self.metrics:
                avg_hit_iou = self.mean_str(self.hit_ious)
                median_hit_iou = self.median_str(self.hit_ious)
                print(f"Average Hit IoU: {avg_hit_iou}")
                print(f"Median Hit IoU: {median_hit_iou}")
                results["avg_hit_iou"] = avg_hit_iou
                results["median_hit_iou"] = median_hit_iou
        if "cd" in self.metrics:
            avg_cd = self.mean_str(self.cds)
            print(f"Average CD: {avg_cd}")
            results["avg_cd"] = avg_cd
            median_cd = self.median_str(self.cds)
            print(f"Median CD: {median_cd}")
            results["median_cd"] = median_cd
            stdev_cd = self.std_str(self.cds)
            print(f"Std Dev CD: {stdev_cd}")
            results["stdev_cd"] = stdev_cd
            #
            avg_cd_holes = self.mean_str(self.cds_holes)
            print(f"\tAverage CD Holes: {avg_cd_holes}")
            results["avg_cd_holes"] = avg_cd_holes
            median_cd_holes = self.median_str(self.cds_holes)
            print(f"\tMedian CD Holes: {median_cd_holes}")
            results["median_cd_holes"] = median_cd_holes
            stddev_cd_holes = self.std_str(self.cds_holes)
            print(f"\tStd Dev CD Holes: {stddev_cd_holes}")
            results["stddev_cd_holes"] = stddev_cd_holes
            #
            avg_cd_no_holes = self.mean_str(self.cds_no_holes)
            print(f"\tAverage CD No Holes: {avg_cd_no_holes}")
            results["avg_cd_no_holes"] = avg_cd_no_holes
            median_cd_no_holes = self.median_str(self.cds_no_holes)
            print(f"\tMedian CD No Holes: {median_cd_no_holes}")
            results["median_cd_no_holes"] = median_cd_no_holes
            stddev_cd_no_holes = self.std_str(self.cds_no_holes)
            print(f"\tStd Dev CD No Holes: {stddev_cd_no_holes}")
            results["stddev_cd_no_holes"] = stddev_cd_no_holes
            if "axis_hit" in self.metrics:
                avg_hit_cd = self.mean_str(self.hit_cds)
                median_hit_cd = self.median_str(self.hit_cds)
                print(f"Average Hit CD: {avg_hit_cd}")
                print(f"Median Hit CD: {median_hit_cd}")
                results["avg_hit_cd"] = avg_hit_cd
                results["median_hit_cd"] = median_hit_cd
        if "overlap" in self.metrics:
            avg_overlap = self.mean_str(self.overlaps)
            median_overlap = self.median_str(self.overlaps)
            has_overlap = (np.array(self.overlaps) > 0.0).sum()
            has_overlap_percent = has_overlap / len(self.overlaps)
            print(f"Average Overlap: {avg_overlap}")
            print(f"Median Overlap: {median_overlap}")
            print(f"Has Overlap: {has_overlap_percent * 100.0:.4f}%")
            results["avg_overlap"] = avg_overlap
            results["median_overlap"] = median_overlap
            results["has_overlap"] = has_overlap
            results["has_overlap_percent"] = has_overlap_percent
        if "contact" in self.metrics:
            avg_contact = self.mean_str(self.contacts)
            median_contact = self.median_str(self.contacts)
            has_contact = (np.array(self.contacts) > 0.0).sum()
            has_contact_percent = has_contact / len(self.contacts)
            print(f"Average Contact: {avg_contact}")
            print(f"Median Contact: {median_contact}")
            print(f"Has Contact: {has_contact_percent * 100.0:.4f}%")
            results["avg_contact"] = avg_contact
            results["median_contact"] = median_contact
            results["has_contact"] = has_contact
            results["has_contact_percent"] = has_contact_percent
        print("-----------------")
        return results


def hit_at_top_k(logits, labels, k=1):
    """
    Hit or not with the top k highest probability predictions
    or a range of k predictions if k is an array
    Assumes batch size of 1
    """
    logits_flat = logits.flatten()
    labels_flat = labels.flatten()
    # If k is an array use the highest value
    use_range = isinstance(k, np.ndarray) or isinstance(k, list)
    if use_range:
        max_k = k[-1]
    else:
        max_k = k
    # We may have fewer elements than the k requested
    max_k = min(max_k, logits.shape[0])
    top_k_values, top_k_indices = torch.topk(logits_flat, k=max_k)

    if use_range:
        # Return multiple hit results for each element in k
        k_results = np.zeros(len(k))
        for index, k_limit in enumerate(k):
            k_limit = min(k_limit, logits.shape[0])
            top_k_limit_indices = top_k_indices[:k_limit]
            top_k_limit_labels = labels_flat[top_k_limit_indices]
            max_label = top_k_limit_labels.max()
            k_results[index] = bool(max_label == 1)
        return k_results
    else:
        # Return a single hit result
        top_k_labels = labels_flat[top_k_indices]
        max_label = top_k_labels.max()
        return bool(max_label == 1)


def calculate_precision_at_k_from_sequence(precision_at_k, use_percent=True):
    """
    Given a sequence of hit results for multiple data samples
    calculate the precision at k percentage for each k
    """
    # The columns here are increasing k
    # and the rows are the results for each sample
    if isinstance(precision_at_k, list):
        precision_at_k = np.array(precision_at_k)
    precision_at_k_hits = np.sum(precision_at_k, axis=0)
    precision_at_k_hit_total = precision_at_k.shape[0]
    precision_at_k_hit = (precision_at_k_hits / precision_at_k_hit_total)
    if use_percent:
        return precision_at_k_hit * 100.0
    else:
        return precision_at_k_hit


def get_k_sequence():
    """ Get the sequence of k values to log"""
    # 1, 2 ... 5, 10, 20 ... 100
    return list(range(1, 6, 1)) + list(range(10, 110, 10))
