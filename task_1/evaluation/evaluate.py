import argparse
import os
from sys import argv
from typing import Any, Dict, Optional

import numpy as np

from .adjacency_utils import edge_prediction_metrics_multisample


def load_submitted_adj(prediction_file: str) -> np.ndarray:
    """Loads submitted adjacency matrix from the given directory.

    Args:
        prediction_file (str): path to the submission file.

    Returns:
        np.ndarray: Loaded adjacency matrix of shape [num_datasets, num_matrices, variables, variables]
    """

    adj_matrices = np.load(prediction_file)

    assert adj_matrices.ndim == 4 or adj_matrices.ndim == 3, "Needs to be a 4D or 3D array"

    if adj_matrices.ndim == 3:
        adj_matrices = adj_matrices[:, np.newaxis, :, :]

    return adj_matrices.astype(bool)


def load_true_adj_matrix(reference_file: str) -> np.ndarray:
    """Loads true adjacency matrix from the given directory.
    
    Args:
        reference_file (str): Path to the ground truth file.
        
    Returns:
        np.ndarray: Loaded adjacency matrix of shape [num_datasets, variables, variables]
    """

    return np.load(reference_file).astype(bool)


def load_predicted_cate_estimate(prediction_file: str) -> np.ndarray:
    """Loads predicted CATE estimate from the given directory.
    
    Args:
        prediction_file (str): Path to the submission file.
        
    Returns:
        np.ndarray: Loaded CATE estimate of shape [num_datasets, num_interventions]
    """

    cate_predictions = np.load(prediction_file)

    assert cate_predictions.ndim == 2

    return cate_predictions


def load_true_cate_estimate(reference_file: str) -> np.ndarray:
    """Loads true CATE estimate from the given directory.
    
    Args:
        reference_file (str): Path to the ground truth file.
        
    Returns:
        np.ndarray: Loaded CATE estimate of shape [num_datasets, num_interventions]
        
    """

    cate_reference = np.load(reference_file)

    assert cate_reference.ndim == 2

    return cate_reference


def write_score_file(
    score_dir: str,
    score_dict: Dict[str, Any],
    summarise: bool = True,
    detailed_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """Writes the score dictionary to the given file.
    
    Args:
        score_dir (str): Dir to write the score dictionary to.
        score_dict (dict[str, Any]): Score dictionary to write.
        summarise (bool): Whether to summarise the score dictionary.
        detailed_dict (dict[str, Any]): Detailed score dictionary to write to HTML output.
    """

    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

    score_file = os.path.join(score_dir, "scores.txt")


    if summarise:
        avg_score = np.mean([v for v in score_dict.values()])

        with open(score_file, "w") as f:
            f.write(f"score: {avg_score}\n")

    else:
        with open(score_file, "w") as f:
            for key, value in score_dict.items():
                f.write(f"{key}: {value}\n")


def evaluate_adjacency(solution_file: str, prediction_file: str) -> Dict[str, Any]:
    """Evaluate adjacency matrices

    Args:
        solution_file (str): Directory with true solutions.
        prediction_file (str): Directory with predictions.

    Returns:
        Dict[str, Any]: Metrics dictionary
    """
    true_adj_matrix = load_true_adj_matrix(solution_file)
    submitted_adj_matrix = load_submitted_adj(prediction_file)

    assert true_adj_matrix.shape[0] == submitted_adj_matrix.shape[0], "Need to submit adjacency for each dataset."

    adj_metrics = []
    for i in range(true_adj_matrix.shape[0]):
        adj_metrics.append(edge_prediction_metrics_multisample(true_adj_matrix[i], submitted_adj_matrix[i]))

    return adj_metrics


def evaluate_cate(solution_file: str, prediction_file: str) -> Dict[str, Any]:
    """Evaluate CATE estimation.

    Args:
        solution_file (str): Path to true solutions.
        prediction_file (str): Path to submission.

    Returns:
        Dict[str, Any]: Metrics dictionary
    """
    true_adj_matrix = load_true_cate_estimate(solution_file)
    submitted_adj_matrix = load_predicted_cate_estimate(prediction_file)

    assert true_adj_matrix.shape == submitted_adj_matrix.shape, "Need to submit CATE for each dataset and intervention."

    # calculating RMSE across each intervention for each dataset.
    rmse = np.sqrt(np.mean(np.square(true_adj_matrix - submitted_adj_matrix), axis=1))

    return [{f"cate_rmse": rmse_slice} for rmse_slice in rmse]


def evaluate_and_write_scores(
    solution_file: str,
    prediction_file: str,
    score_dir: str,
    eval_adjacency: bool = True,
    eval_ate: bool = True,
    summarise: bool = True,
) -> None:
    """Run main evaluation and write scores to file.

    Args:
        solution_file (str): Ground truth file path.
        prediction_file (str): Submission file path
        score_dir (str): Directory to write scores to.
        evaluate_adjacency (bool, optional): Whether to evaluate adjacency matrix. Defaults to True.
        evaluate_ate (bool, optional): Whether to evalaute ATEs. Defaults to True.
        summarise (bool, optional): Whether to summarise scores. Defaults to True.
    """
    if summarise and eval_adjacency and eval_ate:
        raise Exception("Cannot evaluate adjacency and ATE at the same time.")

    score_dict = {}
    if eval_adjacency:
        detailed_metrics = evaluate_adjacency(solution_file, prediction_file)

        for i, adj_metric in enumerate(detailed_metrics):
            score_dict[f"orientation_fscore_dataset_{i}"] = adj_metric["orientation_fscore"]

    if eval_ate:
        detailed_metrics = evaluate_cate(solution_file, prediction_file)

        for i, ate_metric in enumerate(detailed_metrics):
            score_dict[f"cate_rmse_dataset_{i}"] = ate_metric["cate_rmse"]

    if summarise:
        # convert list of dicts to flat dict
        detailed_metrics_dict = {f"{k}_dataset_{i}": v for i, d in enumerate(detailed_metrics) for k, v in d.items()}
        write_score_file(
            score_dir,
            score_dict,
            summarise=True,
            detailed_dict=detailed_metrics_dict,
        )
    else:
        write_score_file(score_dir, score_dict, summarise=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate scripts for local development.")
    parser.add_argument("--input_file", type=str, help="Path to the submission file.")
    parser.add_argument("--ground_truth", type=str, help="Path to the ground truth file")
    parser.add_argument("--score_dir", type=str, help="Path to write the scores to")

    parser.add_argument("--evaluate_adjacency", action="store_true", help="Evaluate adjacency matrices for task 1.")
    parser.add_argument("--evaluate_cate", action="store_true", help="Evaluate CATE for task 2.")

    parser.add_argument("--summarise", action="store_true", help="Summarise scores.")

    args = parser.parse_args()

    solution_file = args.ground_truth
    prediction_file = args.input_file
    score_dir = args.score_dir

    evaluate_and_write_scores(
        solution_file,
        prediction_file,
        score_dir,
        args.evaluate_adjacency,
        args.evaluate_cate,
        args.summarise,
    )

