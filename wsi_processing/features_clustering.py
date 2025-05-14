import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
import logging
from .utils import dump_json


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


logger = logging.getLogger(__name__)


def clustering(feats, num_clusters, filepath=None, save_npz=True):
    """
    Apply KMeans clustering to feature vectors.

    Args:
        feats (np.ndarray): Input feature array (N x D).
        num_clusters (int): Number of clusters to form.
        filepath (str or Path, optional): Path to save .npz output.
        save_npz (bool): Whether to save the clustering result.

    Returns:
        np.ndarray: Cluster assignments (N x 1).
    """
    k_means = KMeans(n_clusters=num_clusters, random_state=985).fit(feats)
    features_cluster_indices = np.expand_dims(k_means.labels_, axis=1)

    if filepath is not None and save_npz:
        np.savez(file=filepath, features_cluster_indices=features_cluster_indices)

    return features_cluster_indices


def save_to_json(features_cluster_indices, num_clusters, filepath=None, save_json=True):
    """
    Convert cluster indices to JSON and optionally save.

    Args:
        features_cluster_indices (np.ndarray): Cluster labels for each patch.
        num_clusters (int): Total number of clusters.
        filepath (str or Path, optional): Path to save JSON file.
        save_json (bool): Whether to save the JSON output.

    Returns:
        List[List[int]]: List of patch indices per cluster.
    """
    cluster_features = [[] for _ in range(num_clusters)]
    for patch_idx, cluster_idx in enumerate(features_cluster_indices):
        cluster_features[cluster_idx.item()].append(patch_idx)

    if filepath is not None and save_json:
        dump_json(cluster_features, filepath)

    return cluster_features


def run(args):
    """
    Main processing logic to apply clustering to all .npz files.
    """
    save_dir = Path(args.feat_dir) / f'k-means-{args.num_clusters}'
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Clustering output will be saved to: {save_dir}")

    img_features_npz = sorted(list(Path(args.feat_dir).glob('*.npz')))
    if not img_features_npz:
        logger.warning(f"No .npz files found in {args.feat_dir}")
        return

    for i, feat_npz in enumerate(tqdm(img_features_npz, desc="Clustering files")):
        case_id = feat_npz.stem
        npz_filepath = save_dir / f'{case_id}.npz'
        json_filepath = save_dir / f'{case_id}.json'

        if npz_filepath.exists() and not args.exist_ok:
            logger.warning(f"{npz_filepath.name} already exists, skipping...")
            continue

        try:
            feat_dict = np.load(str(feat_npz))
            features = feat_dict['img_features']
        except Exception as e:
            logger.error(f"Failed to load {feat_npz.name}: {e}")
            continue

        if features.shape[0] < args.num_clusters:
            logger.warning(f"{case_id}: {features.shape[0]} features < {args.num_clusters} clusters. Skipping.")
            continue

        features_cluster_indices = clustering(
            feats=features,
            num_clusters=args.num_clusters,
            filepath=npz_filepath,
            save_npz=not args.no_npz
        )

        save_to_json(
            features_cluster_indices=features_cluster_indices,
            num_clusters=args.num_clusters,
            filepath=json_filepath,
            save_json=not args.no_json
        )

        logger.info(f"[{i+1}/{len(img_features_npz)}] Processed {case_id} with {features.shape[0]} features")


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True,
                        help="Directory containing feature files (.npz)")
    parser.add_argument('--num_clusters', type=int, default=10,
                        help="Number of KMeans clusters")
    parser.add_argument('--exist_ok', action='store_true', default=False,
                        help="Allow overwriting existing clustering results")
    parser.add_argument('--no_json', action='store_true', default=False,
                        help="Do not save cluster output as JSON")
    parser.add_argument('--no_npz', action='store_true', default=False,
                        help="Do not save cluster output as NPZ")
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
