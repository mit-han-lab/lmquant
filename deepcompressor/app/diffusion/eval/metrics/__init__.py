import logging
import os

from .fid import compute_fid
from .image_reward import compute_image_reward
from .multimodal import compute_image_multimodal_metrics
from .similarity import compute_image_similarity_metrics
from ..benchmarks import get_dataset

logging.getLogger("PIL").setLevel(logging.WARNING)

__all__ = ["compute_image_metrics"]


def compute_image_metrics(
    gen_root: str,
    benchmarks: str | tuple[str, ...] = ("DCI", "GenAIBench", "GenEval", "MJHQ", "T2ICompBench"),
    max_dataset_size: int = -1,
    chunk_start: int = 0,
    chunk_step: int = 1,
    chunk_only: bool = False,
    ref_root: str = "",
    gt_stats_root: str = "",
    gt_metrics: tuple[str, ...] = ("clip_iqa", "clip_score", "image_reward", "fid"),
    ref_metrics: tuple[str, ...] = ("psnr", "lpips", "ssim", "fid"),
) -> dict:
    if chunk_start == 0 and chunk_step == 1:
        chunk_only = False
    assert chunk_start == 0 and chunk_step == 1, "Chunking is not supported for image benchmarks."
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if isinstance(benchmarks, str):
        benchmarks = (benchmarks,)
    gt_multimodal_metrics, gt_similarity_metrics, gt_other_metrics = categorize_metrics(gt_metrics)
    _, ref_similarity_metrics, ref_other_metrics = categorize_metrics(ref_metrics)
    results = {}
    for benchmark in benchmarks:
        benchmark_results = {}
        dataset = get_dataset(benchmark, max_dataset_size=max_dataset_size, return_gt=True)
        dirname = f"{dataset.config_name}-{dataset._unchunk_size}"
        if dataset._chunk_start == 0 and dataset._chunk_step == 1:
            filename = f"{dirname}.npz"
        else:
            filename = os.path.join(dirname, f"{dataset._chunk_start}-{dataset._chunk_step}.npz")
            if chunk_only:
                dirname += f".{dataset._chunk_start}.{dataset._chunk_step}"
        gen_dirpath = os.path.join(gen_root, "samples", benchmark, dirname)
        if gt_metrics:
            gt_results = compute_image_multimodal_metrics(dataset, gen_dirpath, metrics=gt_multimodal_metrics)
            if "image_reward" in gt_other_metrics:
                gt_results.update(compute_image_reward(dataset, gen_dirpath))
            if benchmark in ("COCO", "DCI", "MJHQ"):
                gt_results.update(compute_image_similarity_metrics(dataset, gen_dirpath, metrics=gt_similarity_metrics))
                if "fid" in gt_other_metrics:
                    gt_results["fid"] = compute_fid(
                        dataset,
                        gen_dirpath,
                        ref_cache_path=(os.path.join(gt_stats_root, benchmark, filename) if gt_stats_root else None),
                        gen_cache_path=os.path.join(gen_root, "fid_stats", benchmark, filename),
                    )
            benchmark_results["with_gt"] = gt_results
        if ref_root and ref_metrics:
            assert os.path.exists(ref_root), f"Reference root directory {ref_root} does not exist."
            ref_dirpath = os.path.join(ref_root, "samples", benchmark, dirname)
            ref_results = compute_image_similarity_metrics(ref_dirpath, gen_dirpath, metrics=ref_similarity_metrics)
            if "fid" in ref_other_metrics:
                ref_results["fid"] = compute_fid(
                    ref_dirpath,
                    gen_dirpath,
                    ref_cache_path=os.path.join(ref_root, "fid_stats", benchmark, filename),
                    gen_cache_path=os.path.join(gen_root, "fid_stats", benchmark, filename),
                )
            benchmark_results["with_orig"] = ref_results
        print(f"{dirname} results:")
        print(benchmark_results)
        results[dirname] = benchmark_results
    return results


def categorize_metrics(metrics: tuple[str, ...]) -> tuple[list[str], list[str], list[str]]:
    """
    Categorize metrics into multimodal, similarity, and other metrics.

    Args:
        metrics (tuple[str, ...]): List of metrics.

    Returns:
        tuple[list[str], list[str], list[str]]: Tuple of multimodal, similarity, and other metrics.
    """
    metrics = tuple(set(metrics))
    multimodal_metrics, similarity_metrics, other_metrics = [], [], []
    for metric in metrics:
        if metric in ("clip_iqa", "clip_score"):
            multimodal_metrics.append(metric)
        elif metric in ("psnr", "lpips", "ssim"):
            similarity_metrics.append(metric)
        else:
            other_metrics.append(metric)
    return multimodal_metrics, similarity_metrics, other_metrics
