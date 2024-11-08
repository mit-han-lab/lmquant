import os.path

import datasets

__all__ = ["get_dataset"]


def get_dataset(
    name: str,
    config_name: str | None = None,
    split: str = "train",
    max_dataset_size: int = -1,
    return_gt: bool = False,
    repeat: int = 4,
    chunk_start: int = 0,
    chunk_step: int = 1,
) -> datasets.Dataset:
    prefix = os.path.dirname(__file__)
    kwargs = {
        "name": config_name,
        "split": split,
        "trust_remote_code": True,
        "token": True,
        "max_dataset_size": max_dataset_size,
    }
    if name.endswith(".yaml") or name.endswith(".yml"):
        path = os.path.join(prefix, "YAML")
        dataset = datasets.load_dataset(path, url=name, repeat=repeat, download_mode="force_redownload", **kwargs)
    else:
        path = os.path.join(prefix, f"{name}")
        if name == "COCO":
            dataset = datasets.load_dataset(path, return_gt=return_gt, **kwargs)
        elif name == "DCI":
            dataset = datasets.load_dataset(path, return_gt=return_gt, **kwargs)
        elif name == "MJHQ":
            dataset = datasets.load_dataset(path, return_gt=return_gt, **kwargs)
        else:
            raise ValueError(f"Unknown dataset name: {name}")
    assert not hasattr(dataset, "_unchunk_size")
    assert not hasattr(dataset, "_chunk_start")
    assert not hasattr(dataset, "_chunk_step")
    unchunk_size = len(dataset)
    if chunk_step > 1 or chunk_start > 0:
        assert 0 <= chunk_start < chunk_step
        dataset = dataset.select(range(chunk_start, len(dataset), chunk_step))
    else:
        chunk_start, chunk_step = 0, 1
    dataset._unchunk_size = unchunk_size
    dataset._chunk_start = chunk_start
    dataset._chunk_step = chunk_step
    return dataset
