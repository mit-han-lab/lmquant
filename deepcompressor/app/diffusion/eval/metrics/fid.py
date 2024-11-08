import os
from datetime import datetime

import numpy as np
import torch
import torchvision
from cleanfid import fid
from cleanfid.resize import build_resizer
from datasets import Dataset
from tqdm import tqdm


def get_dataset_features(
    dataset: Dataset,
    model,
    mode: str = "clean",
    batch_size: int = 128,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    to_tensor = torchvision.transforms.ToTensor()
    fn_resize = build_resizer(mode)
    np_feats = []
    for batch in tqdm(
        dataset.iter(batch_size=batch_size, drop_last_batch=False),
        desc=f"Extracting {dataset.config_name} features",
        total=(len(dataset) + batch_size - 1) // batch_size,
    ):
        resized_images = [fn_resize(np.array(image.convert("RGB"))) for image in batch["image"]]
        image_tensors = []
        for resized_image in resized_images:
            if resized_image.dtype == "uint8":
                image_tensor = to_tensor(resized_image) * 255
            else:
                image_tensor = to_tensor(resized_image)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors, dim=0)
        np_feats.append(fid.get_batch_features(image_tensors, model, device))
    np_feats = np.concatenate(np_feats, axis=0)
    return np_feats


def get_fid_features(
    dataset_or_folder: str | Dataset | None = None,
    cache_path: str | None = None,
    num: int | None = None,
    mode: str = "clean",
    num_workers: int = 8,
    batch_size: int = 64,
    device: str | torch.device = "cuda",
    force_overwrite: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path is not None and os.path.exists(cache_path) and not force_overwrite:
        npz = np.load(cache_path)
        mu, sigma = npz["mu"], npz["sigma"]
    else:
        feat_model = fid.build_feature_extractor(mode, device)
        if isinstance(dataset_or_folder, str):
            np_feats = fid.get_folder_features(
                dataset_or_folder,
                feat_model,
                num_workers=num_workers,
                num=num,
                batch_size=batch_size,
                device=device,
                verbose=verbose,
                mode=mode,
                description=f"Extracting {dataset_or_folder} features",
            )
        else:
            assert isinstance(dataset_or_folder, Dataset)
            np_feats = get_dataset_features(
                dataset_or_folder, model=feat_model, mode=mode, batch_size=batch_size, device=device
            )

        mu = np.mean(np_feats, axis=0)
        sigma = np.cov(np_feats, rowvar=False)
        if cache_path is not None:
            os.makedirs(os.path.abspath(os.path.dirname(cache_path)), exist_ok=True)
            np.savez(cache_path, mu=mu, sigma=sigma)

    return mu, sigma


def compute_fid(
    ref_dirpath_or_dataset: str | Dataset,
    gen_dirpath: str,
    ref_cache_path: str | None = None,
    gen_cache_path: str | None = None,
    use_symlink: bool = True,
    timestamp: str | None = None,
) -> float:
    sym_ref_dirpath, sym_gen_dirpath = None, None
    if use_symlink:
        if timestamp is None:
            timestamp = datetime.now().strftime("%y%m%d.%H%M%S")

        os.makedirs(".tmp", exist_ok=True)

        if isinstance(ref_dirpath_or_dataset, str):
            sym_ref_dirpath = os.path.join(".tmp", f"ref-{hash(str(ref_dirpath_or_dataset))}-{timestamp}")
            os.symlink(os.path.abspath(ref_dirpath_or_dataset), os.path.abspath(sym_ref_dirpath))
            ref_dirpath_or_dataset = sym_ref_dirpath

        sym_gen_dirpath = os.path.join(".tmp", f"gen-{hash(str(gen_dirpath))}-{timestamp}")
        os.symlink(os.path.abspath(gen_dirpath), os.path.abspath(sym_gen_dirpath))
        gen_dirpath = sym_gen_dirpath
    mu1, sigma1 = get_fid_features(dataset_or_folder=ref_dirpath_or_dataset, cache_path=ref_cache_path)
    mu2, sigma2 = get_fid_features(dataset_or_folder=gen_dirpath, cache_path=gen_cache_path)
    fid_score = fid.frechet_distance(mu1, sigma1, mu2, sigma2)
    fid_score = float(fid_score)
    if use_symlink:
        if sym_ref_dirpath is not None:
            os.remove(sym_ref_dirpath)
        os.remove(sym_gen_dirpath)
    return fid_score
