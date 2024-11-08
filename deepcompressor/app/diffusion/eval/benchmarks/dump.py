import argparse
import os

import yaml
from tqdm import tqdm

from . import get_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=str, nargs="*", default=["COCO", "DCI", "MJHQ"])
    parser.add_argument("--max-dataset-size", type=int, default=-1)
    parser.add_argument("--dump-root", type=str, default="benchmarks")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--prompts-only", action="store_true")
    args = parser.parse_args()

    for benchmark in args.benchmarks:
        dataset = get_dataset(benchmark, max_dataset_size=args.max_dataset_size, return_gt=True)
        prompts = {}
        benchmark_root = os.path.join(args.dump_root, benchmark, f"{dataset.config_name}-{dataset._unchunk_size}")
        for row in tqdm(dataset, desc=f"Dumping {dataset.config_name}"):
            prompts[row["filename"]] = row["prompt"]
            if not args.prompts_only:
                image = row.get("image", None)
                if image is not None:
                    image_root = os.path.join(benchmark_root, "images")
                    os.makedirs(image_root, exist_ok=True)
                    if args.copy_images:
                        image.save(os.path.join(image_root, row["filename"] + ".png"))
                    else:
                        ext = os.path.basename(row["image_path"]).split(".")[-1]
                        os.symlink(
                            os.path.abspath(os.path.expanduser(row["image_path"])),
                            os.path.abspath(os.path.expanduser(os.path.join(image_root, row["filename"] + f".{ext}"))),
                        )
        os.makedirs(benchmark_root, exist_ok=True)
        with open(os.path.join(benchmark_root, "prompts.yaml"), "w") as f:
            yaml.dump(prompts, f)
