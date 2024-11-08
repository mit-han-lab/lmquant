# -*- coding: utf-8 -*-
"""Evaluate generated images or videos using the specified metrics."""

import json
import os

from ...config import DiffusionPtqRunConfig

if __name__ == "__main__":
    config, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"
    assert len(unused_cfgs) == 0, f"Unused configurations: {unused_cfgs}"
    assert unused_args is None, f"Unused arguments: {unused_args}"
    assert isinstance(config, DiffusionPtqRunConfig)
    results = config.eval.evaluate(pipeline=None, skip_gen=True)
    save_path = os.path.join(config.eval.gen_root, f"results-{config.output.timestamp}.json")
    os.makedirs(os.path.abspath(os.path.dirname(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(results)
