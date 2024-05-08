# -*- coding: utf-8 -*-
"""Language model evaluation adaptor for lm_eval."""

import logging
import math
import typing as tp
from dataclasses import dataclass, field

import lm_eval
import lm_eval.models
import omniconfig
import torch
import torch.nn as nn
from datasets import load_dataset
from omniconfig import configclass
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmquant.eval.config import BaseEvaluationConfig
from lmquant.utils import tools

__all__ = ["LlmEvalConfig"]


@configclass
@dataclass
class LlmEvalConfig(BaseEvaluationConfig):
    """Arguments for evaluating a large language model.

    Attributes:
        num_gpus (int, optional): The number of GPUs to use. Defaults to ``1``.
        batch_size (int, optional): The batch size used for inference. Defaults to ``1``.
        output_dirname (str | None, optional): The output directory name. Defaults to ``None``.
        tasks (list[str]): Task names, e.g. wikitext, hellaswag, piqa, winogrande.
                           Defaults to ``['wikitext']``.
        max_seq_length (int): Maximum sequence length. Defaults to ``-4096``.
        evaluator (str): The evaluator to use. Defaults to ``gptq``.
    """

    tasks: list[str] = field(
        default_factory=lambda: ["wikitext"],
        metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": str}},
    )
    max_seq_length: int = -4096
    evaluator: str = "gptq"

    def __post_init__(self):
        super().__post_init__()
        if "zero-shot" in self.tasks:
            self.tasks.remove("zero-shot")
            self.tasks.extend(("wikitext", "hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge"))
        self.tasks = sorted(set(self.tasks))
        assert self.evaluator in ("lm_eval", "gptq"), f"Invalid evaluator: {self.evaluator}"
        if self.evaluator == "gptq":
            self.tasks = [task for task in self.tasks if task.startswith(("wikitext", "pile"))]
            assert len(self.tasks) > 0, "No valid tasks for GPTQ evaluation"

    def evaluate(
        self, model: AutoModelForCausalLM, /, tokenizer: AutoTokenizer, model_name: str
    ) -> dict[int, dict[str, dict[tp.Any, dict[str, tp.Any]]]]:
        """Evaluate the model.

        Args:
            model_name (str): The name of the model.
            model (AutoModelForCausalLM): The model.
            tokenizer (AutoTokenizer): The tokenizer.

        Returns:
            dict[str, dict[tp.Any, dict]]: The evaluation results.
        """
        logger = logging.getLogger(f"{__name__}.LlmEval")
        tools.logging.Formatter.indent_inc()
        lm = _EvalLM(model_name, model, tokenizer=tokenizer, batch_size=self.batch_size)
        lm_max_seq_length = lm.max_length
        tools.logging.Formatter.indent_dec()
        max_seq_lengths = set([2048, 4096, lm_max_seq_length])
        if self.max_seq_length < 0:
            if self.max_seq_length == -1:
                max_seq_length = lm_max_seq_length
            else:
                max_seq_length = min(lm_max_seq_length, -self.max_seq_length)
            max_seq_lengths = [length for length in sorted(max_seq_lengths) if length <= max_seq_length]
        elif self.max_seq_length == 0:
            max_seq_lengths = [lm_max_seq_length]
        else:
            max_seq_lengths = [self.max_seq_length]
        logger.info(f"- Evaluator: {self.evaluator}")
        logger.info(f"- Tasks: {self.tasks}")
        logger.info(f"- Batch_size: {self.batch_size}")
        results = {}
        tools.logging.Formatter.indent_inc()
        for max_seq_length in max_seq_lengths:
            logger.info(f"+ Max_seq_length: {max_seq_length}")
            lm._max_length = max_seq_length
            tools.logging.Formatter.indent_inc()
            tools.logging.Formatter.indent_inc()
            if self.evaluator == "lm_eval":
                rst = lm_eval.evaluator.simple_evaluate(
                    model=lm, tasks=self.tasks, batch_size=self.batch_size, verbosity="ERROR"
                )
                if "samples" in rst:
                    del rst["samples"]
            else:
                rst = dict(results={}, versions={}, config=dict(model=model_name))
                for task in self.tasks:
                    rst["results"][task] = {
                        "word_perplexity": _eval_ppl_with_gptq_evaluator(
                            model, tokenizer, task=task, seq_length=max_seq_length
                        )
                    }
                    rst["versions"][task] = 1
            results[max_seq_length] = rst
            tools.logging.Formatter.indent_dec()
            logger.info("- Results:")
            tools.logging.Formatter.indent_inc()
            tools.logging.info(LlmEvalConfig.make_table(rst), logger=logger)
            tools.logging.Formatter.indent_dec()
            tools.logging.Formatter.indent_dec()
        tools.logging.Formatter.indent_dec()
        return results

    @staticmethod
    def make_table(rst: dict[str, dict[tp.Any, dict[str, tp.Any]]]) -> str:
        """Generate table of results.

        Args:
            results (dict[str, dict[tp.Any, dict[str, tp.Any]]]): The evaluation results.

        Returns:
            str: The string representation of the results in a table.
        """
        from pytablewriter import MarkdownTableWriter

        md_writer = MarkdownTableWriter()
        md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
        values = []
        for k, dic in rst["results"].items():
            version = rst["versions"][k]
            for m, v in dic.items():
                if "_stderr" in m:
                    continue
                mse = "_stderr,".join(m.split(","))
                appended = False
                if mse in dic:
                    se = dic[mse]
                    if isinstance(se, (int, float)):
                        values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
                        appended = True
                if not appended and isinstance(v, (int, float)):
                    values.append([k, version, m, "%.4f" % v, "", ""])
                k = ""
                version = ""
        md_writer.value_matrix = values
        return md_writer.dumps()


class _EvalLM(lm_eval.models.huggingface.HFLM):
    """Language model evaluation adaptor for lm_eval."""

    def __init__(
        self,
        model_name: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int = 1,
        max_length: int | None = None,
    ):
        """Initialize the adaptor for language model evaluation.

        Args:
            model_name (str): The name of the model.
            model (AutoModelForCausalLM): The model.
            tokenizer (AutoTokenizer): The tokenizer.
            batch_size (int, optional): The batch size used for inference. Defaults to ``1``.
            max_length (int | None, optional): The maximum length of a sequence. Defaults to ``None``.
        """
        assert isinstance(batch_size, int)
        model.eval()
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
        )
        self.name = model_name
        self.family = model_name.split("-")[0].lower()

    def _model_call(self, inps, attn_mask=None, labels=None):
        logits = super()._model_call(inps, attn_mask, labels)
        return logits[:, :, :50257] if self.family == "opt" else logits


def _eval_ppl_with_gptq_evaluator(
    model: AutoModelForCausalLM,
    /,
    tokenizer: AutoTokenizer,
    task: str,
    seq_length: int = 2048,
    max_num_samples: int = -1,
) -> float:
    """Evaluate the perplexity of a model on a task using GPTQ style evaluation.

    Args:
        model (AutoModelForCausalLM): The model.
        tokenizer (AutoTokenizer): The tokenizer.
        task (str): The task name.
        seq_length (int, optional): The sequence length. Defaults to ``2048``.
        max_num_samples (int, optional): The maximum number of samples to evaluate. Defaults to ``-1``.

    Returns:
        float: The perplexity.
    """
    assert seq_length > 0, "seq_length must be positive"
    if task.startswith("wikitext"):
        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test_dataset = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    elif task.startswith("pile"):
        test_dataset = load_dataset("pile", task, split="test")
        test_dataset = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    else:
        raise ValueError(f"Invalid task: {task}")

    test_dataset = test_dataset.input_ids.to(model.device)
    num_samples = test_dataset.numel() // seq_length
    if max_num_samples > 0:
        num_samples = min(num_samples, max_num_samples)
    model = model.eval()

    nlls = []
    for i in tqdm(range(num_samples), desc=f"evaluating on {task} with seq_length {seq_length}"):
        batch = test_dataset[:, (i * seq_length) : ((i + 1) * seq_length)]
        with torch.inference_mode():
            shift_logits = model(batch.to(model.device)).logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seq_length
        nlls.append(neg_log_likelihood)
    return math.exp(sum(nlls) / (num_samples * seq_length))
