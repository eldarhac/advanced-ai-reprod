import logging
import os
from dataclasses import dataclass, field
from typing import Optional
import torch

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    T5ForConditionalGeneration,
)


from torch.utils.data.dataset import Dataset

from models import LineByLineTextDatasetSymtime

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    duration_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Duration model path"
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class TrainingArguments:
    output_dir: str = field(default='/content/drive/MyDrive/AI_recovery/ptntime/improved_model/experiment_result')
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    num_train_epochs: int = field(default=50)
    per_gpu_train_batch_size: int = field(default=4)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    per_gpu_eval_batch_size: int = field(default=4)
    save_steps: int = field(default=5000)
    logging_steps: int = field(default=1000)
    overwrite_output_dir: bool = field(default=True)
    seed: int = field(default=10)
    local_rank=-1
    device='cuda'
    n_gpu=1
    fp16=False
    train_batch_size=4
    eval_batch_size=4
    gradient_accumulation_steps=4
    block_size=-1
    max_steps=-1
    max_grad_norm=1.0
    warmup_steps=0
    learning_rate=3e-5
    weight_decay=0.0
    adam_epsilon=1e-8
    eval_steps=5000
    logging_steps=1000


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        originals = []
        labels = []
        for l in lines:
            if len(l.split("\t")) < 2:
                continue
            originals.append(l.split("\t")[0])
            labels.append(l.split("\t")[1])

        self.inputs = tokenizer.batch_encode_plus(originals, pad_to_max_length=True)
        self.labels = tokenizer.batch_encode_plus(labels, pad_to_max_length=True)

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, i):
        source_ids = self.inputs["input_ids"][i]
        target_ids = self.labels["input_ids"][i]
        src_mask = self.inputs["attention_mask"][i]
        target_mask = self.labels["attention_mask"][i]
        return {"input_ids": source_ids, "attention_mask": src_mask, "lm_labels": target_ids, "decoder_attention_mask": target_mask}


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        ret = LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path)
        print("DATA SIZE: ")
        print(len(ret))
        return ret
    else:
        return None


def get_dataset_symtime(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        ret = LineByLineTextDatasetSymtime(tokenizer=tokenizer, file_path=file_path)
        print("DATA SIZE: ")
        print(len(ret))
        return ret
    else:
        return None


def parse_args_from_dict(params: dict):
    model_args = ModelArguments(
        model_type=params['model_type'],
        tokenizer_name=params['tokenizer_name'],
        model_name_or_path=params['model_name_or_path'],
        duration_model_path=params['duration_model_path']
    )

    data_args = DataTrainingArguments(
        train_data_file=params['train_data_file'],
        eval_data_file=params['eval_data_file'],
        line_by_line=params['line_by_line']
    )

    training_args = TrainingArguments(
        output_dir=params['output_dir'],
        do_train=params['do_train'],
        do_eval=params['do_eval'],
        num_train_epochs=params['num_train_epochs'],
        per_gpu_train_batch_size=params['per_gpu_train_batch_size'],
        per_device_train_batch_size=params['per_device_train_batch_size'],
        gradient_accumulation_steps=params['gradient_accumulation_steps'],
        per_device_eval_batch_size=params['per_device_eval_batch_size'],
        per_gpu_eval_batch_size=params['per_gpu_eval_batch_size'],
        save_steps=params['save_steps'],
        logging_steps=params['logging_steps'],
        overwrite_output_dir=params['overwrite_output_dir'],
        seed=params['seed']
    )

    return model_args, data_args, training_args
