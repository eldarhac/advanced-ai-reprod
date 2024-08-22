import os
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training_args import parse_args_from_dict, get_dataset
from models import DoNothingDataCollator, T5ForConditionalGenerationCustom, DoNothingDataCollatorSymtime

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

from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler

logger = logging.getLogger(__name__)

def train_and_eval_base_ptntime(params):
    # Parse the arguments
    model_args, data_args, training_args = parse_args_from_dict(params)
    if data_args.eval_data_file is None and model_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)


    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path != "new":
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
        )
    else:
        config = AutoConfig.from_pretrained("t5-small")
        model = T5ForConditionalGeneration(config=config)

    model.resize_token_embeddings(len(tokenizer))


    print(data_args.block_size)
    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    data_collator = DoNothingDataCollator()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        # trainer.train(model_path=model_path)
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        model.eval()
        sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=training_args.eval_batch_size,
            collate_fn=data_collator.collate_batch,
        )
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        writer = open(output_eval_file, "w")
        for inputs in tqdm(data_loader, "Prediction"):
            for k, v in inputs.items():
                inputs[k] = v.cuda()

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=12
                )
                dec = [tokenizer.decode(ids) for ids in outputs]

                for i in range(0, len(dec)):
                    writer.write(dec[i] + "\n")

    return results


def train_and_eval_symtime(params):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    model_args, data_args, training_args = parse_args_from_dict(params)

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    duration_model = T5ForConditionalGeneration.from_pretrained(
        model_args.duration_model_path,
    )
    model = T5ForConditionalGenerationCustom.from_pretrained(
        model_args.model_name_or_path,
    )
    model.duration_t5_model = duration_model

    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    data_collator = DoNothingDataCollatorSymtime()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        # trainer.train(model_path=model_path)
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # eval_output = trainer.evaluate()
        if model_args.eval_model_path != None:
            model = T5ForConditionalGenerationCustom.from_pretrained(model_args.eval_model_path).cuda()
            model.duration_t5_model = duration_model
        else:
            model = T5ForConditionalGenerationCustom.from_pretrained(training_args.output_dir).cuda()
        model.eval()
        sampler = SequentialSampler(eval_dataset)
        data_collator = DoNothingDataCollator()
        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=training_args.eval_batch_size,
            collate_fn=data_collator.collate_batch,
        )
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        writer = open(output_eval_file, "w")
        # 2841 -> negative
        # 1465 -> positive
        for inputs in tqdm(data_loader, "Prediction"):
            for k, v in inputs.items():
                inputs[k] = v.cuda()

            with torch.no_grad():
                outputs_lm_logits = model(**inputs)[2].detach().cpu().numpy()
                outputs_end = model(**inputs)[1].detach().cpu().numpy()

                for klm in range(0, len(outputs_lm_logits)):
                    label_1 = "positive"
                    if outputs_lm_logits[klm][2][2841] > outputs_lm_logits[klm][2][1465]:
                        label_1 = "negative"
                    label_2 = str(outputs_end[klm])
                    writer.write("\t".join([label_1, label_2]) + "\n")

    return results


def run_and_eval(model_name_or_path: str, output_dir: str, train_data_file: str,
                 eval_data_file: str, per_gpu_train_batch_size: int = 4,
                 per_device_train_batch_size: int = 4, gradient_accumulation_steps: int = 4,
                 per_device_eval_batch_size: int = 4, per_gpu_eval_batch_size: int = 4,
                 save_steps: int = 10000, duration_model_path: str = None):
    params = {
      "model_type": "t5",
      "tokenizer_name": "t5-large",
      "model_name_or_path": model_name_or_path,
      "duration_model_path": duration_model_path,
      "output_dir": output_dir,
      "do_train": True,
      "do_eval": True,
      "num_train_epochs": 50,
      "train_data_file": train_data_file,
      "eval_data_file": eval_data_file,
      "line_by_line": True,
      "per_gpu_train_batch_size": per_gpu_train_batch_size,
      "per_device_train_batch_size": per_device_train_batch_size,
      "gradient_accumulation_steps": gradient_accumulation_steps,
      "per_device_eval_batch_size": per_device_eval_batch_size,
      "per_gpu_eval_batch_size": per_gpu_eval_batch_size,
      "save_steps": save_steps,
      "logging_steps": 100,
      "overwrite_output_dir": True,
      "seed": 10,
    }

    if 'symtime' in model_name_or_path:
        return train_and_eval_symtime(params)
    else:
        return train_and_eval_base_ptntime(params)
