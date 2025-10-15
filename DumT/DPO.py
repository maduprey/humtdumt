import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    GenerationConfig,
)
from tqdm.auto import tqdm
from trl import DPOTrainer, DPOConfig
import itertools
import os
import argparse
import wandb
import sys
import json


# load model
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_EPOCHS = 10
cache_dir = YOUR_CACHE_DIR  # REPLACE WITH CACHE DIR
wandb.login(key="YOUR KEY HERE")

#############################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model loaded on %s" % device)


def main():
    # Check if filename argument is provided
    if len(sys.argv) < 2:
        print("Usage: python dpo_pset_code.py <number_of_objectives> <preference type>")
        return

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Process a file")
    parser.add_argument("preference_data", type=str, help="the preference data")
    parser.add_argument("model_name", type=str, help="name of model output file")
    parser.add_argument("output", type=str, help="name of inference output file")

    # Parse the arguments
    args = parser.parse_args()
    preference_data = args.preference_data
    output_file = args.output
    model_name = args.model_name
    with open(preference_data, "r") as f:
        PREFERENCE_DATA = json.load(f)

    prompt_list = [data["prompt"] for data in PREFERENCE_DATA]
    chosen_list = [data["chosen"] for data in PREFERENCE_DATA]
    rejected_list = [data["rejected"] for data in PREFERENCE_DATA]
    position_list = ["support" for _ in range(len(PREFERENCE_DATA))]
    train_dataset = Dataset.from_dict(
        {
            "prompt": prompt_list,
            "position": position_list,
            "chosen": chosen_list,
            "rejected": rejected_list,
        }
    )

    training_args = DPOConfig(
        output_dir="llama",
        logging_steps=10,
        per_device_train_batch_size=1,
        save_only_model=True,
        # num_train_epochs=NUM_EPOCHS,
        # gradient_accumulation_steps=8,
        learning_rate=LEARNING_RATE,
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        # beta=0.1,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    dpo_trainer.train()

    dpo_trainer.save_model(model_name)

    #############################
    # inference
    print("Running inference now:)!")

    # Load test dataset and prompts
    test_dataset = pd.read_csv("test_data.csv")
    test_prompts = [prompt for prompt in test_dataset["prompt"]]

    results = []

    for prompt in tqdm(test_prompts, desc="Running Inference", unit="prompt"):
        messages = [
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1] :]
        model_response = tokenizer.decode(response, skip_special_tokens=True)
        results.append(model_response)
        print("********************DPO response: ", model_response)

    # Save inference results to a CSV file
    test_dataset["dpo_response"] = results
    test_dataset.to_csv(output_file, index=False)
    # Finish W&B run
    # wandb.finish()


if __name__ == "__main__":
    main()
