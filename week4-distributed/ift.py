import argparse
import copy
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Instruction fine-tuning on Alpaca dataset (10k examples) with prompt loss masking for next token prediction"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Hugging Face model name or path (e.g. 'gpt2' or a local directory)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sft_model",
        help="Directory to save the fine-tuned model",
    )
    # You can add a max_length argument if desired (default 512)
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length for tokenization"
    )
    return parser.parse_args()

def preprocess_function(example, tokenizer, max_length=512):
    # Assume the dataset has fields: "instruction", optional "input", and "output"
    instruction = example.get("instruction", "")
    user_input = example.get("input", "")
    output = example.get("output", "")

    # Build the prompt part.
    if user_input:
        prompt_text = f"Instruction: {instruction}\nInput: {user_input}\nResponse:"
    else:
        prompt_text = f"Instruction: {instruction}\nResponse:"
    
    # The full text is the prompt followed by a space and then the output.
    full_text = prompt_text + " " + output

    # Tokenize the full text with truncation and pad to max_length.
    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    # For determining how many tokens correspond to the prompt, tokenize prompt_text without padding.
    prompt_tokens = tokenizer(prompt_text, truncation=True, add_special_tokens=False)["input_ids"]
    prompt_length = len(prompt_tokens)
    
    # Copy the tokenized input_ids to create labels.
    labels = copy.deepcopy(tokenized_full["input_ids"])
    # For causal LM training with next-token prediction, Hugging Face models automatically shift the labels.
    # Therefore, to mask out the prompt portion we set the labels corresponding to the prompt (except possibly
    # the last prompt token which is used to predict the first token of the completion) to -100.
    # In other words, we disable loss for positions 1 up to prompt_length.
    for i in range(1, min(prompt_length + 1, len(labels))):
        labels[i] = -100
    tokenized_full["labels"] = labels

    return tokenized_full

def main():
    args = parse_args()

    # Load tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    tokenizer.pad_token = tokenizer.eos_token

    # Load the Alpaca dataset. Here we use the "yahma/alpaca-cleaned" version.
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    print(f"Total examples in dataset: {len(dataset)}")
    if len(dataset) > 10000:
        dataset = dataset.shuffle(seed=42).select(range(10000))

    # Preprocess the dataset: each example gets tokenized to a fixed length with padding.
    tokenized_dataset = dataset.map(
        lambda ex: preprocess_function(ex, tokenizer, max_length=args.max_length),
        batched=False,
    )

    # Set training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=100,
        fp16=True,
        save_steps=1000,
        save_total_limit=2,
    )

    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train and save the model.
    trainer.train()
    trainer.save_model(args.output_dir)
    print("Training complete and model saved to:", args.output_dir)

if __name__ == "__main__":
    main()
