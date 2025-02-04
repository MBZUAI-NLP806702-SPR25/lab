import argparse
import os
import copy
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
import deepspeed

def parse_args():
    parser = argparse.ArgumentParser(description="Manual SFT Training with optional DeepSpeed")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Hugging Face model name or path (e.g. 'distilgpt2')")
    parser.add_argument("--output_dir", type=str, default="./sft_manual")
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--micro_batch", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()

def preprocess_function(example, tokenizer, max_length=512):
    # Assume each example has 'instruction', optional 'input', and 'output'
    instruction = example.get("instruction", "")
    user_input = example.get("input", "")
    output = example.get("output", "")
    
    # Build a prompt string
    if user_input:
        prompt = f"Instruction: {instruction}\nInput: {user_input}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"
    full_text = prompt + " " + output
    
    # Tokenize full text to fixed length with padding
    tokenized = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length")
    # Tokenize prompt (without padding) to know its token length
    prompt_tokens = tokenizer(prompt, truncation=True, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_tokens)
    
    # Create labels as a copy of input_ids
    labels = copy.deepcopy(tokenized["input_ids"])
    # For next-token prediction the model automatically shifts labels.
    # Here, we mask out the prompt portion (positions 1 up to prompt_len) by setting them to -100.
    for i in range(1, min(prompt_len, len(labels))):
        labels[i] = -100
    tokenized["labels"] = labels
    return tokenized

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    
    tokenizer.pad_token = tokenizer.eos_token

    # Load a sample instruction dataset (here, a cleaned Alpaca dataset)
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    # Select 10k examples if dataset is larger
    if len(dataset) > 10000:
        dataset = dataset.shuffle(seed=42).select(range(10000))
    
    # Preprocess each example (build full text and mask prompt tokens)
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, max_length=args.max_length))
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Optionally initialize DeepSpeed
    if args.use_deepspeed:
        # A simple DeepSpeed config: enable fp16 training and set batch size
        ds_config = {
            "train_batch_size": args.batch_size,
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3,
            },
            "zero_allow_untested_optimizer": True,
        }
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            config=ds_config,
            model_parameters=model.parameters()
        )
        print("Training with DeepSpeed enabled.")
    else:
        model_engine = model
        print("Training without DeepSpeed.")
    
    model_engine.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            # Move batch tensors to device
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model_engine(**batch)
            loss = outputs.loss
            if args.use_deepspeed:
                model_engine.backward(loss)
                model_engine.step()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_deepspeed:
        model_engine.save_checkpoint(args.output_dir)
    else:
        model.save_pretrained(args.output_dir)
    print("Training complete. Model saved to", args.output_dir)

if __name__ == "__main__":
    main()