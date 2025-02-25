import torch
import argparse
from model import GPT
from utils import GPTTokenization
import os
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Generate text using trained minGPT model')
    parser.add_argument('--model_path', type=str, default='models/shakespeare/shakespeare_20250223-205312_best.pth',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--seed_text', type=str, default='SCENE I. London. The palace.',
                        help='Seed text to start generation')
    parser.add_argument('--max_tokens', type=int, default=5000,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for sampling (higher = more random)')
    parser.add_argument('--do_sample', action='store_true',
                        help='Use sampling instead of greedy decoding')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling parameter (set to None to disable)')
    parser.add_argument('--context_len', type=int, default=2000,
                        help='Context length used during training')
    parser.add_argument('--n_decoders', type=int, default=6,
                        help='Number of decoder blocks in the model')
    parser.add_argument('--n_heads', type=int, default=6,
                        help='Number of attention heads per block')
    parser.add_argument('--embedding_dim', type=int, default=192,
                        help='Embedding dimension used in the model')
    parser.add_argument('--ff_multiplier', type=int, default=4,
                        help='Feed-forward network multiplier used in the model')
    parser.add_argument('--delay', type=float, default=0.02,
                        help='Delay between token generation (seconds) for visualization')
    return parser.parse_args()

def load_model(args):
    # Initialize tokenizer with small sample to get vocab size
    # (GPT-2 tokenizer has fixed vocabulary)
    tokenizer = GPTTokenization("sample", context_len=args.context_len)
    vocab_len = tokenizer.vocab_len

    # Create model with same architecture as during training
    model = GPT(
        src_vocab_len=vocab_len,
        max_seq_len=args.context_len,
        dropout=0.1,  # Dropout doesn't matter for inference
        n_decoders=args.n_decoders,
        n_heads=args.n_heads,
        embedding_dim=args.embedding_dim,
        ff_multiplier=args.ff_multiplier
    )

    # Check if model path exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    # Load the trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, tokenizer, device

def generate_text_realtime(model, tokenizer, args, device):
    # Tokenize the seed text
    input_ids = tokenizer.encode_text(args.seed_text)
    input_tensor = torch.tensor(input_ids, dtype=torch.long)[None].to(device)
    
    print(f"\nGenerating text with seed: '{args.seed_text}'")
    print(f"Using temperature: {args.temperature}, top_k: {args.top_k if args.do_sample else 'N/A (greedy)'}")
    print("-" * 50)
    
    # Print seed text first
    seed_text = tokenizer.decode_tokens(input_ids)
    sys.stdout.write(seed_text)
    sys.stdout.flush()
    
    # Buffer to store the entire generated text
    full_text = input_ids.copy()
    
    # Generate text one token at a time with real-time output
    with torch.no_grad():
        for i in range(args.max_tokens):
            # Ensure we don't exceed context length
            context = input_tensor[:, -args.context_len:] if input_tensor.size(1) > args.context_len else input_tensor
            
            # Get logits for the next token
            logits = model(context)[:, -1, :] / args.temperature
            
            # Apply top-k sampling if specified
            if args.do_sample and args.top_k is not None:
                top_values, _ = torch.topk(logits, args.top_k)
                logits[logits < top_values[:, [-1]]] = -float('inf')
            
            # Either sample or do greedy decoding
            if args.do_sample:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(logits, k=1, dim=-1)
                
            # Decode and display the token
            token_id = next_token[0].item()
            token_text = tokenizer.decode_tokens([token_id])
            sys.stdout.write(token_text)
            sys.stdout.flush()
            
            # Add to full text
            full_text.append(token_id)
            
            # Append to input tensor for next iteration
            input_tensor = torch.cat((input_tensor, next_token), dim=1)
            
            # Add a small delay for visual effect
            time.sleep(args.delay)
    
    print("\n\n" + "-" * 50)
    return tokenizer.decode_tokens(full_text)

def main():
    args = parse_args()
    model, tokenizer, device = load_model(args)
    
    print(f"Model loaded from {args.model_path}")
    print(f"Running on {device}")
    
    # Generate text with real-time display
    full_generated_text = generate_text_realtime(model, tokenizer, args, device)
    
    # Save the generated text to a file
    output_file = "generated_shakespeare.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_generated_text)
    
    print(f"\nGenerated text saved to {output_file}")

if __name__ == "__main__":
    main()