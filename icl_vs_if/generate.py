from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import pandas as pd

# Define the model paths
MODEL_PATHS = {
    "gemma": "google/gemma-7b",
    "gemma-tuned": "weqweasdas/RM-Gemma-7B"
}

BATCH_SIZE = 1

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--batch', type=str, required=True)
parser.add_argument('--lang', type=str, required=True)
args = parser.parse_args()

# Ensure the model argument is valid
assert args.model in MODEL_PATHS, f"Invalid model name: {args.model}"

# Construct input and output CSV paths
in_csv = f'/kaggle/working/understanding-forgetting/icl_vs_if/in_csvs/{args.batch}.csv'  # Adjusted to use args.batch directly
out_csv = f'/kaggle/working/understanding-forgetting/icl_vs_if/out_csvs/{args.batch}_{args.model}_{args.lang}.csv'

df = pd.read_csv(in_csv)

model = None
tokenizer = None

def load_model(model_name, max_context_length=1024):
    global model, tokenizer
    # Unload previous model and tokenizer
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS[model_name],
        device='cuda' if torch.cuda.is_available() else 'cpu',  # Adjust device based on availability
        load_in_8bit=True,  # Enable quantization with bitsandbytes
        max_length=max_context_length,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS[model_name],
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

def model_forward_batch(input_batch):
    inputs = tokenizer(input_batch, return_tensors="pt", add_special_tokens=False)
    output_tokens_batch = model.generate(inputs['input_ids'], temperature=0.0, max_new_tokens=10)
    return tokenizer.batch_decode(output_tokens_batch, skip_special_tokens=True)

num_samples = len(df['prompt'])
assert num_samples % BATCH_SIZE == 0

model_outputs = []

for i in tqdm(range(0, num_samples, BATCH_SIZE)):
    input_batch = df['prompt'][i:i+BATCH_SIZE].values.tolist()
    output_batch = model_forward_batch(input_batch)
    
    for j in range(len(output_batch)):
        output_batch[j] = output_batch[j][len(input_batch[j]):]
        model_outputs.append(output_batch[j])

df['model_outputs'] = model_outputs
df.to_csv(out_csv, index=False)

# Unload model and tokenizer after use
del model
del tokenizer
