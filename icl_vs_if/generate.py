from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import pandas as pd
import bitsandbytes as bnb

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

assert args.model in MODEL_PATHS

in_csv = f'/kaggle/working/understanding-forgetting/icl_vs_if/in_csvs/{args.batch}-{args.lang}.csv'
out_csv = f'/kaggle/working/understanding-forgetting/icl_vs_if/out_csvs/{args.batch}-{args.model}-{args.lang}.csv'

df = pd.read_csv(in_csv)

def load_model(model_name, max_context_length=1024):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        load_in_8bit=True,  # Enable quantization with bitsandbytes
        max_length=max_context_length,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map='auto',
    )    

    return model, tokenizer

model, tokenizer = load_model(MODEL_PATHS[args.model])

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
df.to_csv(out_csv)
