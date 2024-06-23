from itertools import product
import pandas as pd
from datetime import datetime

def concat_csvs(location, filenames, resulting_filename):
    combined_csv = pd.concat([pd.read_csv(location + f) for f in filenames], ignore_index=True)
    combined_csv.to_csv(location + resulting_filename + '.csv', index=False)

# Only English language and Gemma model
task_list = [('capslock-math', 2), ('repeat-math', 2), ('capslock-startblank', 2), ('repeat-startblank', 2)]
lang_list = ['en']
model_list = ['gemma']
instr_list = ['instr']
prompt_template_list = ['input']

print('Generate batch of datasets for generate.py...')
print('  Models: Gemma')

files = []
for (task, shot), lang, instr, prompt_template in product(task_list, lang_list, instr_list, prompt_template_list):
    files.append(f"{task}-{instr}-{prompt_template}-{lang}-{shot}shot.csv")

# Use a generic name for the batch file
concatenated_filehandle = "batch"

concat_csvs("/kaggle/working/understanding-forgetting/icl_vs_if/in_csvs/", files, concatenated_filehandle)

print(f'Generated {concatenated_filehandle}.csv')

# Generate Python script for Gemma model only
complete_command = """
import subprocess

subprocess.run(["python3", "/kaggle/working/understanding-forgetting/icl_vs_if/generate.py", "--model", "gemma", "--batch", "batch"])
"""

with open("/kaggle/working/understanding-forgetting/batch_generate.py", "w") as f:
    f.write(complete_command)

print("Generated batch_generate.py")