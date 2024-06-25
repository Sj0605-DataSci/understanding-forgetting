from itertools import product
import pandas as pd
from datetime import datetime

def concat_csvs(location, filenames, resulting_filename):
    combined_csv = pd.concat([pd.read_csv(location + f) for f in filenames], ignore_index=True)
    combined_csv.to_csv(location + resulting_filename + '.csv', index=False)

# Task, language, and model lists
task_list = [('capslock-math', 2), ('repeat-math', 2), ('capslock-startblank', 2), ('repeat-startblank', 2)]
lang_list = ['en', 'fr', 'es', 'nl', 'hu', 'ls', 'pl']
model_list = ['gemma', 'gemma-tuned']
instr_list = ['instr']
prompt_template_list = ['input']

print('Generate batch of datasets for generate.py...')
print('  Models:', model_list)

for model in model_list:
    for (task, shot), lang, instr, prompt_template in product(task_list, lang_list, instr_list, prompt_template_list):
        files = f"{task}-{instr}-{prompt_template}-{lang}-{shot}shot.csv"

        # Use a generic name for the batch file
        concatenated_filehandle = f"batch_{model}_{lang}"

        concat_csvs("/kaggle/working/understanding-forgetting/icl_vs_if/in_csvs/", [files], concatenated_filehandle)

        print(f'Generated {concatenated_filehandle}.csv')

        # Generate Python script for the current model and language
        complete_command = f"""
import subprocess

subprocess.run(["python3", "/kaggle/working/understanding-forgetting/icl_vs_if/generate.py", "--model", "{model}", "--batch", "{concatenated_filehandle}", "--lang", "{lang}"])
"""

        script_filename = f"/kaggle/working/understanding-forgetting/batch_generate_{model}_{lang}.py"
        with open(script_filename, "w") as f:
            f.write(complete_command)

        print(f"Generated {script_filename}")
