import numpy as np
import pandas as pd
from evaluate_accuracy import get_results_dict
import sys

# Mocking the function 'get_results_dict' for the context of this example
def get_results_dict(base_csv_path):
    MODELS = ['Gemma', '', 'Gemma-tuned']
    CHUNK_SIZE = 100

    MODEL_LATEX_MAPPING = {
        'gemma': 'Gemma',
        'gemma-tuned': 'Gemma-Tuned',
     }

    dfs = {MODEL_LATEX_MAPPING[model]: pd.read_csv(base_csv_path.format(model=model)) for model in MODELS}

    num_samples_per_model = [len(dfs[model]['model_outputs']) for model in dfs]
    num_samples = num_samples_per_model[0]
    assert all([num == num_samples_per_model[0] for num in num_samples_per_model])
    assert num_samples % CHUNK_SIZE == 0

    scores = {MODEL_LATEX_MAPPING[model]: [] for model in MODELS}

    files = []

    for model in dfs:
        df = dfs[model]
        for i in range(0, num_samples, CHUNK_SIZE):
            num_correct = 0
            file_name = df['config'][i]
            files.append(file_name)
            for j in range(CHUNK_SIZE):
                correct_output = df['icl_ans'][i+j]
                model_output = df['model_outputs'][i+j]
                model_output = model_output.replace('\_', '_')

                if correct_output == ' Qu': 
                    correct_output = ' Combien'
                if correct_output == ' QU': 
                    correct_output = ' COMBIEN'

                is_correct = (model_output.startswith(correct_output) or model_output.startswith(correct_output[1:]))

                num_correct += is_correct
            scores[model].append(num_correct)

    return files, scores

# Mapping from language codes to names
lang_codes_to_name = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'nl': 'Dutch',
    'hu': 'Hungarian',
    'ls': 'Leetspeak',
    'pl': 'Pig Latin',
}

# Function to format mean and standard deviation
def format_mean_std(mean, std):
    return f"{mean:0.2f} %"

# Function to get mean and std for a specific task, language, and model
def get_mean_std(task, instr, templates, lang, shot, model):
    good_files = [f'{task}-{instr}-{template}-{lang}-{shot}shot' for template in templates]
    vals = list(map(lambda pair: pair[1], filter(lambda pair: pair[0] in good_files, zip(files, results[model]))))
    mean, std = np.mean(vals), np.std(vals)
    return mean, std

# Initialize paths and parameters
base_csv_path = '/kaggle/working/understanding-forgetting/icl_vs_if/out_csvs/{model}.csv'
files, results = get_results_dict(base_csv_path)
MODELS = ['Gemma', 'Gemma-Tuned']
TASK_SHOTS = [('capslock-math', 2), ('repeat-math', 2),
              ('capslock-startblank', 2), ('repeat-startblank', 2)]
TEMPLATES = ['input']
LANGS = ['en', 'fr', 'es', 'nl', 'hu', 'ls', 'pl']
INSTR = 'instr'
FINE_TUNES = [['Gemma', 'Gemma-Tuned']]

# Function to create a DataFrame cell for a specific task and model
def get_dataframe_cell(task, model, shot):
    cell_data = {}
    for lang in LANGS:
        mean, std = get_mean_std(task, INSTR, TEMPLATES, lang, shot, model)
        cell_data[lang_codes_to_name[lang]] = format_mean_std(mean, std)
    return cell_data

# Function to create a row for the DataFrame for a specific task
def get_dataframe_row(task, models, shot):
    row_data = []
    for model in models:
        row_data.append(get_dataframe_cell(task, model, shot))
    return row_data

# Create DataFrames for each task
task_dataframes = []

for (task, shot) in TASK_SHOTS:
    task_name = f'{task.split("-")[0].capitalize()} {task.split("-")[1].capitalize()}'
    task_df = pd.DataFrame()
    row_data = get_dataframe_row(task, MODELS, shot)
    
    for model_idx, model in enumerate(MODELS):
        model_data = row_data[model_idx]
        model_df = pd.DataFrame(model_data, index=[task_name])
        model_df['Model'] = model
        task_df = pd.concat([task_df, model_df])
    
    task_dataframes.append(task_df)

# Concatenate all task DataFrames
final_task_df = pd.concat(task_dataframes)
final_task_df = final_task_df.reset_index().rename(columns={'index': 'Task'})

# Display the final DataFrame
print(final_task_df)

# Calculate averages and differences for fine-tuned models
average_results = {
    lang_codes_to_name[lang]: {
        model: np.mean([get_mean_std(task, INSTR, TEMPLATES, lang, shot, model)[0] for (task, shot) in TASK_SHOTS])
        for model in MODELS
    } for lang in LANGS
}

average_data = []
for fine_tune in FINE_TUNES:
    base, tuned = fine_tune
    for lang in LANGS:
        lang_name = lang_codes_to_name[lang]
        base_mean = average_results[lang_name][base]
        tuned_mean = average_results[lang_name][tuned]
        diff = base_mean - tuned_mean
        average_data.append([lang_name, base, base_mean, tuned, tuned_mean, diff])

average_df = pd.DataFrame(average_data, columns=['Language', 'Base Model', 'Base Avg %', 'Tuned Model', 'Tuned Avg %', 'Difference %'])

# Display the averages DataFrame
print(average_df)
