import pandas as pd
from pprint import pprint

def get_results_dict(BASE_CSV_PATH):
    MODELS = ['gemma', 'gemma-tuned']
    CHUNK_SIZE = 100

    MODEL_LATEX_MAPPING = {
        'gemma': 'Gemma',
        'gemma-tuned': 'Gemma-Tuned',
    }

    # Load the CSV files for each model
    dfs = {}
    for model in MODELS:
        try:
            dfs[MODEL_LATEX_MAPPING[model]] = pd.read_csv(BASE_CSV_PATH.format(model=model))
        except Exception as e:
            print(f"Error loading CSV for model {model}: {e}")
            continue

    # Check if all models have the same number of samples
    num_samples_per_model = {model: len(dfs[model]['model_outputs']) for model in dfs}
    num_samples = list(num_samples_per_model.values())[0]
    assert all([num == num_samples for num in num_samples_per_model.values()]), "Sample size mismatch across models"
    assert num_samples % CHUNK_SIZE == 0, "Total samples should be divisible by chunk size"

    scores = {model: [] for model in dfs}
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

                # Specific handling for French corrections
                if correct_output == ' Qu': 
                    correct_output = ' Combien'
                if correct_output == ' QU': 
                    correct_output = ' COMBIEN'

                is_correct = (model_output.startswith(correct_output) or model_output.startswith(correct_output[1:]))
                num_correct += is_correct

            scores[model].append(num_correct)

    return files, scores

# Example usage
BASE_CSV_PATH = '/kaggle/working/understanding-forgetting/icl_vs_if/out_csvs/{model}.csv'
files, scores = get_results_dict(BASE_CSV_PATH)

print("Files processed:")
pprint(files)

print("\nScores:")
pprint(scores)
