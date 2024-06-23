import pandas as pd
import pdb
from pprint import pprint

def get_results_dict(BASE_CSV_PATH):
    MODEL = 'gemma'
    CHUNK_SIZE = 100

    MODEL_LATEX_MAPPING = {
        'gemma': 'Gemma',
    }

    df = pd.read_csv(BASE_CSV_PATH.format(model=MODEL))

    num_samples = len(df['model_outputs'])
    assert num_samples % CHUNK_SIZE == 0

    scores = {MODEL_LATEX_MAPPING[MODEL]: []}

    files = []

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
        scores[MODEL_LATEX_MAPPING[MODEL]].append(num_correct)

    return files, scores
