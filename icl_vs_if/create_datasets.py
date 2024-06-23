import pandas as pd
import random
from itertools import product

# TASK_LIST specifies tasks and shots
TASK_LIST = [('capslock', 'math', 2), ('repeat', 'math', 2), ('capslock', 'startblank', 2), ('repeat', 'startblank', 2)]
INSTRUCTION_TEMPLATE_LIST = ["instr"]
PROMPT_TEMPLATE_LIST = ["input"]

print('Generating datasets for English...')

for (ICL_TASK, IF_TASK, kshot), INSTRUCTION_TEMPLATE, PROMPT_TEMPLATE in product(TASK_LIST, INSTRUCTION_TEMPLATE_LIST, PROMPT_TEMPLATE_LIST):    
    with open("/kaggle/working/understanding-forgetting/icl_vs_if/sentences/random_sentences_en.txt", 'r', encoding='utf-8') as f:
        sentences = [line.strip('\n') for line in f]

    # Define ICL tasks
    if ICL_TASK == "repeat":
        def icl_task(input_line):
            return input_line
    elif ICL_TASK == "capslock":
        def icl_task(input_line):
            return input_line.upper()

    # Define IF tasks
    if IF_TASK == "startblank":
        def if_task(input_line):
            words = input_line.split(' ')
            wrong_ans = words[0]
            words[0] = '_' * len(words[0])
            return ' '.join(words), words[0], wrong_ans
    elif IF_TASK == "math":
        def if_task(input_line):
            num1 = random.randint(4, 20)
            num2 = random.randint(4, 20)
            op_idx = random.randint(0, 2)
            op_text = ['plus', 'minus', 'times'][op_idx]
            op_math = [lambda x: x[0] + x[1], lambda x: x[0] - x[1], lambda x: x[0] * x[1]][op_idx]
            prompt = f"What is {num1} {op_text} {num2}?"
            icl_ans = "What"
            wrong_ans_int = op_math((num1, num2))
            wrong_ans = str(wrong_ans_int)
            return prompt, icl_ans, wrong_ans

    # Prefixes and instructions for English
    prefixes = ('Input: ', 'Output: ')
    instr = {
        'repeat': 'Repeat the input.\n\n',
        'capslock': 'Capitalize every character.\n\n',
    }[ICL_TASK]

    def concat_examples(samples):
        ret_sample = instr
        for i, sample in enumerate(samples):
            if i != 0: 
                ret_sample += "\n\n"
            if i != len(samples) - 1: 
                ret_sample += prefixes[0] + sample + '\n'
                ret_sample += prefixes[1] + icl_task(sample)
            else:
                if_task_prompt, icl_ans, if_ans = if_task(sample)
                icl_ans = icl_task(icl_ans)
                if_ans = icl_task(if_ans)
                ret_sample += prefixes[0] + if_task_prompt + '\n'
                ret_sample += prefixes[1][:-1]
        return ret_sample, icl_ans, if_ans

    def generate_sample(kshot=4):
        prompt, icl_ans, if_ans = concat_examples(random.sample(sentences, kshot+1))
        return {
            "prompt": prompt,
            "icl_ans": ' ' + icl_ans,
            "if_ans": ' ' + if_ans,
            "config": f'{ICL_TASK}-{IF_TASK}-{INSTRUCTION_TEMPLATE}-{PROMPT_TEMPLATE}-en-{kshot}shot',
        }
    
    random.seed(10)
    data = [generate_sample(kshot=kshot) for _ in range(100)]

    df = pd.DataFrame.from_dict(data)
    df.to_csv(f"/kaggle/working/understanding-forgetting/icl_vs_if/in_csvs/{ICL_TASK}-{IF_TASK}-{INSTRUCTION_TEMPLATE}-{PROMPT_TEMPLATE}-en-{kshot}shot.csv")
