from datasets import load_dataset
import random
import re
from typing import Optional
import json
import os

CHOICE_TOKENS = tuple("ABCDEFGH")

_BOXED_RE = re.compile(r"""\\boxed\s*\{\s*([A-Ha-h])\s*\}""")
_BRACKET_RE = re.compile(r"""[\(\[\{]\s*([A-Ha-h])\s*[\)\]\}]""")
_STANDALONE_RE = re.compile(r"""\b([A-Ha-h])\b""")
_PREFIXED_RE = re.compile(r"""^(?:option|answer|ans|choice)\s*[:\-]?\s*([A-Ha-h])\b""", re.IGNORECASE)

MULTIPLE_CHOICE_DATASETS = ['rbench', 'gpqa_diamond', 'arc_challenge', "AR-LSAT", "FOLIO", "LogicalDeduction", "ProntoQA", "ProofWriter"]
LOGIC_DATASETS = ["AR-LSAT", "FOLIO", "LogicalDeduction", "ProntoQA", "ProofWriter"]
CODE_DATASETS = ["MBPP"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def rel_path(path):
    return os.path.join(BASE_DIR, path)

def build_prompt_texts(item, tokenizer):
    prompt_texts = []
    prompt_texts.append(tokenizer.apply_chat_template(
            item["prompt"], tokenize=False, add_generation_prompt=True
        ))
    return prompt_texts

def load_dataset_and_in_context_examples(dataset_name):
    assert dataset_name in LOGIC_DATASETS, f"Invalid dataset: {dataset_name}"

    data_path = rel_path(f"./data/{dataset_name}/dev.json")
    icl_path = rel_path(f"./data/icl_examples/{dataset_name}_CoT.txt")

    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    with open(icl_path, 'r') as f:
        icl_example = f.read()

    return raw_data, icl_example

def _simple_data_processing(data, id, problem, answer):
    processed_data = []

    for idx, item in enumerate(data):
        item['extra_info'] = {"index": item[id] if id is not None else idx}

        if "prompt" not in item:
            item["prompt"] = [{"role": "user", "content": item[problem]}]

        item["reward_model"] = {"ground_truth": {"target": item[answer] if answer != "final_answer" else item[answer][0]}}
        processed_data.append(item)

    data = processed_data
    return data

def _logic_data_processing(raw_data, icl_example):
    processed_data = []

    for item in raw_data:
        prompt_text = prompt_LSAT(icl_example, item)

        processed_data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
                "reward_model": {"ground_truth": {"target": item['answer']}},
                "extra_info": {"index": item.get('id', None)}
            })

    return processed_data

def _mbpp_data_processing(raw_data):
    processed_data = []

    for item in raw_data:
        processed_data.append({
            "prompt": [{"role": "user", "content": item['prompt']}],
                "reward_model": {"ground_truth": {"target": None}},
                "extra_info": {"index": item.get('task_id', None)}
            })

    return processed_data

def _create_question_gpqa(item):
    random.seed(42)
    q = item['Question']
    answers = [
        item['Correct Answer'],
        item['Incorrect Answer 1'], 
        item['Incorrect Answer 2'],
        item['Incorrect Answer 3']
    ]
    shuffled_answers = answers.copy()
    random.shuffle(shuffled_answers)
    option_labels = ["A", "B", "C", "D"]
    correct_label = None

    for i, answer in enumerate(shuffled_answers):
        if answer == item['Correct Answer']:
            correct_label = option_labels[i]
            break

    options_str = "\n".join([f"{option_labels[i]}: {shuffled_answers[i]}" for i in range(len(option_labels))])
    question_str = (
        "Think step-by-step and choose the best answer from the following options. "
        "Answer only with the letter of the option inside \\boxed{}.\n"
        f"{q}\n{options_str}"
    )
    return {
        "prompt": [{"role": "user", "content": question_str}],
        "correct_answer_label": correct_label
    }

def _create_question_rbench(item):
    q = item['question']
    option_labels = ["A", "B", "C", "D", "E", "F"]
    options = [f"{opt}: {item[opt]}" for opt in option_labels if opt in item]
    options_str = "\n".join(options)
    question_str = (
        "Think step-by-step and choose the best answer from the following options. Answer only with the letter of the option inside \\boxed{}.\n"
        f"{q}\n{options_str}"
    )
    return {"prompt": [{"role": "user", "content": question_str}]}

def _create_question_arc_challenge(item):
    labels = item["choices"]["label"]
    texts  = item["choices"]["text"]
    options_str = "\n".join(f"{lbl}: {txt}" for lbl, txt in zip(labels, texts))
    question_str = (
        "Think step-by-step and choose the best answer from the following options. "
        "Answer only with the letter of the option inside \\boxed{}.\n"
        f"{item['question']}\n{options_str}"
    )
    return {"prompt": [{"role": "user", "content": question_str}]}

def build_dataset(dataset_name):
    if dataset_name in ["math", "amc23", "aime2025"]:
        data_path = f"../data/{dataset_name}/test.parquet"
        data = load_dataset("parquet", data_files=data_path)['train']

    elif dataset_name == "math500":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        data = _simple_data_processing(data, "unique_id", "problem", "answer")

    elif dataset_name == "minerva":
        data = load_dataset("math-ai/minervamath", split="test")
        data = _simple_data_processing(data, None, "question", "answer")

    elif dataset_name == "olympiad_math":
        data = load_dataset("math-ai/olympiadbench", split="test")
        data = _simple_data_processing(data, "id", "question", "final_answer")

    elif dataset_name == "aime2024":
        data = load_dataset("HuggingFaceH4/aime_2024", split="train")
        data = _simple_data_processing(data, "id", "problem", "answer")

    elif dataset_name == "rbench":
        data = load_dataset('R-Bench/R-Bench', "rbench-t_en", split='test')
        data = data.map(_create_question_rbench)
        data = _simple_data_processing(data, "index", None, "answer")

    elif dataset_name == "gpqa_diamond":
        data = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', split="train")
        data = data.map(_create_question_gpqa) 
        data = _simple_data_processing(data, "Record ID", None, "correct_answer_label")

    elif dataset_name == "arc_challenge":
        data = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        data = data.map(_create_question_arc_challenge)
        data = _simple_data_processing(data, "id", None, "answerKey")

    elif dataset_name == "scibench":
        data = load_dataset("xw27/scibench", split="train")
        data = _simple_data_processing(data, "problemid", "problem_text", "answer_number")

    elif dataset_name in ["AR-LSAT", "FOLIO", "LogicalDeduction", "ProntoQA", "ProofWriter"]:
        raw_data, icl_example = load_dataset_and_in_context_examples(dataset_name)
        data = _logic_data_processing(raw_data, icl_example)

    elif dataset_name == "MBPP":
        examples = list(read_test_examples(rel_path("./data/MBPP/mbpp.jsonl")))
        data = _mbpp_data_processing(examples)

    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    return data
    
def get_choice(answer_str: str) -> Optional[str]:
    if not answer_str:
        return None

    s = str(answer_str).strip()
    s = s[-2000:]

    for rx in (_BOXED_RE, _BRACKET_RE, _PREFIXED_RE):
        m = rx.search(s)
        if m:
            return m.group(1).upper()

    if len(s) >= 1:
        ch = s[0].upper()
        if ch in CHOICE_TOKENS:
            if len(s) == 1:
                return ch
            if len(s) >= 2 and s[1] in (')','.',':','-',' '):
                return ch

    indicators = [
        'the correct option is', 'the correct answer is',
        'thus, the answer is', 'answer is', 'answer:', 'option is'
    ]
    low = s.lower()

    for ind in indicators:
        idx = low.find(ind)

        if idx >= 0:
            tail = s[idx+len(ind):].strip()
            ch = get_choice(tail)

            if ch:
                return ch

    m = _STANDALONE_RE.search(s)

    if m:
        ch = m.group(1).upper()

        if ch in CHOICE_TOKENS:
            return ch

    return None

def is_multiple_choice(name: str) -> bool:
    return any(dataset in name for dataset in MULTIPLE_CHOICE_DATASETS)

def is_logic_dataset(name: str) -> bool:
    return any(dataset in name for dataset in LOGIC_DATASETS)

def is_code_dataset(name: str) -> bool:
    return any(dataset in name for dataset in CODE_DATASETS)

def prompt_LSAT(in_context_example: str, test_example: dict) -> str:
    full_prompt = in_context_example
    context = test_example['context'].strip()
    question = test_example['question'].strip()
    options = '\\n'.join([opt.strip() for opt in test_example['options']])
    full_prompt = full_prompt.replace('[[CONTEXT]]', context)
    full_prompt = full_prompt.replace('[[QUESTION]]', question)
    full_prompt = full_prompt.replace('[[OPTIONS]]', options)
    return full_prompt

def read_test_examples(data_path: str):
    def format_test_example(q, tests, code: str=None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))

        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []

    for i in range(1, 4):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        ex_prompt = format_test_example(q, test, code)
        example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
        examples_str += [example_prompt]

    for i in range(10, 510):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        
        prompt = format_test_example(q, test, code=None)

        prompt_with_shots = '''
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
'''.strip().format('\n\n'.join(examples_str), prompt)
        yield {
            'task_id': ex['task_id'],
            'prompt': prompt_with_shots
        }