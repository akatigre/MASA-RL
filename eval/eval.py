
import json
import json
import os
import argparse
from tqdm import tqdm
from utils import get_choice, is_multiple_choice, is_logic_dataset, is_code_dataset, rel_path
from verl.utils.reward_score.math import compute_score
from verl.utils.reward_score.utils import extract_answer_math, strip_string
from human_eval.evaluation import evaluate_functional_correctness

LABEL_PHRASE = "The correct option is:"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_path', type=str, default='./response')
    parser.add_argument('--eval_path', type=str, default='./eval')
    args = parser.parse_args()

    response_path = os.path.abspath(args.response_path)
    eval_path = os.path.abspath(args.eval_path)
    os.makedirs(eval_path, exist_ok=True)

    for input_path in os.listdir(response_path):

        if input_path.endswith('.json'):
            input_path = os.path.join(response_path, input_path)

            if os.path.exists(input_path.replace(response_path, eval_path)):
                print(f"Skipping {input_path} because eval file already exists")
                continue

        else:
            continue

        print(f"Processing {input_path}")

        if is_code_dataset(input_path):
            temp_dir = "./tmp"
            os.makedirs(temp_dir, exist_ok=True)
            pass_at_k, results= evaluate_functional_correctness(
                input_file=input_path,
                tmp_dir=temp_dir,
                problem_file=rel_path("./data/MBPP/mbpp_test.jsonl"),
                language='python',
                is_mbpp=True
            )

            eval_dict = {}

            for key, value in results.items():
                eval_dict[key] = {}
                preds = []
                scores = []

                for item in value:
                    pred = item[1]['code']
                    score = int(item[1]['passed'])
                    preds.append(pred)
                    scores.append(score)

                eval_dict[key]['preds'] = preds
                eval_dict[key]['score_list'] = scores
                eval_dict[key]['score'] = sum(scores) / len(scores) if len(scores) > 0 else 0.0

        else:
            with open(input_path, 'r') as f:
                data = json.load(f)

            eval_dict = {}

            for key, value in tqdm(data.items()):
                eval_dict[key] = {}
                gt = value['answer']
                preds = [obj for obj in value['answer_response']]

                if is_multiple_choice(input_path):
                    gt = get_choice(gt)
                    preds_parsed = []
                    score_list = []

                    for pred in preds:
                        if is_logic_dataset(input_path):
                            if LABEL_PHRASE in pred:
                                pred = pred.split(LABEL_PHRASE)[-1].strip()
                            else:
                                pred = pred.strip()
                        ch = get_choice(str(pred))
                        preds_parsed.append(ch if ch is not None else "N/A")
                        score_list.append(int(ch == gt))

                    score = (sum(score_list) / len(score_list)) if len(score_list) > 0 else 0.0

                    answer_list = preds_parsed

                else:
                    ground_truth = {'question': value['problem'], 'target': strip_string(gt)}
                    score_list = [compute_score(pred, ground_truth) for pred in preds]
                    score = sum(score_list) / len(score_list) if len(score_list) > 0 else 0.0

                    answer_list = []

                    for pred in preds:
                        try:
                            answer_list.append(extract_answer_math(pred[-1000:]))

                        except:
                            answer_list.append("N/A")

                eval_dict[key]['score'] = score
                eval_dict[key]['score_list'] = score_list
                eval_dict[key]['preds'] = answer_list
                eval_dict[key]['gt'] = gt

        save_path = input_path.replace(response_path, eval_path)

        with open(save_path, 'w') as f:
            json.dump(eval_dict, f, indent=4)
        
if __name__ == "__main__":
    main()