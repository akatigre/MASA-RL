import math
import json
import os
import argparse

def pass_at_k(n, c, k):
    assert k <= n, f'k should be less than n but got k={k}, n={n}'

    return 1 - (math.comb(n - c, k) / math.comb(n, k))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', type=str, default='./eval')
    parser.add_argument('--k', type=str, default='')
    args = parser.parse_args()

    k_list = [int(x) for x in args.k.split(',')] if args.k else []

    eval_path = os.path.abspath(args.eval_path)

    for file in sorted(os.listdir(eval_path)):
        if file.endswith('.json'):
            with open(os.path.join(eval_path, file), "r") as f:
                data = json.load(f)

        pass_at_k_dict = {}
        first_elem = next(iter(data.values()))
        num_responses = len(first_elem['preds'])

        if k_list:
            pass_at_k_list = [k for k in k_list if k <= num_responses]
        else:
            pass_at_k_list = [1, num_responses]

        for k in pass_at_k_list:
            pass_at_k_dict[k] = []

        for key, value in data.items():
            for k in pass_at_k_dict.keys():
                pass_at_k_dict[k].append(pass_at_k(len(value['preds']), int(sum(value['score_list'])), k))

        print(f"Pass@k for {file}")
        for k, v in pass_at_k_dict.items():
            print(f"Pass@{k} = {sum(v)/len(v):.6f}")

if __name__ == "__main__":
    main()