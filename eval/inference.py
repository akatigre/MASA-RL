import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
import argparse
from utils import build_prompt_texts, build_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B", help="Path to the model checkpoint or model name from Hugging Face.")
    parser.add_argument("--dataset", type=str, choices=["math", "amc23", "aime2025", "math500", "minerva", "olympiad_math", "aime2024", "rbench", "gpqa_diamond", "arc_challenge", "scibench", "AR-LSAT", "FOLIO", "LogicalDeduction", "ProntoQA", "ProofWriter", "MBPP"], default="math500")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=20000)
    parser.add_argument("--max_new_token", type=int, default=16384)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--n", type=int, default=32)
    args = parser.parse_args()

    if args.model_path[-1] == "/":
        args.model_path = args.model_path[:-1]

    model_path = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model               = model_path,
        dtype               = args.dtype,
        trust_remote_code   = True,
        tensor_parallel_size= args.tp,
        max_model_len=args.max_model_len,
    )
    

    n = 1 if args.temperature == 0.0 else args.n

    sampler = SamplingParams(
        max_tokens   = args.max_new_token,
        temperature  = args.temperature,
        top_p        = 1.0,
        n            = n,
    )

    data = build_dataset(args.dataset)

    all_prompts = []

    for item in tqdm(data, total=len(data)):
        all_prompts.extend(build_prompt_texts(item, tokenizer))

    outputs = llm.generate(all_prompts, sampler)

    log_response = {}
    for i, data_row in enumerate(data):
        answer = [o.text for o in outputs[i].outputs]
        log_response[data_row["extra_info"]['index']] = {
            "problem": data_row["prompt"][0]['content'],
            "answer": data_row["reward_model"]['ground_truth']['target'],
            "answer_response": answer,
        }

    model_name = "_".join(args.model_path.split('/'))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    out_file = f"{args.output_dir}/{model_name}_{args.dataset}_{args.temperature}_{args.n}_vllm.json"

    with open(out_file, "w") as f:
        json.dump(log_response, f, indent=2, ensure_ascii=False)

    print("saved →", out_file)

if __name__ == "__main__":
    main()
