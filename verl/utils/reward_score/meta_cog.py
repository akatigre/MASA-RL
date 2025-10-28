# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import math
import json
import random


def is_json_text(s: any) -> bool:
    """Return True if `s` is a str containing valid JSON."""
    if not isinstance(s, str):
        return False
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def singularize(word: str) -> str:
    """
    Convert a plural noun to its singular form for regular English plurals.
    Examples:
      - identities -> identity
      - cosines    -> cosine
      - medians    -> median
    """
    w = word.lower()
    # 1) zzzies → zzy  (e.g. identities → identity)
    if w.endswith("ies") and w != "series":
        return word[:-3] + "y"

    if w.endswith(("ches", "shes", "xes", "zes", "oes")):
        return word[:-2]
    # 2) zzs → zz  (e.g. cosines → cosine, medians → median)
    #    but avoid chopping -ss (e.g. “glass” → “glas”?)
    if w.endswith("s") and not w.endswith("ss") and w != "series":
        return word[:-1]
    # 3) otherwise, return as-is
    return word


def compute_score(solution_str:dict, ground_truth:dict) -> float:
    """
    ground_truth: includes a list of "problem", "reasoning", "response", "score", "length"
    solution_str: json string with three keys: math_notion, solution_length, pass_rate
    """
    score = 0
    notion_score = 0
    pred_acc_score = 0
    length_score = 0
    
    do_print = random.randint(1, 64) == 1

    has_notion = int("math_notion" in solution_str)
    has_length = int("solution_length" in solution_str) and isinstance(solution_str["solution_length"], int)
    has_difficulty = int("pass_rate" in solution_str and isinstance(solution_str["pass_rate"], int))

    #! Notion score if the notion is included in the response
    if has_notion:
        predicted_notions = solution_str.get("math_notion", "")
        try:
            if "," in predicted_notions:
                predicted_notions = predicted_notions.split(",")
                predicted_notions = [notion.strip().strip("[").strip("]") for notion in predicted_notions]
            elif isinstance(predicted_notions, list):
                pass
            else:
                predicted_notions = [predicted_notions]

            # stopwords = [
            #     "theorem", "formula", "definition", "lemma", "theorems", "formulae", "definitions", "lemmas",
            #     "corollary", "proposition", "postulate", "corollaries", "propositions", "postulates",
            #     "property", "rule", "principle", "law", "concept", "axiom", "properties", "rules", "principles", "laws", "concepts", "axioms",
            #     "law of", "principle of", "rule of", "theorem of", "formula of", "equation of", "definition of", "lemma of", "corollary of", "proposition of", "postulate of", "corollaries of", "propositions of", "postulates of", "properties of", "rules of", "principles of", "laws of", "concepts of", "axioms of",
            # ]

            # stopwords = sorted(stopwords, key=len, reverse=True)
            # pattern = re.compile(
            #     r'(?:(?<=\s)|^)(?:' + '|'.join(map(re.escape, stopwords)) + r')\s*',
            #     re.IGNORECASE
            # )
            predicted_notions = [notion for notion in list(map(singularize, predicted_notions)) if notion not in ground_truth["problem"]]
            correct_resps = [resp for resp, correct in zip(ground_truth["response"], ground_truth["score"]) if correct == 1]
            incorrect_resps = [resp for resp, correct in zip(ground_truth["response"], ground_truth["score"]) if correct == 0]
            notion_dict = {notion: 0 for notion in predicted_notions}
            
            for resp in correct_resps:
                for notion in predicted_notions:
                    if notion in resp:
                        notion_dict[notion] += 1
                        
            for resp in incorrect_resps:
                for notion in predicted_notions:
                    if notion in resp:
                        notion_dict[notion] -= 1
                        
            score_seq = [1 if cnt > 0 else 0 for cnt in notion_dict.values()]
            if len(score_seq):
                notion_score = sum(score_seq) / len(score_seq)
        except Exception as e:
            has_notion = False
            print(f"Error in notion score: {e}")      
    
    if has_length:
        try:
            correct_length = [l for l, correct in zip(ground_truth["length"], ground_truth["score"]) if correct == 1]
            pred_length = int(solution_str['solution_length'])
            if len(correct_length) == 0:
                length_score = 0
            else:
                length_score = int(max(correct_length) > pred_length and pred_length > min(correct_length))
        except Exception as e:
            has_length = False
            print(f"Error in length score: {e}")
        
    if has_difficulty:
        try:
            average_score = sum(
                ground_truth["score"]) / len(ground_truth["score"])
            pred_score = solution_str["pass_rate"]
            pred_score /= 8 
            score_diff = abs(average_score - pred_score)
            pred_acc_score = (0.01) ** score_diff
        except Exception as e:
            has_difficulty = False
            print(f"Error in difficulty score: {e}")


    score = (notion_score + pred_acc_score + length_score) / 3

    if do_print:
        if has_notion:
            print(f"Predicted Notion Score: {notion_score} | predicted_notions: {predicted_notions}")
        if has_length:
            print(f"Predicted Length Score: {length_score} | pred_length: {pred_length} | length: {ground_truth['length']}")  
        if has_difficulty:
            print(
                f"Predicted Accuracy Score: {pred_acc_score} | pred_acc: {pred_score} | average_score: {average_score}")

        print(f"Total Score: {score}")

    return {
            "score": score,
            "notion_score": notion_score,
            "pred_acc_score": pred_acc_score,
            "length_score": length_score,
        }