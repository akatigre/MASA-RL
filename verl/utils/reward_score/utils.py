import re
import json
INVALID_ANS = "[invalid]"
INVALID_JSON = "[None]"


def extract_jsons_from_batch(input_ids_batch, tokenizer, skip_special_tokens=True):
    """
    Given batched input_ids and a tokenizer, decodes each input sequence and
    extracts the last JSON substring using extract_last_json_object.

    Args:
        input_ids_batch: An iterable of token ID sequences (e.g., list of lists or tensor).
        tokenizer: A tokenizer with a .decode() method.
        skip_special_tokens: Whether to skip special tokens during decoding.

    Returns:
        List of strings, each being the extracted JSON substring or INVALID_ANS.
    """
    results = []
    for ids in input_ids_batch:
        # Decode token IDs to text
        text = tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        results.append(extract_last_json_object(text))
    return results


def extract_last_json_object(meta_response_str: str):
    """
    Extracts the last JSON object substring in the input string.
    If no complete {...} block is found, returns INVALID_ANS.
    Returns the substring including braces.
    """
    # depth = 0
    # start_idx = None
    # last_obj_str = None

    # for i, ch in enumerate(input_string):
    #     if ch == "{":
    #         if depth == 0:
    #             start_idx = i
    #         depth += 1
    #     elif ch == "}":
    #         depth -= 1
    #         if depth == 0 and start_idx is not None:
    #             last_obj_str = input_string[start_idx: i + 1]
    #             start_idx = None

    # if last_obj_str is None:
    #     return INVALID_JSON
    # try:
    #     last_obj_str = str(json.loads(last_obj_str))
    # except:
    #     return INVALID_JSON

    # return last_obj_str
    try:
        json_candidates = []
        pattern = r'(\{["\']math_notion["\']:[^\}]*\})' # find a pattern that starts with "{math_notion:" and ends with "}"
        
        for match in re.finditer(pattern, meta_response_str, re.DOTALL):
            json_candidates.append(match.group(1))
        meta_json_str = json_candidates[-1]
        meta_json_dict = json.loads(meta_json_str)
    except:
        meta_json_dict = {}
                    
        notion_match = re.search(r'math_notion\s*:\s*(\[[^\]]*\])', meta_response_str) # find a pattern that starts with "math_notion:"
        if notion_match:
            list_str = notion_match.group(1)
            try:
                math_notion_list = json.loads(list_str)
            except Exception:
                # fallback: try to parse as string list
                math_notion_list = [s.strip().strip('"').strip("'") for s in list_str.strip('[]').split(',') if s.strip()]
            if not len(math_notion_list):
                math_notion_list = None
            meta_json_dict["math_notion"] = math_notion_list
        # Extract integer value (0-10) after 'problem_difficulty:'
        difficulty_match = re.search(r'problem_difficulty\s*:\s*(\d{1,2})', meta_response_str)
        if difficulty_match:
            try:
                problem_difficulty = int(difficulty_match.group(1))
                if not (0 <= problem_difficulty <= 10):
                    problem_difficulty = None
            except Exception:
                problem_difficulty = None
            meta_json_dict["problem_difficulty"] = problem_difficulty
        
        length_match = re.search(r'solution_length\s*:\s*(\d{1,2})', meta_response_str)
        if length_match:
            try:
                solution_length = int(length_match.group(1))
                if not (1024 <= solution_length <= 16384):
                    solution_length = None
            except Exception:
                solution_length = None
            meta_json_dict["solution_length"] = solution_length
    return meta_json_dict


def normalize_float_string(input_string):
    res = input_string.replace(",", "").replace("$", "").replace("%", "")
    if res[-1] == ".":
        res = res[:-1]
    res = float(res)
    res = round(res, 3)
    res = str(res)

    return res


def extract_answer_gsm8k(input_string):
    ans_pattern = re.compile(r"#### (\-?[0-9\.\,]+)$")
    try:
        match = ans_pattern.search(input_string)
    except:
        print("---")
        print(type(input_string))
        print(input_string)
        print("---")
        raise ValueError("Error in regex search")
    try:
        if match:
            match_str = match.group(1).strip()
            match_str = normalize_float_string(match_str)
        elif "he answer is" in input_string:
            match_str = input_string.split("he answer is")[-1].strip()
            match_str = normalize_float_string(match_str)
        else:
            match_str = INVALID_ANS
    except:
        return INVALID_ANS

    return match_str


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    # replace \\ with
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def extract_answer_math(pred_str):
    if 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if len(ans) == 0:
            return ""
        elif (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0):
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        pred = a
    elif ('he answer is' in pred_str):
        pred = pred_str.split('he answer is')[-1].strip()
    # elif extract_program_output(pred_str) != "":
    #     # fall back to program
    #     pred = extract_program_output(pred_str)
    else:  # use the last number
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str.replace(",", ""))
        if (len(pred) >= 1):
            pred = pred[-1]
        else:
            pred = ''

    # multiple line
    pred = pred.split("\n")[0]
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred)
    return pred


if __name__ == "__main__":
    import jsonlines
