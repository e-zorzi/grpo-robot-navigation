import json
from datetime import datetime
COMPLETIONS_LOG = "/kaggle/working/completions_log.jsonl"
import re
import logging
import numpy as np
import ast
logger = logging.getLogger(__name__)

_RUBRIC_PROMPT = """For each rubric criterion in these rubrics\n{RUBRICS}\nEvaluate whether the reasoning '{REASONING}' satisfy it.\n
If a rubric is satisfied, return 1, else 0. Return a list '[]' filled with these values, one for each rubric.
"""

COLOR_WORDS = [
    "white", "black", "red", "blue", "green",
    "yellow", "brown", "gray", "grey", "orange",
    "purple", "pink", "dark", "light", "bright",
    "coral", "beige", "cream", "golden", "silver",
    "teal", "navy", "wooden", "marble",
]

TEXTURE_WORDS = [
    "wood", "wooden", "marble", "glass", "metal",
    "plastic", "fabric", "leather", "smooth", "rough",
    "polished", "matte", "glossy", "grain", "laminate",
    "painted", "natural", "rustic", "concrete", "brick",
    "velvet", "steel", "chrome", "ceramic", "linen",
]

SPATIAL_WORDS = [
    "left", "right", "center", "middle", "corner",
    "against", "beside", "near", "next", "above",
    "below", "front", "back", "wall", "floor",
    "window", "positioned", "placed", "adjacent",
    "behind", "facing", "side", "top", "bottom",
]

from monorepo import AsyncClientBasedLLM, ClientBasedLLM
import time

MODEL_NAME = "Qwen/Qwen3-30B-A3B"  # swap for your checkpoint (local path or HF repo id)

_ASYNC_CLIENT = AsyncClientBasedLLM(model_id=MODEL_NAME)
def ask_batch_prompts_async(prompts):
    now = time.time()
    responses = _ASYNC_CLIENT.ask_batch(prompts)
    later = time.time()
    #print(f"> Took {later - now}")
    return responses

#_CLIENT = ClientBasedLLM(model_id=MODEL_NAME)
# def ask_batch_prompts(prompts):
#     now = time.time()
#     responses = [_CLIENT.ask(prompt=p) for p in prompts]
#     later = time.time()
#     print(f"> Took {later - now}")
#     return responses



def score_reward_func(completions, score, **kwargs):
    pattern = r"<score>(\d+)</score>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content) for content in completion_contents]

    # -9239 is a placeholder (very unlikely value) when score is not available. Here we don't give 
    # negative reward as we have already penalized the lack of a correct template
    completion_scores = [int(match.group(1)) if match else -9239 for match in matches]

    # If the scores match, reward is 1.0 | if completion score is -9239 then it was invalid 
    # therefore no reward | else (the completion score was valid but does not match) no reward
    return [1.0 if score_gt == score_compl else None if score_compl == -9239 else 0.0 for score_gt, score_compl in zip(score, completion_scores)]

def reasoning_reward_func(completions, rubrics, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    rubric_points = np.array(list(map(lambda x: x['points'], rubrics[0])))
    pattern = r"<motivation>(.+)</motivation>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content) for content in completion_contents]
    completion_reasonings = [match.group(1) if match else None for match in matches]
    scores = []

    prompts = [
        _RUBRIC_PROMPT.format(
            RUBRICS=[x["criterion"] for x in rubrics[i]],
            REASONING=completion_reasonings[i],
        )
        for i in range(len(completion_reasonings))
    ]

    responses = ask_batch_prompts_async(prompts)
        
    for response in responses:
        try:
            if "</think>" in response:
                response = response.split("</think>")[1].lstrip(" \n").rstrip(" \n")
            else:
                response = response.lstrip(" \n").rstrip(" \n")
            # Eval
            if response.startswith("[") and response.endswith("]") and "def" not in response and "1" in response and "0" in response:
                evaluation = np.asarray(ast.literal_eval(response), dtype=np.float32)

                scores.append((evaluation*rubric_points).sum().item())
            else:
                scores.append(None)
        except:
            scores.append(None)
    return scores


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<motivation>.*?</motivation><score>.*?</score>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]