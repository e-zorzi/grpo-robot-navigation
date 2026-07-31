import json
from datetime import datetime
import re
import logging
import numpy as np
import ast
from config import GRPOConfig
import weave

MAX_RUBRICS = 20
MAX_REASONING_LEN = 800
MAX_RUBRIC_LEN = 300

logger = logging.getLogger(__name__)
_RUBRIC_PROMPT = """For each rubric criterion in these rubrics\n{RUBRICS}\nEvaluate whether the reasoning '{REASONING}' satisfy it.\n
If a rubric is satisfied, return 1, else 0. Return a list filled with these values, one for each rubric. Only include values 1 and 0 in the list,
nothing else.
"""

COLOR_WORDS = [
    "white",
    "black",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "gray",
    "grey",
    "orange",
    "purple",
    "pink",
    "dark",
    "light",
    "bright",
    "coral",
    "beige",
    "cream",
    "golden",
    "silver",
    "teal",
    "navy",
    "wooden",
    "marble",
]

TEXTURE_WORDS = [
    "wood",
    "wooden",
    "marble",
    "glass",
    "metal",
    "plastic",
    "fabric",
    "leather",
    "smooth",
    "rough",
    "polished",
    "matte",
    "glossy",
    "grain",
    "laminate",
    "painted",
    "natural",
    "rustic",
    "concrete",
    "brick",
    "velvet",
    "steel",
    "chrome",
    "ceramic",
    "linen",
]

SPATIAL_WORDS = [
    "left",
    "right",
    "center",
    "middle",
    "corner",
    "against",
    "beside",
    "near",
    "next",
    "above",
    "below",
    "front",
    "back",
    "wall",
    "floor",
    "window",
    "positioned",
    "placed",
    "adjacent",
    "behind",
    "facing",
    "side",
    "top",
    "bottom",
]

from monorepo import AsyncClientBasedLLM, ClientBasedLLM
import time

MODEL_NAME = GRPOConfig.evaluator_model_name

_ASYNC_CLIENT = AsyncClientBasedLLM(model_id=MODEL_NAME)


@weave.op(tracing_sample_rate=0.1)
def ask_batch_prompts_async(prompts):
    return _ASYNC_CLIENT.ask_batch(prompts, max_tokens=64)


def reasoning_reward_func(completions, rubrics, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    rubric_points = [
        np.array(list(map(lambda x: x["points"], r[:MAX_RUBRICS]))) for r in rubrics
    ]  # max 10 rubrics
    pattern = r"<motivation>(.+)</motivation>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content) for content in completion_contents]
    completion_reasonings = [
        match.group(1)[:MAX_REASONING_LEN] if match else None for match in matches
    ]
    scores = []

    prompts = []
    for i in range(len(completion_reasonings)):
        try:
            prompts.append(
                _RUBRIC_PROMPT.format(
                    RUBRICS=[
                        x["criterion"][:MAX_RUBRIC_LEN]
                        for x in rubrics[i][:MAX_RUBRICS]
                    ],
                    REASONING=completion_reasonings[i],
                )
            )
        except:  # noqa
            prompts.append("Just return None")

    try:
        responses = ask_batch_prompts_async(prompts)
    except Exception as e:
        print(e)
        print(
            "Problem with the client. The responses will be set to None, with no reward assigned for this batch"
        )
        responses = [None for _ in range(len(prompts))]

    for j, response in enumerate(responses):
        try:
            if "</think>" in response:
                response = response.split("</think>")[1].lstrip(" \n").rstrip(" \n")
            else:
                response = response.lstrip(" \n").rstrip(" \n")
            # Eval
            evaluation = np.asarray(ast.literal_eval(response), dtype=np.float32)
            scores.append((evaluation * rubric_points[j]).sum().item() / 10.0)
        except:
            scores.append(None)
    return scores


def score_reward_func(completions, score, **kwargs):
    pattern = r"<score>(\d+)</score>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content) for content in completion_contents]

    # -9239 is a placeholder (very unlikely value) when score is not available. Here we don't give
    # negative reward as we have already penalized the lack of a correct template
    completion_scores = [int(match.group(1)) if match else -9239 for match in matches]

    # If the scores match, reward is 1.0 | if completion score is -9239 then it was invalid
    # therefore no reward | else (the completion score was valid but does not match) no reward
    return [
        1.0 if score_gt == score_compl else None if score_compl == -9239 else 0.0
        for score_gt, score_compl in zip(score, completion_scores)
    ]


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<motivation>.*?</motivation><score>.*?</score>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
