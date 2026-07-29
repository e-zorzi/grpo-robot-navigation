import json
from datetime import datetime
COMPLETIONS_LOG = "/kaggle/working/completions_log.jsonl"
import re
import logging
from cerebras_client import evaluate_reasoning

logger = logging.getLogger(__name__)

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
    return [0.0 for _ in range(len(completions))]
    #return evaluate_reasoning(prompts, completions, **kwargs)

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<motivation>.*?</motivation><score>.*?</score>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

# def compute_reasoning_reward(motivation, gt_reasoning):
#     if not motivation or not gt_reasoning:
#         return 0.0
#     m = motivation.lower()
#     g = gt_reasoning.lower()
#     gt_colors = set(w for w in COLOR_WORDS if w in g)
#     pred_colors = set(w for w in COLOR_WORDS if w in m)
#     color = len(pred_colors & gt_colors) / len(gt_colors) if gt_colors else 0.5
#     gt_tex = set(w for w in TEXTURE_WORDS if w in g)
#     pred_tex = set(w for w in TEXTURE_WORDS if w in m)
#     texture = len(pred_tex & gt_tex) / len(gt_tex) if gt_tex else 0.5
#     gt_spa = set(w for w in SPATIAL_WORDS if w in g)
#     pred_spa = set(w for w in SPATIAL_WORDS if w in m)
#     spatial = len(pred_spa & gt_spa) / len(gt_spa) if gt_spa else 0.5
#     reasoning_r = (color + texture + spatial) / 3.0
#     print(f"    color={color:.2f} texture={texture:.2f} spatial={spatial:.2f} → reasoning={reasoning_r:.2f}")
#     return reasoning_r

def reward_function(prompts, completions, reasoning, score, alpha=0.5, beta=0.3, gamma=0.2, **kwargs):
    rewards = []
    for completion, gt_reasoning, gt_score in zip(completions, reasoning, score):
        if isinstance(completion, list):
            text = completion[0]["content"]
        else:
            text = str(completion)

        # ✅ Log full completion to disk
        with open(COMPLETIONS_LOG, "a") as f:
            json.dump({
                "timestamp":        datetime.now().isoformat(),
                "completion":       text,
                "follows_template": text.strip().startswith("<motivation>"),
                "gt_score":         int(gt_score),
            }, f)
            f.write("\n")

        print(f"  COMPLETION: {text}")
       
        motivation, pred_score = extract_motivation_and_score(text)
        template_r = compute_template_reward(text)
        score_r = score_reward_func(pred_score, gt_score)
        reasoning_r = 0.0
        if motivation:
            reasoning_r = reasoning_reward_func(motivation, gt_reasoning)
        final = alpha * reasoning_r + beta * score_r + gamma * template_r
        final = max(-5.0, min(1.0, final))
        rewards.append(final)
        print(f"  template={template_r:.2f} score={score_r:.2f} reasoning={reasoning_r:.2f} → final={final:.2f}")
    return rewards
