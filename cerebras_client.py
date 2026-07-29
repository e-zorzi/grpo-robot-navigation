import re
from time import sleep
import logging
from typing import Optional
from openai import OpenAI, RateLimitError, APIError
from prompts import COLOR_PROMPT, TEXTURE_PROMPT, SPATIAL_PROMPT
from config import (
    EVALUATOR_MODEL_NAME
)

from monorepo import load_api_keys, CerebrasLLM

load_api_keys()
logger = logging.getLogger(__name__)


class CerebrasClient:
    def __init__(self):
        self.client = CerebrasLLM(model_id=EVALUATOR_MODEL_NAME) 
        logger.info(f"Cerebras client ready (model: {EVALUATOR_MODEL_NAME})")

    def evaluate_reasoning(
        self,
        predicted_motivation: str,
        ground_truth_reasoning: str,
    ) -> float:
        # 3-axis evaluation: color accuracy, texture/material, spatial context
        
        aggregated_score = 0.0
        for prompt in [COLOR_PROMPT, TEXTURE_PROMPT, SPATIAL_PROMPT]:
            formatted = prompt.format(GT=ground_truth_reasoning, REASONING=predicted_motivation)
            answer = self.client.ask(prompt=formatted)
            sleep(0.2)
            try:
                score = float(answer.strip())
            except ValueError:
                score = 0.0
            aggregated_score += score

        return aggregated_score / 3.0

        # for attempt in range(1, self.max_retries + 1):
        #     try:
        #         response = self.client.chat.completions.create(
        #             model=self.model,
        #             messages=messages,
        #             max_tokens=10,
        #             temperature=0.1,
        #         )
        #         print("Cerebras response:", response.choices[0].message.content.strip())

        #         text = response.choices[0].message.content.strip()

        #         score_match = re.search(r"[1-5]", text)
        #         if score_match:
        #             raw_score = int(score_match.group())
        #             normalized = (raw_score - 1) / 4.0
        #             logger.debug(f"Reasoning score: {raw_score}/5 = {normalized:.2f}")
        #             return normalized

        #         logger.warning(f"Could not parse score from: {text}")
        #         return 0.5

        #     except RateLimitError:
        #         wait = self.retry_delay * (2 ** (attempt - 1))
        #         logger.warning(f"Rate limit! Waiting {wait}s... (attempt {attempt})")
        #         time.sleep(wait)
        #         if attempt == self.max_retries:
        #             return 0.5

        #     except APIError as e:
        #         logger.error(f"API error: {e}")
        #         return 0.5

        #     except Exception as e:
        #         logger.error(f"Unexpected error: {e}")
        #         return 0.5

        # return 0.5

    def health_check(self) -> bool:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say OK."}],
                max_tokens=5,
            )
            logger.info("Cerebras health check passed!")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Singleton pattern: create once, reuse across module
_client: Optional[CerebrasClient] = None


def get_cerebras_client() -> CerebrasClient:
    global _client
    if _client is None:
        _client = CerebrasClient()
    return _client


def evaluate_reasoning(
    predicted_motivation: str,
    ground_truth_reasoning: str,
) -> float:
    client = get_cerebras_client()
    return client.evaluate_reasoning(
        predicted_motivation,
        ground_truth_reasoning,
    )


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)

    print("Testing Cerebras client...")
    client = CerebrasClient()

    print("\n[1/2] Health check...")
    if client.health_check():
        print("[2/2] Testing evaluation...")
        score = client.evaluate_reasoning(
            predicted_motivation="I see a white marble desk.",
            ground_truth_reasoning="There is a white marble desk against a coral wall."
        )
        print(f"Score: {score:.3f}")
    else:
        print("API not accessible!")