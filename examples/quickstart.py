"""CacheVerifier quickstart -- verify one gray-zone cache hit, then report
the outcome back. Runs against the live API.

    pip install cacheverifier
    CACHEVERIFIER_API_KEY=cv_... python quickstart.py

Get a free key at https://www.cacheverifier.com (self-serve verify and
fine-tuning are free forever).
"""

import os

from cacheverifier import CacheVerifier


def call_your_llm(query: str) -> str:
    """Stand-in for your own LLM call. CacheVerifier doesn't do this part --
    your cache backend already has one."""
    return "Go to Settings > Billing > Cancel subscription."


def main() -> None:
    api_key = os.environ["CACHEVERIFIER_API_KEY"]

    # A gray-zone example: your cache's similarity search found a candidate
    # close enough to look like a match, but "cancel" vs "pause" is a real
    # difference a plain similarity threshold can't tell apart.
    query = "how do I cancel my subscription"
    candidate = "Go to Settings > Billing > Pause subscription for a month."

    with CacheVerifier(api_key=api_key) as cv:
        result = cv.verify(query, candidate)
        print(f"verify -> approved={result.approved} score={result.score:.3f} "
              f"threshold={result.threshold} model={result.model_version}")

        if result.approved:
            answer = candidate
            print("approved: serving the cached answer, no LLM call made.")
        else:
            answer = call_your_llm(query)
            print(f"rejected: fell through to your LLM -> {answer!r}")

        # Once you know the real outcome (thumbs-down, reopened ticket,
        # manual review), report it. This is what fine-tuning and drift
        # monitoring learn from.
        row_id = cv.feedback(query, answer, was_correct=True, similarity_score=0.86)
        print(f"feedback recorded (row {row_id})")


if __name__ == "__main__":
    main()
