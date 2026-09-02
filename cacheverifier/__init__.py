"""cacheverifier -- Python client for the hosted CacheVerifier API.

CacheVerifier verifies semantic-cache hits: given a query and a candidate
cached answer, it approves or rejects serving that answer from cache, so a
similarity match that is close but wrong doesn't become a silent error.

    from cacheverifier import CacheVerifier

    cv = CacheVerifier(api_key="cv_...")
    if cv.verify(query, candidate_answer).approved:
        ...  # serve from cache
    else:
        ...  # fall through to your LLM

Docs: https://www.cacheverifier.com/docs
Research behind it: https://github.com/imxinchengyou/CacheVerifier
"""

from cacheverifier.client import CacheVerifier, CacheVerifierError, VerifyResult

__version__ = "0.2.0"
__all__ = ["CacheVerifier", "CacheVerifierError", "VerifyResult", "__version__"]
