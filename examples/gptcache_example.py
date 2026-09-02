"""Wire CacheVerifier into a GPTCache pipeline as the similarity evaluator.

    pip install "cacheverifier[gptcache]"
    CACHEVERIFIER_API_KEY=cv_... python gptcache_example.py

The only CacheVerifier-specific line is `similarity_evaluation=...`. Every
gray-zone lookup GPTCache would otherwise decide on cosine similarity alone
now gets one synchronous verifier call instead.
"""

import os

from gptcache import cache
from gptcache.adapter.api import get, put
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager

from cacheverifier.integrations.gptcache import CacheVerifierEvaluation

embedding = Onnx()
data_manager = get_data_manager(
    CacheBase("sqlite"),
    VectorBase("faiss", dimension=embedding.dimension),
)

evaluator = CacheVerifierEvaluation(api_key=os.environ["CACHEVERIFIER_API_KEY"])

cache.init(
    embedding_func=embedding.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=evaluator,
)

put("how do I cancel my subscription?", "Go to Settings > Billing > Cancel subscription.")

# Close-but-different intent. Cosine similarity would likely reuse the
# cached answer; the verifier call decides whether that is actually safe.
hit = get("how do I pause my subscription for a month?")
print("cache returned:", hit)

# When you later learn whether a served hit was right:
evaluator.report_feedback(
    "how do I pause my subscription for a month?",
    hit or "",
    was_correct=False,
)
