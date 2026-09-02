"""Cache-backend integrations for CacheVerifier.

Each integration wraps `cacheverifier.CacheVerifier` in the extension point
a given cache library already exposes, so verification is a drop-in change
with no upstream fork required. GPTCache is the first; the same pattern
applies to any backend with a "should I trust this candidate" hook.
"""
