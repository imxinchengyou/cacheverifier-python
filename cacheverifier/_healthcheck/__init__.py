"""`cacheverifier healthcheck` -- run the hosted Health Check Report's
stock-vs-fine-tuned AUC evaluation entirely on your own machine.

Everything under here needs the `healthcheck` extra
(`pip install "cacheverifier[healthcheck]"`: torch, sentence-transformers,
scikit-learn, numpy) and is imported only when the subcommand runs.
"""
