# Constraint-Complete Video Retrieval

Preregistered code for deciding whether logic-constrained multimodal untrimmed
video retrieval is a defensible CVPR research direction.

The first stage uses only MUVR metadata:

```powershell
python -m pip install -e ".[test]"
ccvr fetch-metadata --output data/raw
ccvr audit-dataset --raw data/raw --output runs/direction_gate
pytest
```

The audit writes a frozen JSONL manifest, a source lock, a machine-readable
report, and a Markdown decision. Feature and prototype commands refuse to run
unless the preceding gate artifact is accepted.

After the data gate passes, the formal server sequence is:

```bash
bash scripts/run_formal.sh
```

The script streams each feature tar directly into `data/features`, verifies
the Hugging Face LFS SHA-256, evaluates OpenCLIP first, and only permits the
EVA-CLIP download when OpenCLIP passes. It uses GPU 0 only. The scoring command
evaluates five parameter-free baselines and twelve two-parameter monotone
coverage variants, selects the latter on deterministic dev-topic subsamples,
and evaluates the frozen test topics.

`HF_ENDPOINT` may point to an HTTPS mirror when the execution host cannot reach
Hugging Face directly. The dataset commit, file hashes, and LFS hashes remain
frozen and are verified independently of the transport endpoint.

The public MUVR paper reports 93,885 possible single/binary tag queries, while
the current public files may contain fewer rows. This repository reports the
observed release exactly and never substitutes the paper count.

Generated `top10_scores.jsonl` rows follow the preregistered long schema.
`per_query_metrics.jsonl`, `summary.json`, and `direction_decision.json` are
the authoritative statistical and direction artifacts.
