# DoL mn-pretrain runtime for A800/A100 (sm_80).
#
# Environment-only image: the repo and data are volume-mounted at run time,
# so code changes never require a rebuild. devel base keeps nvcc available
# for the mamba-ssm build fallback (prebuilt wheels don't cover every
# torch/cuda pairing).
#
# Build:  docker build -t dol-pretrain:cu124 docker/
# Run:    see docker/a800.md

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# tokenizers: Tokenizer package runtime; einops/regex: model deps;
# causal-conv1d before mamba-ssm (its build probes torch+cuda).
RUN pip install tokenizers einops regex \
 && (pip install causal-conv1d || CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install causal-conv1d --no-build-isolation) \
 && (pip install mamba-ssm || MAMBA_FORCE_BUILD=TRUE pip install mamba-ssm --no-build-isolation)

WORKDIR /work
ENV PYTHONPATH=/work/repo
# repo   -> /work/repo    (DoL-OCR checkout)
# data   -> /work/data    (pretrain shards + eval split)
# runs   -> /work/runs    (checkpoints, persisted on host)
CMD ["/bin/bash"]
