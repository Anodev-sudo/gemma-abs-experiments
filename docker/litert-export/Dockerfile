FROM --platform=linux/amd64 python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 \
    && python -m pip install litert-torch

WORKDIR /workspace
