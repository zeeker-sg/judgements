#!/bin/bash
cd /workspace/extra/zeeker-judgements
export PATH="/workspace/extra/zeeker-judgements/.venv-py311/bin:$PATH"
export JUDGMENTS_SUMMARY_MAX_PER_RUN=3
export JUDGMENTS_EXTRACT_MAX_PER_RUN=0
python -m zeeker build judgments > build_test5.log 2>&1
