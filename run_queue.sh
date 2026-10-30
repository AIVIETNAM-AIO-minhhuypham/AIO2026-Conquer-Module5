#!/usr/bin/env bash
set -e                      # stop at the first failing step
cd "$(dirname "$0")"
source venv/Scripts/activate

python main.py sanity
python main.py overfit --overfit-steps 1500
python main.py bench
python main.py queue --plan lr_sweep
python main.py queue --plan lr_sweep_ext
python main.py select-lr
python main.py queue --plan pilot10
python main.py summarize

echo "All steps finished."