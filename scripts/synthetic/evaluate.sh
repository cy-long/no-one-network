#!/bin/bash
set -e
source synthetic/settings.sh


export PYTHONPATH=${CLV_DIR}
bash synthetic/helpers/evaluate.py \
-g ${DATASET_DIR}/glv.npz \
-m 3723.06368786 \
-s 6567.91725508
