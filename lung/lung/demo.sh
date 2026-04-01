#!/usr/bin/env bash
python predict.py \
  --wsi /data4/????35?/1??/20-02253_svs/20-02253-132023-06-30_09_34_19.svs \
  --weights /home/deeplearning/data/data2/wyj/test/weight/LUNG_NEW_final.h5 \
  --output /home/deeplearning/data/data2/wyj/test/output_predict_demo \
  --batch_size 32
