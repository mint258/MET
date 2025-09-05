#!/bin/bash
# submit_all_nohup.sh �� ʹ�� nohup �ڱ��ڵ��̨�������� fine_tune_training.py

# �ɸ�����Ҫ����������������
module load anaconda

# ȷ����־Ŀ¼����
mkdir -p freeze_layer_test

for layer in 1; do
  for seed in 1; do

    LOG="freeze_layer_test/freeze_layer_${layer}_${seed}.log"
    echo "Launching layer=$layer seed=$seed �� log: $LOG"

    nohup \
      python fine_tune_training.py \
        --pretrained_checkpoint_path best_model_dim128.pth \
        --data_root ../data/qm7_standard/ \
        --target_property rot_A \
        --batch_size 128 \
        --device cuda \
        --epochs 200 \
        --dropout 0 \
        --learning_rate 1e-4 \
        --freeze_up_to_layer ${layer} \
        --seed ${seed} \
      2>/dev/null \
      > "$LOG" &

    # ����ѡ����ѭ������΢ sleep������˲�������������
    sleep 0.2

  done
done

echo "All jobs launched with nohup."
