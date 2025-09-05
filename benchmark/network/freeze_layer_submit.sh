#!/bin/bash
# submit_all_nohup.sh — 使用 nohup 在本节点后台批量运行 fine_tune_training.py

# 可根据需要调整环境激活命令
module load anaconda

# 确保日志目录存在
mkdir -p freeze_layer_test

for layer in 1; do
  for seed in 1; do

    LOG="freeze_layer_test/freeze_layer_${layer}_${seed}.log"
    echo "Launching layer=$layer seed=$seed → log: $LOG"

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

    # （可选）在循环中稍微 sleep，避免瞬间启动过多进程
    sleep 0.2

  done
done

echo "All jobs launched with nohup."
