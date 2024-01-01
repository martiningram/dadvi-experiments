TARGET_DIR='/home/martin.ingram/experiment_runs/december_2023/run_log'

mkdir -p "$TARGET_DIR"

bash run_all.sh 2>&1 | tee -a "$TARGET_DIR"/tennis_run.log
# bash run_all_arm.sh 2>&1 | tee -a "$TARGET_DIR"/run_arm.log
