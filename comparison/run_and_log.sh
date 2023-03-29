TARGET_DIR='/home/martin.ingram/experiment_runs/march_2023/run_log_others'

mkdir -p "$TARGET_DIR"

bash run_all.sh 2>&1 | tee -a "$TARGET_DIR"/raabbvi_tennis_run.log
# bash run_all_arm.sh 2>&1 | tee -a "$TARGET_DIR"/run_arm.log

# bash run_all.sh 2>&1 | tee -a "$TARGET_DIR"/raabbvi_rerun_post_restart.log
