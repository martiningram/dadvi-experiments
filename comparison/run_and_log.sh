TARGET_DIR='/media/martin/External Drive/projects/lrvb_paper/dadvi_runs_february_2023_2/log'

mkdir -p "$TARGET_DIR"

# bash run_all_arm.sh 2>&1 | tee -a "$TARGET_DIR"/run_arm.log
# bash run_all.sh 2>&1 | tee -a "$TARGET_DIR"/run_others.log
bash run_all.sh 2>&1 | tee -a "$TARGET_DIR"/raabbvi_rerun_post_restart.log
