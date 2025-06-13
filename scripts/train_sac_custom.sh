#!/usr/bin/bash
timestamp=`date "+%Y%m%d_%H%M%S"`

SEED=0

TASK=tidy_house
SUBTASK=pick_and_place
SPLIT=train
OBJ=002_master_chef_can

ENV_ID="PickAndPlaceSubtaskTrain-v0"
EXP_NAME="$ENV_ID/$TASK/sac-$OBJ-$timestamp"

# NOTE (arth): tensorboard=False since there seems to be an issue with tensorboardX crashing on very long runs
if [[ -z "${MS_ASSET_DIR}" ]]; then
    MS_ASSET_DIR="$HOME/.maniskill"
fi


SAPIEN_NO_DISPLAY=1 python -m mshab.train_sac configs/sac_pick_and_place.yml \
        seed=$SEED \
        env.env_id="$ENV_ID" \
        env.task_plan_fp="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/$SUBTASK/$SPLIT/$OBJ.json" \
        env.spawn_data_fp="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt" \
        eval_env.task_plan_fp="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/$SUBTASK/$SPLIT/$OBJ.json" \
        eval_env.spawn_data_fp="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt" \
        eval_env.record_video="False" \
        eval_env.info_on_video="False" \
        logger.exp_name="$EXP_NAME" \
