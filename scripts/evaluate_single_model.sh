#!/usr/bin/bash

#############################################
# Changed
########################################

SEED=0

ckpt_name="sac-002_master_chef_can-20250613_132311"
ckpt_root="mshab_exps/PickAndPlaceSubtaskTrain-v0/tidy_house"
ckpt_dir="$ckpt_root/$ckpt_name"

TASK=tidy_house
SUBTASK=pick_and_place # Dependent on ckpt
SPLIT=train
OBJ=002_master_chef_can

record_video=True
info_on_video=False
save_trajectory=False
max_trajectories=8
policy_type=rl_all_obj # Unchanged

continuous_task=True

#############################################

#############################################
# Unchanged
########################################
WORKSPACE="mshab_exps"
GROUP=eval_single_model-rcad-$TASK-$SUBTASK
EXP_NAME="eval_single_model/$TASK/$SUBTASK/$SPLIT/$OBJ/$ckpt_name"
# shellcheck disable=SC2001
PROJECT_NAME="MS-HAB-RCAD-$(echo $SUBTASK | sed 's/\b\(.\)/\u\1/g')-$TASK-$ckpt_name"
if [[ -z "${MS_ASSET_DIR}" ]]; then
    MS_ASSET_DIR="$HOME/.maniskill"
fi


WANDB=False
TENSORBOARD=True

#############################################

#############################################
# Dependent on Above
########################################

if [[ $SUBTASK == "sequential" ]]; then

        # env id
        ENV_ID="SequentialTask-v0"

        # num envs
        if [[ $SPLIT == "train" ]]; then
                NUM_ENVS=63
        else
                NUM_ENVS=21
        fi

        # horizon
        # NOTE (arth): we ignore steps needed for the navigate task since we teleport
        # NOTE (arth): while we set max subtask steps to 200, really we don't need that many for overall horizon
        if [[ $TASK == "tidy_house" ]]; then
                MAX_EPISODE_STEPS=1000
        elif [[ $TASK == "prepare_groceries" ]]; then
                MAX_EPISODE_STEPS=600
        else
                MAX_EPISODE_STEPS=800
        fi

        # extra args
        extra_args=()
else
        if [[ $SUBTASK == "pick_and_place" ]]; then
                # env id
                ENV_ID="PickAndPlaceSubtaskTrain-v0"
                
                # horizon
                MAX_EPISODE_STEPS=600

        else
                # env id
                # shellcheck disable=SC2001
                ENV_ID="$(echo $SUBTASK | sed 's/\b\(.\)/\u\1/g')SubtaskTrain-v0"

                # horizon
                MAX_EPISODE_STEPS=200
        fi

        # num envs
        if [[ $record_video == "True" ]]; then
                if [[ $SPLIT == "train" ]]; then
                        NUM_ENVS=63
                else
                        NUM_ENVS=21
                fi
        else
                NUM_ENVS=252
        fi

        # extra args
        # shellcheck disable=SC2089
        spawn_data_fp="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt"
        if [[ $SUBTASK == "pick" ]]; then
                extra_stat_keys='<list>success, subtask_type, is_grasped, robot_target_pairwise_force, robot_force, robot_cumulative_force</list>'
        elif [[ $SUBTASK == "place" ]]; then
                extra_stat_keys='<list>success, subtask_type, is_grasped, obj_at_goal, robot_force, robot_cumulative_force</list>'
        elif [[ $SUBTASK == "open" ]]; then
                extra_stat_keys='<list>success, subtask_type, articulation_open, robot_target_pairwise_force, robot_force, robot_cumulative_force, handle_active_joint_qpos, handle_active_joint_qmax, handle_active_joint_qmin</list>'
        elif [[ $SUBTASK == "close" ]]; then
                extra_stat_keys='<list>success, subtask_type, articulation_closed, robot_target_pairwise_force, robot_force, robot_cumulative_force, handle_active_joint_qpos, handle_active_joint_qmax, handle_active_joint_qmin</list>'
        elif [[ $SUBTASK == "place" ]]; then
                extra_stat_keys='<list>success, subtask_type, is_grasped, ever_grasped, obj_at_goal, robot_force, robot_cumulative_force</list>'
        fi
        extra_args+=(
                "eval_env.spawn_data_fp=\"$spawn_data_fp\""
                "eval_env.extra_stat_keys=\"$extra_stat_keys\""
        )
fi


if [[ $policy_type == "bc" ]]; then
        FRAME_STACK=1
        STACK=null
elif [[ $policy_type == "dp" ]]; then
        FRAME_STACK=null
        STACK=2
else
        FRAME_STACK=3
        STACK=null
fi

#############################################

NUM_ENVS=1

SAPIEN_NO_DISPLAY=1 python -m mshab.evaluate_single_model configs/evaluate.yml \
        seed=$SEED \
        task=$TASK \
        save_trajectory=$save_trajectory \
        max_trajectories=$max_trajectories \
        policy_type=$policy_type \
        ckpt_dir=$ckpt_dir \
        eval_env.env_id="$ENV_ID" \
        eval_env.task_plan_fp="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/$SUBTASK/$SPLIT/$OBJ.json" \
        \
        eval_env.make_env="True" \
        \
        eval_env.num_envs=$NUM_ENVS \
        eval_env.frame_stack=$FRAME_STACK \
        eval_env.stack=$STACK \
        \
        eval_env.max_episode_steps=$MAX_EPISODE_STEPS \
        eval_env.continuous_task=$continuous_task \
        \
        eval_env.record_video="$record_video" \
        eval_env.info_on_video="$info_on_video" \
        eval_env.save_video_freq=1 \
        \
        logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
        logger.wandb="$WANDB" \
        logger.tensorboard="$TENSORBOARD" \
        logger.project_name="$PROJECT_NAME" \
        logger.workspace="$WORKSPACE" \
        logger.clear_out="False" \
        logger.wandb_cfg.group="$GROUP" \
        logger.exp_name="$EXP_NAME" \
        "${extra_args[@]}" 
        # NOTE (arth): one can set easier/harder task conditions as below
        # eval_env.env_kwargs.task_cfgs="{pick: {robot_cumulative_force_limit: 100000000}, place: {goal_type: zone, robot_cumulative_force_limit: 100000000}, open: {robot_cumulative_force_limit: 100000000}, close: {robot_cumulative_force_limit: 100000000}}"
