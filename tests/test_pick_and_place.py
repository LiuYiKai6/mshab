import gymnasium as gym

from mani_skill import ASSET_DIR
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from mshab.envs.planner import plan_data_from_file
from mshab.envs.pick_and_place import PickAndPlaceSubtaskTrainEnv

task = "tidy_house"  # "tidy_house", "prepare_groceries", or "set_table"
subtask = "pick_and_place"
split = "train"  # "train", "val"


REARRANGE_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"

plan_data = plan_data_from_file(
    REARRANGE_DIR / "task_plans" / task / subtask / split / "all.json"
)
spawn_data_fp = REARRANGE_DIR / "spawn_data" / task / subtask / split / "spawn_data.pt"

env = gym.make(
    "PickAndPlaceSubtaskTrain-v0",
    # Simulation args
    num_envs=1,  # RCAD has 63 train scenes, so 252 envs -> 4 parallel envs reserved for each scene
    obs_mode="rgbd",
    sim_backend="cpu",
    robot_uids="fetch",
    control_mode="pd_joint_delta_pos",
    # Rendering args
    reward_mode="normalized_dense",
    render_mode="human",
    shader_dir="rt-fast",
    # TimeLimit args
    max_episode_steps=600,
    # SequentialTask args
    task_plans=plan_data.plans,
    scene_builder_cls=plan_data.dataset,
    # SubtaskTrain args
    spawn_data_fp=spawn_data_fp,
    # optional: additional env_kwargs
    require_build_configs_repeated_equally_across_envs=False,
)

# add env wrappers here

env = ManiSkillVectorEnv(
    env,
    max_episode_steps=600,  # set manually based on task
    ignore_terminations=True,  # set to False for partial resets
)

uenv: PickAndPlaceSubtaskTrainEnv = env.unwrapped

for _ in range(10):
    obs, info = env.reset()

    print(uenv.subtask_objs[0]._objs[0].name)

    viewer = env.render()
    viewer.paused = True
    viewer = env.render()

    for _ in range(10):
        env.step(env.action_space.sample() * 0)
