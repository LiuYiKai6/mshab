import os
import json
from typing import List, Dict
from dataclasses import asdict

from mani_skill import ASSET_DIR

from mshab.envs.planner import (
    TaskPlan,
    PlanData,
    PickSubtask,
    PlaceSubtask,
    PickAndPlaceSubtask,
    plan_data_from_file,
)

obj_names = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "024_bowl",
    "all",
]

task = "tidy_house"  # "tidy_house", "prepare_groceries", or "set_table"
split = "train"  # "train", "val"

REARRANGE_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"

plan_data = plan_data_from_file(
    REARRANGE_DIR / "task_plans" / task / "sequential" / split / "all.json"
)

new_plans_dict: Dict[str, List[TaskPlan]] = dict(
    (obj_name, [])
    for obj_name in obj_names
)
for plan in plan_data.plans:
    subtasks = plan.subtasks
    build_config_name = plan.build_config_name
    init_config_name = plan.init_config_name
    for i in range(int(len(subtasks)/4)):
        pick_subtask: PickSubtask = subtasks[i*4+1]
        place_subtask: PlaceSubtask = subtasks[i*4+3]
        pick_and_place_subtask = PickAndPlaceSubtask(
            obj_id=pick_subtask.obj_id,
            goal_rectangle_corners=place_subtask.goal_rectangle_corners,
            goal_pos=place_subtask.goal_pos,
            validate_goal_rectangle_corners=place_subtask.validate_goal_rectangle_corners,
            articulation_config=pick_subtask.articulation_config,
        )
        new_plan = TaskPlan(subtasks=[pick_and_place_subtask], build_config_name=build_config_name, init_config_name=init_config_name)
        new_plans_dict["all"].append(new_plan)
        for obj_name in obj_names:
            if obj_name in pick_and_place_subtask.obj_id:
                new_plans_dict[obj_name].append(new_plan)
                break
for obj_name in obj_names:
    new_plan_data = PlanData(dataset=plan_data.dataset, plans=new_plans_dict[obj_name])
    out_fp = REARRANGE_DIR / "task_plans" / task / "pick_and_place" / split / f"{obj_name}.json"
    os.makedirs(out_fp.parent, exist_ok=True)
    with open(out_fp, "w+") as f:
        json.dump(
            asdict(
                new_plan_data,
            ),
            f,
        )