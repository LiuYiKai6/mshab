from typing import Any, Dict, List

import torch

from mani_skill import ASSET_DIR
from mani_skill.envs.utils import randomization
from mani_skill.utils import common
from mani_skill.utils.geometry.rotation_conversions import quaternion_raw_multiply
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Articulation, Pose
from mani_skill.utils.io_utils import load_json

from mshab.envs.planner import (
    PickAndPlaceSubtask,
    PickAndPlaceSubtaskConfig,
    Subtask,
    TaskPlan
)
from mshab.envs.subtask import SubtaskTrainEnv
from mshab.envs.sequential_task import GOAL_POSE_Q


@register_env("PickAndPlaceSubtaskTrain-v0", max_episode_steps=600)
class PickAndPlaceSubtaskTrainEnv(SubtaskTrainEnv):
    """
    Task Description
    ----------------
    Add a task description here

    Randomizations
    --------------

    Success Conditions
    ------------------

    Visualization: link to a video/gif of the task being solved
    """

    pick_and_place_cfg = PickAndPlaceSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
        robot_cumulative_force_limit=12500,
        goal_type="sphere",
    )

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        **kwargs,
    ):

        tp0 = task_plans[0]
        assert len(tp0.subtasks) == 1 and isinstance(
            tp0.subtasks[0], PickAndPlaceSubtask
        ), f"Task plans for {self.__class__.__name__} must be one {PickAndPlaceSubtask.__name__} long"

        self.subtask_cfg = self.pick_and_place_cfg

        self.place_obj_ids = set()
        for tp in task_plans:
            self.place_obj_ids.add("-".join(tp.subtasks[0].obj_id.split("-")[:-1]))
            
        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # PROCESS TASKS
    # -------------------------------------------------------------------------------------------------

    def _merge_pick_and_place_subtasks(
        self, 
        env_idx: torch.Tensor,
        subtask_num: int,
        parallel_subtasks: List[PickAndPlaceSubtask]
    ):
        merged_obj_name = f"obj_{subtask_num}"
        self.subtask_objs.append(
            self._create_merged_actor_from_subtasks(
                parallel_subtasks, name=merged_obj_name
            )
        )
        self.subtask_goals.append(self.premade_goal_list[subtask_num])

        merged_goal_pos = common.to_tensor(
            [subtask.goal_pos for subtask in parallel_subtasks]
        )
        merged_goal_rectangle_corners = common.to_tensor(
            [subtask.goal_rectangle_corners for subtask in parallel_subtasks]
        )

        self.subtask_goals[-1].set_pose(
            Pose.create_from_pq(q=GOAL_POSE_Q, p=merged_goal_pos[env_idx])
        )
        
        # NOTE (arth): this is a bit tricky, since prepare_groceries sometimes has an articulation config
        #   (pick from fridge), but sometimes does not (pick from countertop). however, when running tasks
        #   sequentially, the prepare_groceries parallel_subtasks will either all have or all not have
        #   articulation_config. this is unlike tasks like tidy_house, which never has articulation_config
        #   or set_table which always has articulation_config.
        #   later, we expect that all subtask_x has len(_objs) == num_envs, so we can't merge for prepare_groceries
        #   as-is. for now, we ignore the prepare_groceries case (since fridge is opened by ao_config anyways)
        #   but in the future we'll need to support a case with e.g. a modified set_table with pick from table
        if all(
            [subtask.articulation_config is not None for subtask in parallel_subtasks]
        ):
            merged_articulation_name = f"articulation-{subtask_num}"
            merged_articulation = (
                self._create_merged_articulation_from_articulation_ids(
                    [
                        subtask.articulation_config.articulation_id
                        for subtask in parallel_subtasks
                    ],
                    name=merged_articulation_name,
                    merging_different_articulations=True,
                )
            )
            self.subtask_articulations.append(merged_articulation)
        else:
            self.subtask_articulations.append(None)

        self.task_plan.append(
            PickAndPlaceSubtask(
                obj_id=merged_obj_name,
                goal_pos=merged_goal_pos,
                goal_rectangle_corners=merged_goal_rectangle_corners,
                validate_goal_rectangle_corners=False,
                articulation_config=None,
            )
        )

    def process_task_plan(
        self,
        env_idx: torch.Tensor,
        sampled_subtask_lists: List[List[Subtask]],
    ):

        self.subtask_objs: List[Actor] = []
        self.subtask_goals: List[Actor] = []
        self.subtask_articulations: List[Articulation] = []
        self.check_progressive_success_subtask_nums: List[int] = []

        # build new merged task_plan and merge actors of parallel task plants
        self.task_plan: List[Subtask] = []
        for subtask_num, parallel_subtasks in enumerate(zip(*sampled_subtask_lists)):
            composite_subtask_uids = [subtask.uid for subtask in parallel_subtasks]
            subtask0: Subtask = parallel_subtasks[0]

            if isinstance(subtask0, PickAndPlaceSubtask):
                self._merge_pick_and_place_subtasks(env_idx, subtask_num, parallel_subtasks)
            else:
                raise AttributeError(
                    f"{subtask0.type} {type(subtask0)} not yet supported"
                )

            self.task_plan[-1].composite_subtask_uids = composite_subtask_uids

        assert len(self.subtask_objs) == len(self.task_plan)
        assert len(self.subtask_goals) == len(self.task_plan)
        assert len(self.subtask_articulations) == len(self.task_plan)

        self.task_horizons = torch.tensor(
            [self.task_cfgs[subtask.type].horizon for subtask in self.task_plan],
            device=self.device,
            dtype=torch.long,
        )
        self.task_ids = torch.tensor(
            [self.task_cfgs[subtask.type].task_id for subtask in self.task_plan],
            device=self.device,
            dtype=torch.long,
        )
    
    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # RESET/RECONFIGURE HANDLING
    # -------------------------------------------------------------------------------------------------
    
    def _load_scene(self, options):
        super()._load_scene(options)
        self.premade_goal_list: List[Actor] = []
        for subtask_num, subtask in enumerate(self.tp0.subtasks):
            if isinstance(subtask, PickAndPlaceSubtask):
                goal = self._make_goal(
                    radius=self.pick_and_place_cfg.obj_goal_thresh,
                    name=f"goal_{subtask_num}",
                    goal_type=(
                        "cylinder"
                        if self.pick_and_place_cfg.goal_type == "zone"
                        else self.pick_and_place_cfg.goal_type
                    ),
                )
            else:
                goal = None
            self.premade_goal_list.append(goal)
    
    # -------------------------------------------------------------------------------------------------
       
    # -------------------------------------------------------------------------------------------------
    # INIT ROBOT SPAWN RANDOMIZATION
    # -------------------------------------------------------------------------------------------------

    def _initialize_episode(self, env_idx, options):
        self.task_cfgs.update(pick_and_place=self.pick_and_place_cfg)
        with torch.device(self.device):
            super()._initialize_episode(env_idx, options)
            if self.target_randomization:
                b = len(env_idx)

                # randomization object pose
                obj_xyz = torch.zeros((b, 3))
                obj_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
                obj_xyz += self.subtask_objs[0].pose.p
                obj_xyz[..., 2] += 0.005

                obj_qs = quaternion_raw_multiply(
                    randomization.random_quaternions(
                        b, lock_x=True, lock_y=True, lock_z=False
                    ),
                    self.subtask_objs[0].pose.q,
                )
                self.subtask_objs[0].set_pose(Pose.create_from_pq(obj_xyz, obj_qs))
                
                # randomization goal pose
                goal_xyz = torch.zeros((b, 3))
                goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
                goal_xyz += self.subtask_goals[0].pose.p
                goal_xyz[..., 2] += 0.005

                goal_qs = quaternion_raw_multiply(
                    randomization.random_quaternions(
                        b, lock_x=True, lock_y=True, lock_z=False
                    ),
                    self.subtask_goals[0].pose.q,
                )
                self.subtask_goals[0].set_pose(Pose.create_from_pq(goal_xyz, goal_qs))

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # SUBTASK STATUS CHECKERS/UPDATERS
    # -------------------------------------------------------------------------------------------------
    
    def _subtask_check_success(self):
        subtask_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        success_checkers = dict()

        currently_running_subtasks = torch.unique(
            torch.clip(self.subtask_pointer, max=len(self.task_plan) - 1)
        )
        for subtask_num in currently_running_subtasks:
            subtask: Subtask = self.task_plan[subtask_num]
            env_idx = torch.where(self.subtask_pointer == subtask_num)[0]
            if isinstance(subtask, PickAndPlaceSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._pick_and_place_check_success(
                    self.subtask_objs[subtask_num],
                    self.subtask_goals[subtask_num],
                    subtask.goal_rectangle_corners,
                    env_idx,
                )
            else:
                raise NotImplementedError(
                    f"{subtask.type} {type(subtask)} not supported"
                )

            for k, v in subtask_success_checkers.items():
                if k not in success_checkers:
                    success_checkers[k] = torch.zeros(
                        self.num_envs, device=self.device, dtype=v.dtype
                    )
                success_checkers[k][env_idx] = v

        return subtask_success, success_checkers

    def _pick_and_place_check_success(
        self,
        obj: Actor,
        obj_goal: Actor,
        goal_rectangle_corners: torch.Tensor,
        env_idx: torch.Tensor,
        check_progressive_completion=False,
    ):
        is_grasped = self.agent.is_grasping(obj, max_angle=30)[env_idx]
        if self.pick_and_place_cfg.goal_type == "zone":
            # (0 <= AM•AB <= AB•AB) and (0 <= AM•AD <=  AD•AD)
            As, Bs, Ds = (
                goal_rectangle_corners[env_idx, 0, :2],
                goal_rectangle_corners[env_idx, 1, :2],
                goal_rectangle_corners[env_idx, 3, :2],
            )
            Ms = obj.pose.p[env_idx, :2]

            AM = Ms - As
            AB = Bs - As
            AD = Ds - As

            AM_dot_AB = torch.sum(AM * AB, dim=1)
            AB_dot_AB = torch.sum(AB * AB, dim=1)
            AM_dot_AD = torch.sum(AM * AD, dim=1)
            AD_dot_AD = torch.sum(AD * AD, dim=1)

            xy_correct = (
                (0 <= AM_dot_AB)
                & (AM_dot_AB <= AB_dot_AB)
                & (0 <= AM_dot_AD)
                & (AM_dot_AD <= AD_dot_AD)
            )
            z_correct = (
                torch.abs(obj.pose.p[env_idx, 2] - obj_goal.pose.p[env_idx, 2])
                <= self.pick_and_place_cfg.obj_goal_thresh
            )
            obj_at_goal = xy_correct & z_correct
        elif self.pick_and_place_cfg.goal_type == "cylinder":
            xy_correct = (
                torch.norm(
                    obj.pose.p[env_idx, :2] - obj_goal.pose.p[env_idx, :2],
                    dim=1,
                )
                <= self.pick_and_place_cfg.obj_goal_thresh
            )
            z_correct = (
                torch.abs(obj.pose.p[env_idx, 2] - obj_goal.pose.p[env_idx, 2])
                <= self.pick_and_place_cfg.obj_goal_thresh
            )
            obj_at_goal = xy_correct & z_correct
        elif self.pick_and_place_cfg.goal_type == "sphere":
            obj_at_goal = (
                torch.norm(
                    obj.pose.p[env_idx] - obj_goal.pose.p[env_idx],
                    dim=1,
                )
                <= self.pick_and_place_cfg.obj_goal_thresh
            )
        else:
            raise NotImplementedError(
                f"{self.pick_and_place_cfg.goal_type=} is not yet supported"
            )
        if check_progressive_completion:
            return obj_at_goal, dict(obj_at_goal=obj_at_goal)
        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= self.pick_and_place_cfg.ee_rest_thresh
        )
        robot_rest_dist = torch.abs(
            self.agent.robot.qpos[env_idx, 4:-2] - self.resting_qpos[1:]
        )
        robot_rest = torch.all(
            robot_rest_dist < self.pick_and_place_cfg.robot_resting_qpos_tolerance, dim=1
        ) & (torch.abs(self.agent.robot.qpos[env_idx, 3] - self.resting_qpos[0]) < 0.01)
        is_static = self.agent.is_static(threshold=0.2, base_threshold=0.05)[env_idx]
        cumulative_force_within_limit = (
            self.robot_cumulative_force[env_idx]
            < self.pick_and_place_cfg.robot_cumulative_force_limit
        )
        subtask_checkers = dict(
            is_grasped=is_grasped,
            obj_at_goal=obj_at_goal,
            ee_rest=ee_rest,
            robot_rest=robot_rest,
            is_static=is_static,
            cumulative_force_within_limit=cumulative_force_within_limit,
        )
        if self._add_event_tracker_info:
            subtask_checkers["robot_target_pairwise_force"] = torch.norm(
                self.scene.get_pairwise_contact_forces(self.agent.finger1_link, obj)[
                    env_idx
                ],
                dim=1,
            ) + torch.norm(
                self.scene.get_pairwise_contact_forces(self.agent.finger2_link, obj)[
                    env_idx
                ],
                dim=1,
            )
        return (
            ~is_grasped
            & obj_at_goal
            & ee_rest
            & robot_rest
            & is_static
            & cumulative_force_within_limit,
            subtask_checkers,
        )
        
    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # OBS AND INFO
    # -------------------------------------------------------------------------------------------------
    
    def _get_obs_extra(self, info: Dict):
        obs = super()._get_obs_extra(info)
        obj_pose_wrt_base = obs["obj_pose_wrt_base"]
        obs.update(obj_pos_wrt_base=obj_pose_wrt_base[:, :3])
        del obs["obj_pose_wrt_base"]
        return obs
    
    # -------------------------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs)

            obj_pos = self.subtask_objs[0].pose.p
            goal_pos = self.subtask_goals[0].pose.p
            rest_pos = self.ee_rest_world_pose.p
            tcp_pos = self.agent.tcp_pose.p

            # NOTE (yikailiu): reward "steps" are as follows:
            #       - if not grasped and not at goal
            #           - not_grasped_reward
            #       - grasped and not at goal
            #           - obj to goal reward
            #       - if at goal
            #           - rest reward
            #       - if at goal and at rest
            #           - static reward
            #       - success_reward
            # ---------------------------------------------------
            # CONDITION CHECKERS
            # ---------------------------------------------------

            obj_to_goal_dist = torch.norm(obj_pos - goal_pos, dim=1)
            tcp_to_goal_dist = torch.norm(tcp_pos - goal_pos, dim=1)
            
            not_grasped_and_not_at_goal = ~info["is_grasped"] & ~info["obj_at_goal"]
            not_grasped_and_not_at_goal_reward = torch.zeros_like(reward[not_grasped_and_not_at_goal])
            
            grasped_and_not_at_goal = info["is_grasped"] & ~info["obj_at_goal"]
            grasped_and_not_at_goal_reward = torch.zeros_like(reward[grasped_and_not_at_goal])
            
            obj_at_goal_maybe_dropped = info["obj_at_goal"]
            obj_at_goal_maybe_dropped_reward = torch.zeros_like(reward[obj_at_goal_maybe_dropped])
            
            ee_to_rest_dist = torch.norm(tcp_pos - rest_pos, dim=1)
            robot_ee_rest_and_at_goal = info["obj_at_goal"] & info["ee_rest"] & info["robot_rest"]
            robot_ee_rest_and_at_goal_reward = torch.zeros_like(reward[robot_ee_rest_and_at_goal])

            # ---------------------------------------------------

            # penalty for ee moving too much
            ee_vel = self.agent.tcp.linear_velocity
            ee_still_rew = 1 - torch.tanh(torch.norm(ee_vel, dim=1) / 5)
            reward += ee_still_rew
            
            # penalty for object moving too much when not grasped
            obj_vel = torch.norm(
                self.subtask_objs[0].linear_velocity, dim=1
            ) + torch.norm(self.subtask_objs[0].angular_velocity, dim=1)
            obj_vel[info["is_grasped"]] = 0
            obj_still_rew = 3 * (1 - torch.tanh(obj_vel / 5))
            reward += obj_still_rew
            
            # encourage arm and torso in "resting" orientation
            arm_to_resting_diff = torch.norm(
                self.agent.robot.qpos[..., 3:-2] - self.resting_qpos,
                dim=1,
            )
            arm_resting_orientation_rew = 1 - torch.tanh(arm_to_resting_diff / 5)
            reward += arm_resting_orientation_rew
            
            # success reward
            success_rew = 8 * info["success"]
            reward += success_rew
            
            # ---------------------------------------------------------------
            
            # colliisions
            step_no_col_rew = 3 * (
                1
                - torch.tanh(
                    3
                    * (
                        torch.clamp(
                            self.robot_force_mult * info["robot_force"],
                            min=self.robot_force_penalty_min,
                        )
                        - self.robot_force_penalty_min
                    )
                )
            )
            reward += step_no_col_rew

            # cumulative collision penalty
            cum_col_under_thresh_rew = (
                2
                * (
                    info["robot_cumulative_force"]
                    < self.pick_and_place_cfg.robot_cumulative_force_limit
                ).float()
            )
            reward += cum_col_under_thresh_rew
            
            # ---------------------------------------------------------------
            
            if torch.any(not_grasped_and_not_at_goal):
                # reaching reward
                tcp_to_obj_dist = torch.norm(
                    obj_pos[not_grasped_and_not_at_goal] - tcp_pos[not_grasped_and_not_at_goal],
                    dim=1
                )
                reaching_rew = 5 * (1 - torch.tanh(3 * tcp_to_obj_dist))
                not_grasped_and_not_at_goal_reward += reaching_rew
                
                # penalty for torso moving up and down too much
                tqvel_z = self.agent.robot.qvel[not_grasped_and_not_at_goal, 3]
                torso_not_moving_rew = 1 - torch.tanh(5 * torch.abs(tqvel_z))
                torso_not_moving_rew[tcp_to_obj_dist < 0.3] = 1
                not_grasped_and_not_at_goal_reward += torso_not_moving_rew

                # penalty for ee not over obj
                ee_over_obj_rew = 1 - torch.tanh(
                    5
                    * torch.norm(
                        obj_pos[not_grasped_and_not_at_goal, :2] - tcp_pos[not_grasped_and_not_at_goal, :2],
                        dim=1,
                    )
                )
                not_grasped_and_not_at_goal_reward += ee_over_obj_rew
            
            if torch.any(grasped_and_not_at_goal):
                # not_grasped reward has max of +10
                # so, we add +10 to grasped reward so reward only increases as task proceeds
                grasped_and_not_at_goal_reward += 7
                
                # ee holding object
                grasped_and_not_at_goal_reward += 2
                
                # arm_to_resting_diff_again
                grasped_and_not_at_goal_reward += arm_resting_orientation_rew[grasped_and_not_at_goal]
                
                # penalty for torso moving down too much
                tqvel_z = torch.clip(self.agent.robot.qvel[grasped_and_not_at_goal, 3], max=0)
                torso_not_moving_rew = 1 - torch.tanh(5 * torch.abs(tqvel_z))
                grasped_and_not_at_goal_reward += torso_not_moving_rew

                # obj and tcp close to goal
                place_rew = 6 * (
                    1
                    - (
                        (
                            torch.tanh(obj_to_goal_dist[grasped_and_not_at_goal])
                            + torch.tanh(tcp_to_goal_dist[grasped_and_not_at_goal])
                        )
                        / 2
                    )
                )
                grasped_and_not_at_goal_reward += place_rew

                # obj and tcp right above goal pos
                correct_height_rew = 4 * (
                    1
                    - torch.tanh(
                        (
                            torch.abs(
                                obj_pos[grasped_and_not_at_goal, 2]
                                - (goal_pos[grasped_and_not_at_goal, 2] + 0.05)
                            )
                            + torch.abs(
                                tcp_pos[grasped_and_not_at_goal, 2]
                                - (goal_pos[grasped_and_not_at_goal, 2] + 0.05)
                            )
                        )
                        / 2
                    )
                )
                grasped_and_not_at_goal_reward += correct_height_rew
                
            if torch.any(obj_at_goal_maybe_dropped):
                # add prev step max rew
                obj_at_goal_maybe_dropped_reward += 21

                # rest reward
                rest_rew = 5 * (
                    1 - torch.tanh(3 * ee_to_rest_dist[obj_at_goal_maybe_dropped])
                )
                obj_at_goal_maybe_dropped_reward += rest_rew

                # additional encourage arm and torso in "resting" orientation
                arm_resting_orientation_rew = 4 * (1 - torch.tanh(arm_to_resting_diff[obj_at_goal_maybe_dropped]))
                obj_at_goal_maybe_dropped_reward += arm_resting_orientation_rew

                # additional torso orientation reward
                torso_resting_orientation_reward = 2 * torch.abs(
                    (
                        self.agent.robot.qpos[obj_at_goal_maybe_dropped, 3]
                        - self.agent.robot.qlimits[obj_at_goal_maybe_dropped, 3, 0]
                    )
                    / (
                        self.agent.robot.qlimits[obj_at_goal_maybe_dropped, 3, 1]
                        - self.agent.robot.qlimits[obj_at_goal_maybe_dropped, 3, 0]
                    )
                )
                obj_at_goal_maybe_dropped_reward += torso_resting_orientation_reward

                # penalty for base moving or rotating too much
                bqvel = self.agent.robot.qvel[obj_at_goal_maybe_dropped, :3]
                base_still_rew = 1 - torch.tanh(torch.norm(bqvel, dim=1))
                obj_at_goal_maybe_dropped_reward += base_still_rew

            if torch.any(robot_ee_rest_and_at_goal):
                robot_ee_rest_and_at_goal_reward += 2

                qvel = self.agent.robot.qvel[robot_ee_rest_and_at_goal, :-2]
                static_rew = 1 - torch.tanh(torch.norm(qvel, dim=1))
                robot_ee_rest_and_at_goal_reward += static_rew

                # penalty for base moving or rotating too much
                bqvel = self.agent.robot.qvel[robot_ee_rest_and_at_goal, :3]
                base_still_rew = 1 - torch.tanh(torch.norm(bqvel, dim=1))
                robot_ee_rest_and_at_goal_reward += base_still_rew
            
            # add rewards to specific envs
            reward[not_grasped_and_not_at_goal] += not_grasped_and_not_at_goal_reward
            reward[grasped_and_not_at_goal] += grasped_and_not_at_goal_reward
            reward[obj_at_goal_maybe_dropped] += obj_at_goal_maybe_dropped_reward
            reward[robot_ee_rest_and_at_goal] += robot_ee_rest_and_at_goal_reward

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 55.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # -------------------------------------------------------------------------------------------------
