import torch
import mplib
import sapien
import argparse
import numpy as np
import gymnasium as gym
import open3d as o3d
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import OrderedDict
from scipy.spatial.transform import Rotation

from localgrasp.LoG import lg_parse, LgNet
from localgrasp.dataset.grasp import GraspGroup as OurGraspGroup
from graspnetAPI.grasp import GraspGroup as GraspNetGraspGroup

from mani_skill import ASSET_DIR, format_path
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.geometry import (
    angle_distance_simple,
    homo_transfer,
    uvz2xyz,
)

from mshab.envs.planner import plan_data_from_file

def angle_distance_simple(q0: np.ndarray, q1: np.ndarray):
    """
    Args:
        q0: (4,) or (N, 4)
        q1: (4,) or (N, 4)
    Returns:
        (1) or (N)
    """
    if len(q0.shape) == 2 and len(q1.shape) == 2:
        assert q0.shape[1] == 4 and q1.shape[1] == 4, "q0, q1 mush be (N, 4)"
        return 1 - np.clip(np.abs(np.einsum('ij, ij -> i', q0, q1)), a_min=0, a_max=1)
    elif len(q0.shape) == 2 and q1.size == 4:
        assert q0.shape[1] == 4 and q1.size == 4, "q0, q1 mush be (N, 4)"
        return 1 - np.clip(np.abs(np.einsum('ij, j -> i', q0, q1)), a_min=0, a_max=1)
    elif q0.size == 4 and q1.size == 4:
        return 1 - np.clip(np.abs(q0 @ q1), a_min=0, a_max=1)

def homo_transfer(R: np.ndarray, T: np.ndarray):
    """
    R, T shape: [N, 3, 3], [N, 3]
    or R, T shape: [3, 3], [3]
    """
    if len(R.shape) == 3:
        assert R.shape[0] == T.shape[0] and R.shape[1:] == (3, 3) and T.shape[1:] == (3,)
        H = np.zeros((R.shape[0], 4, 4))
        H[:, :3, :3] = R
        H[:, :3, 3] = T
        H[:, 3, 3] = 1
    elif len(R.shape) == 2:
        assert R.shape == (3, 3) and T.shape == (3,)
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = T
    return H

def uvz2xyz(uvz, intrinsic):
    intrinsic_inv = torch.linalg.inv(intrinsic)
    zuzvz = deepcopy(uvz)
    zuzvz[:, 0] *= zuzvz[:, 2]
    zuzvz[:, 1] *= zuzvz[:, 2]
    xyz = torch.matmul(intrinsic_inv, zuzvz.T).T
    return xyz

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-graspcenter", type=int, default=32)
    parser.add_argument("--vis-gg-offset", type=float, default=-0.02)
    parser.add_argument("--prev-grasp-offset", type=float, default=0.1)
    parser.add_argument("--grasp-offset", type=float, default=0)
    parser.add_argument("--max-plan-nums", type=int, default=5)
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output.", default=True)
    parser.add_argument("--visualize", action="store_true", help="visualize results.", default=False)

    parser = lg_parse(parser)
    args, opts = parser.parse_known_args()
    args.vis = args.visualize
    args.checkpoint_path = "/home/robolab/projects/mshab/localgrasp/localgrasp/epoch_11_acc_0.915_cover_0.765"
    grasp_detector = LgNet(args)
    
    # Parse env kwargs
    if not args.quiet:
        print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    if not args.quiet:
        print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args, grasp_detector

def get_imgs_pcd(handcam, vis=False):
    handcam.capture()
    images = handcam.get_obs()
    print(images.keys())
    position = images["position"] * 0.001 # [H, W, 3]
    imgs = OrderedDict()
    imgs["rgb"], imgs["depth"], imgs["actor_seg"] = (
        images["rgb"][..., :3],
        -position[..., [2]],
        images["segmentation"],
    )
    if vis:
        print("hand rgb, depth, actor_seg")
        plt.subplot(1, 3, 1)
        plt.imshow(imgs["rgb"].squeeze())
        plt.subplot(1, 3, 2)
        plt.imshow(imgs["depth"].squeeze())
        plt.subplot(1, 3, 3)
        plt.imshow(imgs["actor_seg"].squeeze())
        plt.show()

    pcd = OrderedDict()
    # Remove invalid points
    pcd["xyz"] = position[..., :3].reshape((position.shape[0], -1, 3))
    pcd["rgb"] = imgs["rgb"].reshape((position.shape[0], -1, 3))/255.0
    pcd["actor_seg"] = imgs["actor_seg"].reshape((position.shape[0], -1, 1))
    if vis:
        print("hand points (camera frame)")
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pcd["xyz"].squeeze())
        pc.colors = o3d.utility.Vector3dVector(pcd["rgb"].squeeze())
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        o3d.visualization.draw_geometries([pc, frame])
    return imgs, pcd

def inv_clip_and_scale_action(action, low, high):
    """Inverse of `clip_and_scale_action`."""
    low, high = np.asarray(low), np.asarray(high)
    action = (action - 0.5 * (high + low)) / (0.5 * (high - low))
    return np.clip(action, -1.0, 1.0)

def convert_traj_to_action(env, traj_qpos, gripper_pos) -> np.ndarray:
    if env.control_mode == "pd_joint_pos":
        return np.hstack([traj_qpos, gripper_pos])
    elif env.control_mode == "pd_joint_delta_pos":
        cur_joint_pos = np.array(env.agent.robot.get_qpos()[0])[:-2]
        delta_action = traj_qpos - cur_joint_pos
        return np.concatenate(
            (
                inv_clip_and_scale_action(
                    delta_action, env.agent.controller.configs["arm"].lower, env.agent.controller.configs["arm"].upper
                ),
                np.ones(1) * gripper_pos,
            )
        )
    else:
        raise NotImplementedError
    
def get_grasps(grasp_detector, paras):
    pred_gg = grasp_detector.infer_from_centers(
            scene_points=paras["scene_pc_ee"].cuda(),
            centers=paras["centers_ee"].cuda(),
        )
    
    if isinstance(pred_gg, GraspNetGraspGroup):
        pred_gg = OurGraspGroup(
            translations=pred_gg.translations,
            rotations=pred_gg.rotation_matrices,
            heights=pred_gg.heights,
            widths=pred_gg.widths,
            depths=pred_gg.depths,
            scores=pred_gg.scores,
            object_ids=pred_gg.object_ids,
        )
    return pred_gg

def transform_points(H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """transforms a batch of pts by a batch of transformation matrices H"""
    assert H.shape[1:] == (4, 4), H.shape
    assert pts.ndim == 3 and pts.shape[2] == 3, pts.shape
    return (
        torch.bmm(pts, H[:, :3, :3].transpose(2, 1)) + H[:, None, :3, 3].repeat(1, pts.shape[1], 1)
    )

def trans_graspgroup(gg, trans):
    gg_homo = homo_transfer(R=gg.rotations, T=gg.translations)  # (N, 4, 4)
    gg_trans_homo = np.array(trans @ gg_homo)  # (N, 4, 4)
    gg_trans = OurGraspGroup(
        translations=gg_trans_homo[:, :3, 3],
        rotations=gg_trans_homo[:, :3, :3],
        heights=gg.heights,
        widths=gg.widths,
        depths=gg.depths,
        scores=gg.scores,
        object_ids=gg.object_ids,
    )
    return gg_trans


task = "tidy_house"  # "tidy_house", "prepare_groceries", or "set_table"
subtask = "pick"  # "sequential", "pick", "place", "open", "close"
# NOTE: sequential loads the full task, e.g pick -> place -> ...
#     while pick, place, etc only simulate a single subtask each episode
split = "train"  # "train", "val"


REARRANGE_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"

plan_data = plan_data_from_file(
    REARRANGE_DIR / "task_plans" / task / subtask / split / "all.json"
)
spawn_data_fp = REARRANGE_DIR / "spawn_data" / task / subtask / split / "spawn_data.pt"

env = gym.make(
    f"{subtask.capitalize()}SubtaskTrain-v0",
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
    max_episode_steps=10000,
    # SequentialTask args
    task_plans=plan_data.plans[15:16],
    scene_builder_cls=plan_data.dataset,
    # SubtaskTrain args
    spawn_data_fp=spawn_data_fp,
    # optional: additional env_kwargs
    require_build_configs_repeated_equally_across_envs=False,
)

# add env wrappers here

venv = ManiSkillVectorEnv(
    env,
    max_episode_steps=1000,  # set manually based on task
    ignore_terminations=True,  # set to False for partial resets
)

# add vector env wrappers here

obs, info = venv.reset()

uenv = venv.unwrapped
uenv.agent.robot.set_pose(sapien.Pose([0.16, -1.6, 0], [1, 0, 0, 1]))
uenv.agent.robot.set_qpos(np.array([0, -1.032, 0, 0.955, -0.1, 1.57, 0, 0.015, 0.015]))
obj_id = "env-0_010_potted_meat_can-3"
seg_id = uenv.scene.actors[obj_id].per_scene_id

venv.render()

args, grasp_detector = parse_args()

print("=== SETUP PLANNER ===")
print("robot_root_pose: ", uenv.agent.robot.get_root_pose())
trans_world2robot = uenv.agent.robot.get_root_pose().inv().to_transformation_matrix()
robot_urdf_path = format_path(uenv.agent.urdf_path)
robot_srdf_path = robot_urdf_path.replace("urdf", "srdf")
planner = mplib.Planner(
    urdf=format_path(uenv.agent.urdf_path),
    srdf=format_path(uenv.agent.urdf_path).replace("urdf", "srdf"),
    move_group="gripper_link",
)
OPEN_POS, CLOSE_POS = 1, -1
# extract mesh point cloud
# scene_full_pcd = uenv.gen_scene_pcd()
# scene_full_pcd_robot = transform_points(trans_world2robot, scene_full_pcd)
# planner.update_point_cloud(scene_full_pcd_robot)

print(f"=== GET EGO-CENTRIC OBSERVATIONS ===")
handcam = uenv.scene.sensors["fetch_hand"]
intrinsics = handcam.get_params()["intrinsic_cv"]
trans_cam2world = handcam.camera.get_model_matrix()
trans_world2ee = uenv.agent.tcp.pose.inv().to_transformation_matrix()
trans_cam2ee = trans_world2ee @ trans_cam2world
imgs, ego_pcd = get_imgs_pcd(handcam, args.visualize)

index = np.argwhere(imgs["actor_seg"].squeeze() == seg_id).T
ids = np.random.randint(0, index.shape[0], args.num_graspcenter)
centers = index[ids][:, [1, 0]]
centers_z = imgs["depth"][0, centers[:, 1], centers[:, 0], 0]
centers_uvz = torch.hstack((centers, centers_z[:, None]))
centers_cam = uvz2xyz(centers_uvz, intrinsic=intrinsics[0]) * torch.Tensor([1, -1, -1])
centers_ee = transform_points(trans_cam2ee, centers_cam.unsqueeze(0))[0]
scene_pc_ee = transform_points(trans_cam2ee, ego_pcd["xyz"])[0]  # (w*h, 3)

print(f"=== SETUP GRASP DETECTOR ===")
detector_para_dist = OrderedDict()
for key in ["scene_pc_ee", "centers_ee"]:
    detector_para_dist[key] = eval(key)

print(f"=== DETECT GRASPS ===")
ggtrans = np.eye(4)
pred_gg = get_grasps(grasp_detector, detector_para_dist)
grasp_num = len(pred_gg)
print("grasp nums: ", grasp_num)
pred_gg_ee = trans_graspgroup(pred_gg, ggtrans)

print(f"=== PLAN PREV-GRASP & TARGET-GRASP ===")
pred_gg_ee_prev_grasp = [deepcopy(pred_gg_ee), deepcopy(pred_gg_ee)]
pred_gg_ee_prev_grasp[0].translations += args.prev_grasp_offset * pred_gg_ee.rotations[:, :3, 2]
pred_gg_ee_prev_grasp[1].translations += args.grasp_offset * pred_gg_ee.rotations[:, :3, 2]

print(f"=== MOVE TO PREV-GRASP ===")
gg_world_homo_close_prev_grasp = []
for pred_gg_ee_i in pred_gg_ee_prev_grasp:
    gg_ee_homo = homo_transfer(R=pred_gg_ee_i.rotations, T=pred_gg_ee_i.translations)  # (N, 4, 4)
    gg_world_homo = np.linalg.inv(trans_world2ee) @ gg_ee_homo  # (N, 4, 4)
    gg_world_homo_sym = gg_world_homo * np.array([-1, -1, 1, 1])  # (N, 4, 4)
    gg_world_homo_quat = np.roll(
        Rotation.from_matrix(gg_world_homo[:, :3, :3]).as_quat(), 1, axis=1
    )  # (N, 4) xyzw
    gg_world_homo_sym_quat = np.roll(
        Rotation.from_matrix(gg_world_homo_sym[:, :3, :3]).as_quat(), 1, axis=1
    )  # (N, 4) xyzw
    tcp_quat = np.array(uenv.agent.tcp.pose.q)
    angle1 = angle_distance_simple(gg_world_homo_quat, tcp_quat)
    angle2 = angle_distance_simple(gg_world_homo_sym_quat, tcp_quat)
    pose_id = np.argmin(np.stack([angle1, angle2]), axis=0)
    gg_world_homo_close = np.stack([gg_world_homo, gg_world_homo_sym])[pose_id, np.arange(grasp_num), ...]  # (N, 4, 4)
    gg_world_homo_close_prev_grasp.append(gg_world_homo_close)

init_plan_status = None
plan_cnt = -1
plan_num = min(args.max_plan_nums, grasp_num)
while init_plan_status != "Success" and plan_cnt + 2 <= plan_num:
    plan_cnt += 1
    ## grasp object
    pre_grasp_matrix = trans_world2robot @ gg_world_homo_close_prev_grasp[0][plan_cnt]
    pre_grasp = sapien.Pose(np.matrix(pre_grasp_matrix.numpy()))
    pre_grasp_list = list(pre_grasp.p) + list(pre_grasp.q)
    # status, goal_qposes = planner.IK(
    #         pre_grasp_list,
    #         uenv.agent.robot.get_qpos().cpu().numpy()[0],
    #         mask=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         threshold=0.05,
    # )
    # if status == "Success":
    #     plan_0 = planner.plan_qpos_to_qpos(
    #         goal_qposes,
    #         uenv.agent.robot.get_qpos().cpu().numpy()[0],
    #         time_step=1 / 250,
    #         fixed_joint_indices=range(1),
    #     )
    # init_plan_status = status
    # print(f"Plan {plan_cnt}: ", status)
    plan_0 = planner.plan_screw(
        target_pose=pre_grasp_list,
        qpos=uenv.agent.robot.get_qpos().cpu().numpy()[0],
        time_step=0.02,
        # use_point_cloud=True,
        use_attach=False,
        wrt_world=False,
        verbose=False,
    )
    init_plan_status = plan_0["status"]
    print(f"Plan {plan_cnt}: ", plan_0["status"])

if init_plan_status == "Success":
    traj_0 = plan_0["position"]
    for i in range(len(traj_0)):
        action = convert_traj_to_action(uenv, traj_0[i], OPEN_POS)
        venv.step(action)
        venv.render()

print(f"=== MOVE TO TARGET-GRASP ===")
grasp_matrix = trans_world2robot @ gg_world_homo_close_prev_grasp[1][plan_cnt]
grasp_pose = sapien.Pose(np.matrix(grasp_matrix.numpy()))
grasp_pose_list = list(grasp_pose.p) + list(grasp_pose.q)
plan_1 = planner.plan_screw(
    target_pose=grasp_pose_list,
    qpos=uenv.agent.robot.get_qpos().cpu().numpy()[0],
    time_step=0.02,
    use_point_cloud=False,
    use_attach=False,
    wrt_world=False,
    verbose=False,
)
if plan_1["status"] == "Success":
    traj_1 = plan_1["position"]
    for i in range(len(traj_1)):
        action = convert_traj_to_action(uenv, traj_1[i], OPEN_POS)
        venv.step(action)
        venv.render()

    print(f"=== CLOSE GRIPPER ===")
    for i in range(30):
        action = convert_traj_to_action(uenv, traj_1[-1], CLOSE_POS)
        venv.step(action)
        venv.render()

goal_pose = list(grasp_pose.p+np.array([0,0,0.2])) + list(grasp_pose.q)
plan_2 = planner.plan_screw(
    target_pose=goal_pose,
    qpos=uenv.agent.robot.get_qpos().cpu().numpy()[0],
    time_step=0.02,
    use_point_cloud=False,
    use_attach=False,
    wrt_world=False,
    verbose=False,
)
if plan_2["status"] == "Success":
    traj_2 = plan_2["position"]
    for i in range(len(traj_2)):
        action = convert_traj_to_action(uenv, traj_2[i], CLOSE_POS)
        venv.step(action)
        venv.render()


viewer = venv.render()
viewer.paused = True
while True:
    venv.render()
    