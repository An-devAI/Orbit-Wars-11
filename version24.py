# Score: 1146.6
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

# Ensure orbit_lite and other local modules are discoverable
try:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE_DIR = os.getcwd()
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

import torch
from torch import Tensor
import numpy as np
from scipy.optimize import linear_sum_assignment

# Internal Library Imports
from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    LaunchEntries,
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves


@dataclass(frozen=True)
class OrbitalOverlordConfig:
    """Cấu hình tham số chiến thuật cho Orbital Overlord."""
    horizon: int = 20
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12
    max_defensive_targets: int = 4
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5
    min_ships_to_launch: float = 4.0
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3


# Preset đặc thù cho trận 4 người (FFA)
PRESET_FFA_4P = dataclasses.replace(
    OrbitalOverlordConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
)


class OverlordMemory:
    """Lưu trữ trạng thái liên phiên của Bot (cache di chuyển, số người chơi)."""
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class OverlordRuntime:
    """Trình thực thi chính, quản lý vòng đời và xử lý tensor."""
    def __init__(self, memory: OverlordMemory | None = None) -> None:
        self.memory = memory if memory is not None else OverlordMemory()

    def reset(self) -> None:
        self.memory.reset()

    def get_action_tensor(self, obs_tensors: dict) -> dict:
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        
        # Chọn cấu hình dựa trên số lượng đối thủ
        config = PRESET_FFA_4P if int(mem.cached_player_count) >= 4 else OrbitalOverlordConfig()
        
        action_row = execute_strategic_turn(
            obs_tensors, 
            config=config,
            player_count=int(mem.cached_player_count), 
            memory=mem,
        )
        mem.last_sparse_action_row = action_row
        return action_row

# --- Core Logic Functions ---

def execute_strategic_turn(obs_tensors: dict, *, config: OrbitalOverlordConfig, player_count: int, memory: OverlordMemory) -> dict:
    """Điều phối toàn bộ quy trình tính toán trong một lượt."""
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    
    if obs.P == 0:
        return empty_action_row(device)

    # Cập nhật và cache trạng thái di chuyển của các hành tinh
    m_cfg = MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )
    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=m_cfg,
        cached_movement=memory.movement,
    )
    memory.movement = movement
    
    # Xây dựng cache khoảng cách và dự báo trạng thái quân đồn trú
    dist_cache = build_distance_cache(movement, max_k=int(config.horizon))
    h_max = int(config.horizon)
    garrison_status = movement.garrison_status(max_horizon=h_max)
    alive_by_step = movement.alive_by_step[: h_max + 1]

    # Lập kế hoạch phóng quân
    launch_entries = plan_tactical_maneuvers(
        movement=movement, 
        obs=obs, 
        obs_tensors=obs_tensors, 
        cache=dist_cache,
        garrison_status=garrison_status, 
        prod=movement.planet_prod,
        alive_by_step=alive_by_step, 
        config=config, 
        player_count=int(player_count),
    )
    
    # Hợp nhất và chuyển đổi sang định dạng payload
    launch_entries = disambiguate_duplicate_launches(launch_entries)
    planned_launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, 
        movement=movement, 
        entries=launch_entries, 
        player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, 
        launches=planned_launches, 
        owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(launch_entries, planet_ids=planet_ids)


def plan_tactical_maneuvers(
    *, movement: PlanetMovement, obs, obs_tensors: dict, cache,
    garrison_status, prod: Tensor, alive_by_step: Tensor,
    config: OrbitalOverlordConfig, player_count: int,
) -> LaunchEntries:
    """Tạo các đợt tấn công và điều quân phòng thủ."""
    P, device, dtype, pid = obs.P, obs.device, obs.ships.dtype, int(obs.player_id)
    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    # Lọc các hành tinh nguồn hợp lệ
    src_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(src_mask.any()): 
        return _empty_entries(device, dtype)

    # Tìm kiếm ứng viên Nguồn và Đích
    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    s_idx, s_exists = _candidate_indices(obs.ships, src_mask, S_cap)
    t_idx, t_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=src_mask,
    )
    if not bool(t_exists.any()): 
        return _empty_entries(device, dtype)
    
    S, T = int(s_idx.shape[0]), int(t_idx.shape[0])
    target_is_mine = obs.owned[t_idx.clamp(0, P - 1)]

    # Tính toán lượng quân có thể rút ra an toàn (drain)
    src_ships = obs.ships[s_idx.clamp(0, P - 1)].to(dtype)
    h_eff_tensor = torch.full((), float(H), dtype=dtype, device=device)
    drain_limit = safe_drain(
        garrison_status, source_idx=s_idx, source_ships=src_ships,
        H_eff=h_eff_tensor, player_id=pid,
    )

    # Tính ngưỡng quân cần thiết để chiếm đóng (capture floor)
    eta_limit = torch.full((T,), float(K_eta), dtype=dtype, device=device)
    cap_floor = capture_floor(
        garrison_status, target_idx=t_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
    )
    K_steps = int(cap_floor.shape[-1])
    
    max_send_per_lane = drain_limit.view(S, 1).expand(S, T).floor()

    # Kiểm tra khả năng tiếp cận (Reachability)
    is_reachable = reachable_mask(
        movement, source_idx=s_idx, target_idx=t_idx,
        fleet_sizes=max_send_per_lane.unsqueeze(-1), eta_cap=eta_limit,
    ).squeeze(-1)
    
    # Tính góc bắn đón hạm đội
    intercept = intercept_angle(movement, s_idx.unsqueeze(1), t_idx.unsqueeze(0), max_send_per_lane, active=is_reachable)
    angles, etas = intercept["angle"], intercept["eta"]
    viable_maneuver = intercept["viable"] & (etas <= eta_limit.view(1, T))

    # Lấy giá trị sàn tại thời điểm đến dự kiến
    if K_steps > 0:
        k_indices = (etas.clamp(min=1.0, max=float(K_steps)).ceil().long() - 1).clamp(0, K_steps - 1)
        floor_at_arrival = cap_floor.unsqueeze(0).expand(S, T, K_steps).gather(-1, k_indices.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arrival = torch.ones(S, T, dtype=dtype, device=device)

    # Chiến thuật: Gửi lượng quân tối đa có thể để đảm bảo chiến thắng
    proposed_sizes = max_send_per_lane.floor()

    # Điều kiện phóng quân: 
    # 1. Nếu chiếm đóng: Quân gửi >= Sàn. 
    # 2. Nếu tiếp viện: Quân gửi >= 1.
    is_offensive_win = proposed_sizes >= floor_at_arrival
    is_friendly_support = target_is_mine.view(1, T).expand(S, T)
    
    valid_mask = (
        viable_maneuver & (is_offensive_win | is_friendly_support) & (proposed_sizes >= 1.0) 
        & (s_idx.view(S, 1) != t_idx.view(1, T))
        & s_exists.view(S, 1) & t_exists.view(1, T)
    )

    # Chuẩn bị dữ liệu cho bộ tối ưu hóa toàn cục
    total_candidates = S * T
    flat_src = s_idx.view(S, 1).expand(S, T).reshape(total_candidates)
    flat_tgt = t_idx.view(1, T).expand(S, T).reshape(total_candidates)
    flat_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(total_candidates)
    flat_send = torch.where(valid_mask, proposed_sizes, torch.zeros_like(proposed_sizes)).reshape(total_candidates)
    flat_angle = angles.reshape(total_candidates)
    flat_eta = torch.where(valid_mask, etas, torch.ones_like(etas)).reshape(total_candidates)
    flat_valid = valid_mask.reshape(total_candidates)

    # Đánh giá điểm số cho các ứng viên
    launch_set = make_launch_set(
        source_slots=flat_src.unsqueeze(-1), target_slots=flat_tgt.unsqueeze(-1),
        ships=flat_send.unsqueeze(-1), eta=flat_eta.unsqueeze(-1), valid=flat_valid.unsqueeze(-1), player_id=pid,
    )
    scores = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launch_set, player_id=pid,
    )
    scores = torch.where(flat_valid, scores, torch.full_like(scores, float("-inf")))

    # Giải bài toán phân công (Global Assignment)
    offensive_entries, remaining_budget = solve_assignment_problem(
        P=P, W=W, device=device, dtype=dtype, score=scores,
        cand_src=flat_src, cand_send=flat_send, cand_angle=flat_angle, cand_eta=flat_eta,
        cand_active=flat_valid, cand_tgt_slot=flat_tgt, cand_tgt_short=flat_tgt_short,
        source_budget=obs.ships.to(dtype).clone(),
        roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return offensive_entries

    # Tính toán áp lực địch để điều quân tái cấu trúc (Regroup)
    threat_map = calculate_threat_heatmap(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=remaining_budget, original_ships=obs.ships.to(dtype), pressure=threat_map,
        config=config, H=H,
    )
    
    return concat_launch_entries([offensive_entries, regroup_entries])


def solve_assignment_problem(
    P, W, device, dtype, score,
    cand_src, cand_send, cand_angle, cand_eta,
    cand_active, cand_tgt_slot, cand_tgt_short,
    source_budget, roi_threshold,
) -> tuple[LaunchEntries, Tensor]:
    """Sử dụng thuật toán Hungarian để tìm bộ lệnh phóng quân tối ưu."""
    total_cand = int(score.shape[0])    
    num_t = int(cand_tgt_short.max() + 1) if cand_tgt_short.numel() > 0 else 1
    num_s = total_cand // num_t
        
    src_ids = cand_src.reshape(num_s, num_t)[:, 0] 
    s_matrix = score.view(num_s, num_t)
    send_matrix = cand_send.view(num_s, num_t)
    active_matrix = cand_active.view(num_s, num_t)
    angle_matrix = cand_angle.view(num_s, num_t)
    eta_matrix = cand_eta.view(num_s, num_t)
    tgt_id_row = cand_tgt_slot.view(num_s, num_t)[0] 
    
    # Ràng buộc: 1 Nguồn -> max 4 Đích; 1 Đích <- max 3 Nguồn.
    STR_PER_SRC = 4
    SWARM_PER_TGT = 3
    
    cost_rows = num_s * STR_PER_SRC
    cost_cols = num_t * SWARM_PER_TGT    
    cost_mat = np.full((cost_rows, cost_cols), 1e7, dtype=np.float32)
    
    # Dọn dẹp điểm số
    s_clean = s_matrix.detach().clone()
    s_clean[torch.isnan(s_clean) | torch.isinf(s_clean)] = -1e6
    s_np = s_clean.cpu().numpy()
    active_np = active_matrix.detach().cpu().numpy()
    
    for s in range(num_s):
        for t in range(num_t):
            if active_np[s, t] and s_np[s, t] > roi_threshold:
                cost_val = -float(s_np[s, t])
                cost_mat[s*STR_PER_SRC : (s+1)*STR_PER_SRC, 
                         t*SWARM_PER_TGT : (t+1)*SWARM_PER_TGT] = cost_val
                            
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    
    candidates = []
    for r, c in zip(row_ind, col_ind):
        if cost_mat[r, c] < 1e7:
            s_idx = r // STR_PER_SRC
            t_idx = c // SWARM_PER_TGT
            candidates.append((s_idx, t_idx, -cost_mat[r, c]))
            
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    allocated_send = torch.zeros((num_s, num_t), device=device, dtype=dtype)
    current_budget = source_budget.clone()
    drain_cap = send_matrix.max(dim=1)[0].clone() 
    
    for s, t, scr in candidates:
        if allocated_send[s, t] > 0: continue
            
        req = send_matrix[s, t].floor()
        if req < 1.0: continue
            
        s_slot = src_ids[s].long()
        avail = torch.min(current_budget[s_slot], drain_cap[s])
        
        if avail >= req:
            allocated_send[s, t] = req
            current_budget[s_slot] -= req
            drain_cap[s] -= req

    keep_mask = allocated_send >= 1.0
    final_s_idx, final_t_idx = torch.where(keep_mask)
    
    if final_s_idx.shape[0] > W:
        top_scores = s_matrix[final_s_idx, final_t_idx]
        _, top_k = torch.topk(top_scores, k=int(W))
        final_s_idx, final_t_idx = final_s_idx[top_k], final_t_idx[top_k]
    
    num_final = int(final_s_idx.shape[0])
    if num_final == 0:
        return _empty_entries(device, dtype), current_budget
        
    entries = LaunchEntries(
        source_slots=src_ids[final_s_idx].long(),
        target_slots=tgt_id_row[final_t_idx].long(),
        ships=allocated_send[final_s_idx, final_t_idx],
        angle=angle_matrix[final_s_idx, final_t_idx],
        eta=eta_matrix[final_s_idx, final_t_idx],
        valid=torch.ones(num_final, device=device, dtype=torch.bool),
    )
    return entries, current_budget


def calculate_threat_heatmap(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Tính toán bản đồ nhiệt áp lực từ quân địch dựa trên khoảng cách và quân số."""
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    
    dist0 = cache.cross_dist[0].to(dtype)                                   
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          
    reach_range = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)    
    
    enemy_mask = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  
    eye = torch.eye(P, device=device, dtype=torch.bool)
    
    valid_links = enemy_mask.view(P, 1) & obs.alive.view(1, P) & ~eye             
    intensity_decay = (1.0 - dist0 / reach_range).clamp(min=0.0)                       
    
    contributions = torch.where(valid_links, ships.view(P, 1) * intensity_decay, torch.zeros_like(intensity_decay))
    return contributions.sum(dim=0)

_GLOBAL_RUNTIME = OverlordRuntime()

def agent(obs, config=None):
    """Điểm kết nối với môi trường Kaggle."""
    player_id = int(obs.get("player", 0) if isinstance(obs, dict) else obs.player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    
    with torch.no_grad():
        sparse_row = _GLOBAL_RUNTIME.get_action_tensor(obs_tensors)
    
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
