# Score: 926.7
from __future__ import annotations
import math
import time
import logging
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field

# Cấu hình logging ra file
logging.basicConfig(
    filename='bot_debug.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(message)s'
)
logger = logging.getLogger('OrbitBot')

# Hằng số
BOARD        = 100.0
SUN_X        = 50.0
SUN_Y        = 50.0
SUN_R        = 10.0
MAX_SPEED    = 6.0
SUN_MARGIN   = 1.5
ROTATION_LIMIT = 50.0
TOTAL_STEPS  = 500

# TIERS DỰ ĐOÁN
HORIZON           = 55
PROACTIVE_HORIZON = 10
AIM_ITERATIONS    = 20
COMET_HORIZON     = 12

# Phase thresholds
EARLY_TURN    = 55
OPENING_TURN  = 110
LATE_REM      = 75
VERY_LATE_REM = 30

# Tham số tấn công / phòng thủ
NEUTRAL_MARGIN       = 2
ENEMY_MARGIN         = 5

# Mission weighting params
ALPHA, BETA, GAMMA, DELTA, EPSILON, MAX_ARRIVAL_TURNS = 1.0, 0.10, 0.70, 8.0, 2.0, 20
STEAL_AFTER_ENEMY_CAPTURE_DELAY = 1
STEAL_WAIT_MAX_TURNS = 25

# Rear push
REAR_MIN_SHIPS   = 15
REAR_MAX_TURNS   = 40
REAR_DIST_RATIO  = 1.25

# Comet evacuation
COMET_EVAC_TURNS = 6

# Timing
DEADLINE_SOFT    = 0.82

# Persistent locks for Relay and Steal missions
RELAY_LOCKS = {}
STEAL_LOCKS = {} # pid -> unlock_turn
LAST_LOCK_STEP = -1

def active_locks(step: int) -> set[int]:
    global LAST_LOCK_STEP
    if step < LAST_LOCK_STEP:
        RELAY_LOCKS.clear()
        STEAL_LOCKS.clear()
    LAST_LOCK_STEP = step
    
    # Clean expired relay locks
    expired_r = [pid for pid, ut in RELAY_LOCKS.items() if ut <= step]
    for pid in expired_r: RELAY_LOCKS.pop(pid, None)
    
    # Clean expired steal locks
    expired_s = [pid for pid, ut in STEAL_LOCKS.items() if ut <= step]
    for pid in expired_s: STEAL_LOCKS.pop(pid, None)
    
    return set(RELAY_LOCKS.keys()) | set(STEAL_LOCKS.keys())

Planet = namedtuple("Planet", ["id","owner","x","y","radius","ships","production"])
Fleet  = namedtuple("Fleet",  ["id","owner","x","y","angle","from_planet_id","ships"])

# Toán học

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def fleet_speed(ships):
    if ships <= 1: return 1.0
    ratio = math.log(max(ships, 1)) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * (max(0.0, min(1.0, ratio)) ** 1.5)

def seg_min_dist(x1, y1, x2, y2, px, py):
    dx, dy = x2 - x1, y2 - y1
    lsq = dx*dx + dy*dy
    if lsq < 1e-12:
        return math.hypot(x1 - px, y1 - py)
    t = max(0.0, min(1.0, ((px - x1)*dx + (py - y1)*dy) / lsq))
    return math.hypot(x1 + t*dx - px, y1 + t*dy - py)

def path_hits_sun(x1, y1, x2, y2, margin=SUN_MARGIN):
    return seg_min_dist(x1, y1, x2, y2, SUN_X, SUN_Y) < SUN_R + margin

def path_hits_planet(x1, y1, x2, y2, px, py, pr):
    return seg_min_dist(x1, y1, x2, y2, px, py) < pr

def get_planet_pos(p_id, t, ang_vel, initial_by_id, comet_map, planet_by_id):
    if p_id in comet_map:
        g = comet_map[p_id]
        pids, paths, pi = g["planet_ids"], g["paths"], g["path_index"]
        idx = pids.index(p_id)
        fi = pi + int(t)
        if 0 <= fi < len(paths[idx]):
            return paths[idx][fi][0], paths[idx][fi][1]
        return None
    
    p = planet_by_id[p_id]
    init = initial_by_id.get(p_id)
    if init is None: return p.x, p.y
    r = math.hypot(init.x - SUN_X, init.y - SUN_Y)
    if r + init.radius >= ROTATION_LIMIT: return p.x, p.y
    cur_ang = math.atan2(p.y - SUN_Y, p.x - SUN_X)
    ang_t = cur_ang + ang_vel * t
    return SUN_X + r * math.cos(ang_t), SUN_Y + r * math.sin(ang_t)

def safe_angle(x1, y1, x2, y2):
    direct = math.atan2(y2 - y1, x2 - x1)
    sun_obs_r = SUN_R + SUN_MARGIN
    if seg_min_dist(x1, y1, x2, y2, SUN_X, SUN_Y) >= sun_obs_r:
        return direct
    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)
    if d <= sun_obs_r: return direct 
    safe_orad = sun_obs_r + 0.1
    half = math.asin(min(1.0, safe_orad / d))
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    cand1 = to_sun + half
    cand2 = to_sun - half
    def angle_diff(a):
        dd = (a - direct) % (2 * math.pi)
        return min(dd, 2 * math.pi - dd)
    return cand1 if angle_diff(cand1) < angle_diff(cand2) else cand2

def comet_life(pid, comet_map):
    g = comet_map.get(pid)
    if not g: return 0
    pids, paths, pi = g["planet_ids"], g["paths"], g["path_index"]
    idx = pids.index(pid)
    if idx < len(paths): return max(0, len(paths[idx]) - pi)
    return 0

# Combat & Timeline Simulation

def resolve_combat(owner, garrison, arrivals):
    by_owner = {}
    for _, o, s in arrivals:
        by_owner[o] = by_owner.get(o, 0) + s
    if not by_owner: return owner, max(0.0, garrison)
    sorted_p = sorted(by_owner.items(), key=lambda x: -x[1])
    top_o, top_s = sorted_p[0]
    if len(sorted_p) > 1 and sorted_p[1][1] == top_s:
        return owner, max(0.0, garrison)
    surv_s = top_s - (sorted_p[1][1] if len(sorted_p) > 1 else 0)
    if surv_s <= 0: return owner, max(0.0, garrison)
    if top_o == owner: return owner, garrison + surv_s
    garrison -= surv_s
    return (top_o, -garrison) if garrison < 0 else (owner, garrison)

def simulate_timeline(planet, raw_arrivals, player, horizon):
    horizon = max(0, int(horizon))
    by_turn = defaultdict(list)
    for turns, o, s in raw_arrivals:
        eta = max(1, int(math.ceil(turns)))
        if eta <= horizon and s > 0:
            by_turn[eta].append((eta, o, int(s)))

    owner, garrison = planet.owner, float(planet.ships)
    owner_at, ships_at, fall_turn = {0: owner}, {0: max(0.0, garrison)}, None

    for t in range(1, horizon + 1):
        if owner != -1: garrison += planet.production
        grp = by_turn.get(t, [])
        if grp:
            prev = owner
            owner, garrison = resolve_combat(owner, garrison, grp)
            if prev == player and owner != player and fall_turn is None:
                fall_turn = t
        owner_at[t], ships_at[t] = owner, max(0.0, garrison)

    keep_needed, holds_full = 0, True
    if planet.owner == player:
        def survives(keep):
            o, g = planet.owner, float(keep)
            for t in range(1, horizon + 1):
                if o != -1: g += planet.production
                grp = by_turn.get(t, [])
                if grp:
                    o, g = resolve_combat(o, g, grp)
                    if o != player: return False
            return o == player
        if survives(int(planet.ships)):
            lo, hi = 0, int(planet.ships)
            while lo < hi:
                mid = (lo + hi) // 2
                if survives(mid): hi = mid
                else: lo = mid + 1
            keep_needed = lo
        else:
            holds_full, keep_needed = False, int(planet.ships)

    return {
        "owner_at": owner_at, "ships_at": ships_at,
        "keep_needed": keep_needed, "fall_turn": fall_turn,
        "holds_full": holds_full, "horizon": horizon,
    }

def build_arrivals(fleets, planets, ang_vel, initial_by_id, comet_map, planet_by_id):
    arrivals = {p.id: [] for p in planets}
    plist = list(planets)
    for f in fleets:
        spd = fleet_speed(f.ships)
        dx, dy = math.cos(f.angle), math.sin(f.angle)
        cur_fx, cur_fy = f.x, f.y
        for t in range(1, 120):
            hit_p = None
            new_fx, new_fy = cur_fx + dx * spd, cur_fy + dy * spd
            for p in plist:
                pos = get_planet_pos(p.id, t - 1, ang_vel, initial_by_id, comet_map, planet_by_id)
                if pos is None: continue
                if path_hits_planet(cur_fx, cur_fy, new_fx, new_fy, pos[0], pos[1], p.radius):
                    hit_p = p; break
            if hit_p:
                arrivals[hit_p.id].append((t, f.owner, int(f.ships)))
                break
            cur_fx, cur_fy = new_fx, new_fy
            if not (-50 <= cur_fx <= 150 and -50 <= cur_fy <= 150): break
    return arrivals

def is_path_safe(w, src_id, x, y, angle, ships, tgt_id, start_time=0):
    spd = fleet_speed(ships)
    dx, dy = math.cos(angle) , math.sin(angle)
    cur_x, cur_y = x, y
    for t in range(1, 150):
        next_x, next_y = cur_x + dx * spd, cur_y + dy * spd
        if seg_min_dist(cur_x, cur_y, next_x, next_y, SUN_X, SUN_Y) < SUN_R + SUN_MARGIN:
            return False
        hit_p = None
        for p in w.planets:
            if p.id == src_id: continue
            pos = w.predict_pos(p.id, start_time + t - 1)
            if pos is None: continue
            if path_hits_planet(cur_x, cur_y, next_x, next_y, pos[0], pos[1], p.radius):
                hit_p = p; break
        if hit_p: return hit_p.id == tgt_id
        cur_x, cur_y = next_x, next_y
        if not (-20 <= cur_x <= 120 and -20 <= cur_y <= 120): break
    return True

# WorldModel

class WorldModel:
    def __init__(self, player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids):
        self.player, self.step = player, step
        self.planets, self.fleets = planets, fleets
        self.initial_by_id, self.ang_vel = initial_by_id, ang_vel
        self.comets, self.comet_ids = comets, set(comet_ids)
        self.planet_by_id = {p.id: p for p in planets}
        self.comet_map = {}
        for g in comets:
            for pid in g.get("planet_ids", []): self.comet_map[pid] = g
        self.my_planets = [p for p in planets if p.owner == player]
        self.enemy_planets = [p for p in planets if p.owner not in (-1, player)]
        self.neutral_planets = [p for p in planets if p.owner == -1]
        self.remaining = max(1, TOTAL_STEPS - step)
        self.is_early, self.is_opening = step < EARLY_TURN, step < OPENING_TURN
        self.is_late, self.is_very_late = self.remaining < LATE_REM, self.remaining < VERY_LATE_REM
        self.owner_ships, self.owner_prod = defaultdict(int), defaultdict(int)
        for p in planets:
            if p.owner != -1:
                self.owner_ships[p.owner] += int(p.ships)
                self.owner_prod[p.owner] += int(p.production)
        for f in fleets: self.owner_ships[f.owner] += int(f.ships)
        self.my_total = self.owner_ships.get(player, 0)
        self.enemy_total = sum(v for k, v in self.owner_ships.items() if k != player)
        self.my_prod = self.owner_prod.get(player, 0)
        self.enemy_prod = sum(v for k, v in self.owner_prod.items() if k != player)
        self.arrivals = build_arrivals(fleets, planets, ang_vel, initial_by_id, self.comet_map, self.planet_by_id)
        self.timelines = {p.id:simulate_timeline(p, self.arrivals[p.id], player, HORIZON) for p in planets}
        total = self.my_total + self.enemy_total
        self.domination = (self.my_total - self.enemy_total) / max(1, total)
        self.is_finishing = (self.domination > 0.33 and self.my_prod > self.enemy_prod * 1.2 and step > 100)
        self._pos_cache_20 = {}
        for p in planets:
            self._pos_cache_20[p.id] = get_planet_pos(p.id, 20, ang_vel, initial_by_id, self.comet_map, self.planet_by_id)
        if self.my_planets:
            sum_x, sum_y, total_w = 0.0, 0.0, 0.0
            sum_xf, sum_yf = 0.0, 0.0
            for p in self.my_planets:
                w_p = max(1.0, p.production)
                sum_x += p.x * w_p; sum_y += p.y * w_p; total_w += w_p
                pf = self._pos_cache_20.get(p.id)
                if pf: sum_xf += pf[0] * w_p; sum_yf += pf[1] * w_p
                else: sum_xf += p.x * w_p; sum_yf += p.y * w_p
            self.my_com_now = (sum_x / total_w, sum_y / total_w)
            self.my_com_f20 = (sum_xf / total_w, sum_yf / total_w)
        else:
            self.my_com_now = (SUN_X, SUN_Y); self.my_com_f20 = (SUN_X, SUN_Y)
        self._shot_cache = {}

    def predict_pos(self, pid, t):
        if t == 0: 
            p = self.planet_by_id.get(pid)
            return (p.x, p.y) if p else None
        if t == 20: return self._pos_cache_20.get(pid)
        return get_planet_pos(pid, t, self.ang_vel, self.initial_by_id, self.comet_map, self.planet_by_id)

    def aim(self, src, tgt, ships):
        key = (src.id, tgt.id, int(ships))
        if key in self._shot_cache: return self._shot_cache[key]
        spd = fleet_speed(max(1, int(ships)))
        intercept_turn, tx, ty = None, tgt.x, tgt.y
        for t_float in [x * 0.5 for x in range(2, 300)]:
            pos = self.predict_pos(tgt.id, t_float)
            if pos is None: break 
            d = dist(src.x, src.y, pos[0], pos[1])
            if d - tgt.radius <= t_float * spd:
                intercept_turn = t_float; tx, ty = pos[0], pos[1]; break
        if intercept_turn is None:
            self._shot_cache[key] = None; return None
        direct_angle = math.atan2(ty - src.y, tx - src.x)
        safe_ang = safe_angle(src.x, src.y, tx, ty) 
        angle_diff = abs((safe_ang - direct_angle + math.pi) % (2 * math.pi) - math.pi)
        dist_to_tgt = dist(src.x, src.y, tx, ty)
        target_angular_width = math.atan2(tgt.radius, max(1.0, dist_to_tgt))
        if angle_diff > target_angular_width * 0.8:
            self._shot_cache[key] = None; return None
        ex = src.x + math.cos(safe_ang) * 1000; ey = src.y + math.sin(safe_ang) * 1000
        if seg_min_dist(src.x, src.y, ex, ey, tx, ty) > tgt.radius + 0.1:
            self._shot_cache[key] = None; return None
        res = (safe_ang, int(math.ceil(intercept_turn)), tx, ty)
        self._shot_cache[key] = res; return res

    def keep_needed(self, pid): return self.timelines[pid]["keep_needed"]
    def holds_full(self, pid): return self.timelines[pid]["holds_full"]
    def incoming_friendly(self, pid): return sum(s for _, o, s in self.arrivals.get(pid, []) if o == self.player)
    def total_keep(self, pid): return self.keep_needed(pid)
    def attack_budget(self, pid, spent):
        return max(0, int(self.planet_by_id[pid].ships) - spent.get(pid, 0) - self.total_keep(pid))

# Phase detection

def detect_phase(w: WorldModel) -> str:
    return "expand" if w.step < 55 else "combat"

@dataclass
class Mission:
    src_id: int; tgt_id: int; angle: float; turns: int; cost: int; score: float; mtype: str = "ATTACK"
    wait_turns: int = 0; relay_final_id: int | None = None; relay_unlock_turn: int | None = None
    debug_info: dict = field(default_factory=dict)

def min_ships_to_own_at(w: WorldModel, tgt_id: int, t_arrival: int, extra_arrivals: list, sim_until: int = None) -> int:
    tgt, base_arrivals = w.planet_by_id[tgt_id], list(w.arrivals.get(tgt_id, []))
    if sim_until is None: sim_until = t_arrival + 1
    margin = 0
    if tgt.owner == -1: margin = NEUTRAL_MARGIN
    elif tgt.owner != w.player: margin = ENEMY_MARGIN
    def owns_after(send: int) -> bool:
        arrivals = base_arrivals + extra_arrivals + ([(t_arrival, w.player, send)] if send > 0 else [])
        o, g, by_turn = tgt.owner, float(tgt.ships), defaultdict(list)
        for eta, own, s in arrivals:
            eta_i = max(1, int(math.ceil(eta)))
            if eta_i <= sim_until and s > 0: by_turn[eta_i].append((eta_i, own, int(s)))
        for t in range(1, sim_until + 1):
            if o != -1: g += tgt.production
            grp = by_turn.get(t, [])
            if grp: o, g = resolve_combat(o, g, grp)
        return o == w.player and g >= (1 + margin)
    if owns_after(0): return 0
    if not owns_after(999): return 9999
    lo, hi = 1, 999
    while lo < hi:
        mid = (lo + hi) // 2
        if owns_after(mid): hi = mid
        else: lo = mid + 1
    return lo

def position_bonus(w: WorldModel, tgt: Planet, phase: str) -> float:
    bonus = (tgt.production ** 1.5) * 5.0 
    if tgt.owner not in (-1, w.player):
        bonus += 12.0
        if w.is_finishing: bonus += 15.0
        if w.owner_ships.get(tgt.owner, 999) < 60: bonus += 10.0
    if dist(tgt.x, tgt.y, SUN_X, SUN_Y) + tgt.radius >= ROTATION_LIMIT: bonus += 4.0
    if phase == "expand":
        if tgt.owner == -1: bonus += 15.0
    else:
        if tgt.owner not in (-1, w.player): bonus *= 1.3
    friendly_in = w.incoming_friendly(tgt.id)
    if friendly_in > 0: bonus -= friendly_in * 0.05
    pos_f = w.predict_pos(tgt.id, 20)
    if pos_f:
        d_now = dist(tgt.x, tgt.y, w.my_com_now[0], w.my_com_now[1])
        d_f20 = dist(pos_f[0], pos_f[1], w.my_com_f20[0], w.my_com_f20[1])
        bonus += (d_now - d_f20) * 0.2
    return bonus

def build_missions(w: WorldModel, spent: dict, targeted: set, planned_commitments: list, phase: str) -> list[Mission]:
    all_locks = active_locks(w.step)
    pool = [p for p in w.planets if not (p.id in targeted or (p.owner == w.player and w.holds_full(p.id)))]
    frontier = w.enemy_planets if w.enemy_planets else w.neutral_planets
    dist_to_front = {p.id: min(dist(p.x, p.y, t.x, t.y) for t in frontier) for p in w.my_planets} if frontier else {}
    f_scores = {p.id: (100.0 - dist_to_front.get(p.id, 100)) * 10.0 + p.production for p in w.my_planets}
    max_profit_from = defaultdict(float)
    vulnerable_friendly = [p.id for p in w.my_planets if w.timelines[p.id].get("fall_turn") is not None and w.timelines[p.id]["fall_turn"] <= MAX_ARRIVAL_TURNS]

    allow_rear_push = True
    if len(w.my_planets) < 2 or not (w.enemy_planets or w.neutral_planets): allow_rear_push = False
    if len(w.my_planets) < len(w.planets) / 5.0 and w.is_early: allow_rear_push = False
    if w.is_late: allow_rear_push = False

    raw_missions = []
    for src in w.my_planets:
        if src.id in all_locks: continue
        src_tl, keep = w.timelines[src.id], w.total_keep(src.id)
        for tgt in pool:
            if tgt.id == src.id: continue
            aim_now = w.aim(src, tgt, max(1, int(src.ships)))
            if not aim_now: continue
            angle, travel_turns, tx, ty = aim_now
            if travel_turns > MAX_ARRIVAL_TURNS: continue
            extra = [(eta, o, tgt_ref, s) for eta, o, tgt_ref, s in planned_commitments if tgt_ref == tgt.id]
            req_cost = min_ships_to_own_at(w, tgt.id, travel_turns, extra)
            if req_cost == 0 or req_cost >= 9999: continue
            wait_turns = 0
            if src_tl["ships_at"][0] < req_cost + keep:
                for dt in range(1, 21):
                    if src_tl["ships_at"].get(dt, 0) >= req_cost + keep: wait_turns = dt; break
            if wait_turns > 15: continue
            total_turns = travel_turns + wait_turns
            if wait_turns > 0:
                f_pos = w.predict_pos(src.id, wait_turns)
                if not f_pos: continue
                src_f = Planet(src.id, src.owner, f_pos[0], f_pos[1], src.radius, req_cost, src.production)
                aim_f = w.aim(src_f, tgt, req_cost)
                if not aim_f: continue
                angle, travel_turns, _, _ = aim_f; total_turns = travel_turns + wait_turns
                req_cost = min_ships_to_own_at(w, tgt.id, total_turns, extra)
                if req_cost == 0 or req_cost >= 9999: continue
            if total_turns > MAX_ARRIVAL_TURNS + 5: continue
            bonus = position_bonus(w, tgt, phase)
            mtype = "ATTACK"
            if tgt.owner == w.player: mtype = "DEFEND"
            elif tgt.owner == -1: mtype = "CAPTURE"
            target_arrival_t = total_turns
            if mtype == "CAPTURE":
                enemy_arr = [a for a in w.arrivals.get(tgt.id, []) if a[1] not in (-1, w.player)]
                if enemy_arr:
                    last_enemy_t = max(int(math.ceil(a[0])) for a in enemy_arr)
                    wait_needed = (last_enemy_t + 1) - travel_turns
                    my_budget = int(src.ships) - spent.get(src.id, 0) - keep
                    cap_cost = min_ships_to_own_at(w, tgt.id, travel_turns, extra)
                    should_steal = False
                    if travel_turns > last_enemy_t: should_steal = True
                    elif wait_needed <= 3: should_steal = True
                    elif cap_cost > my_budget and cap_cost != 9999: should_steal = True
                    if should_steal and wait_needed <= 15:
                        wait_turns = max(0, wait_needed); total_turns = travel_turns + wait_turns
                        target_arrival_t = total_turns; mtype = "STEAL"; bonus += 500.0
                        tl_tgt = w.timelines[tgt.id]; enemy_surv = tl_tgt["ships_at"].get(last_enemy_t, tl_tgt["ships_at"][tl_tgt["horizon"]])
                        owner_at_t = tl_tgt["owner_at"].get(last_enemy_t, tl_tgt["owner_at"][tl_tgt["horizon"]])
                        req_cost = min_ships_to_own_at(w, tgt.id, total_turns, extra)
                        if owner_at_t not in (-1, w.player):
                            steal_floor = int(enemy_surv) + int(tgt.production) + ENEMY_MARGIN
                            req_cost = max(req_cost, steal_floor)
                        if req_cost == 0 or req_cost >= 9999: continue
            if not is_path_safe(w, src.id, src.x, src.y, angle, req_cost, tgt.id, wait_turns): continue
            if mtype in ("ATTACK", "CAPTURE", "STEAL"):
                req_cost = max(req_cost, 1); full_send = int(src.ships) - spent.get(src.id, 0) - keep
                if full_send >= req_cost:
                    aim_full = w.aim(src, tgt, full_send)
                    if aim_full:
                        fa, ft, _, _ = aim_full
                        if mtype == "STEAL":
                            if target_arrival_t >= ft:
                                fw = target_arrival_t - ft
                                if fw <= 15: angle, travel_turns, wait_turns, total_turns, req_cost = fa, ft, fw, travel_turns+wait_turns, full_send
                        elif wait_turns == 0 and ft <= MAX_ARRIVAL_TURNS + 5:
                            req_at_full = min_ships_to_own_at(w, tgt.id, ft, extra)
                            if 0 < req_at_full <= full_send: angle, travel_turns, wait_turns, total_turns, req_cost = fa, ft, 0, ft, full_send
            tl_tgt = w.timelines[tgt.id]
            owner_at_arrival = tl_tgt["owner_at"].get(total_turns, tl_tgt["owner_at"][tl_tgt["horizon"]])
            ships_at_arrival = tl_tgt["ships_at"].get(total_turns, tl_tgt["ships_at"][tl_tgt["horizon"]])
            profit = tgt.production * (min(w.remaining, 999) - total_turns)
            if tgt.id in w.comet_ids:
                life = comet_life(tgt.id, w.comet_map)
                if life <= total_turns: continue
                profit = tgt.production * (min(w.remaining, life) - total_turns)
                if profit <= req_cost: continue
            if mtype == "ATTACK" and (ships_at_arrival - req_cost) < -req_cost * 0.5 and profit < req_cost * 2: continue
            if vulnerable_friendly:
                if mtype == "CAPTURE": bonus -= 100.0
                elif mtype == "DEFEND" and tgt.id in vulnerable_friendly: bonus += 200.0 + (tgt.production * 50.0)
            active_threat = sum(arr_s * (total_turns / max(total_turns, float(arr_eta)))**2 for arr_eta, arr_o, arr_s in w.arrivals.get(tgt.id, []) if arr_o not in (-1, w.player)) if mtype != "DEFEND" else 0.0
            potential_threat = sum(en.ships * (total_turns / max(total_turns, float(math.ceil(dist(en.x, en.y, tgt.x, tgt.y) / fleet_speed(en.ships)))))**2 for en in w.enemy_planets) if mtype != "DEFEND" else 0.0
            threat_total = min(1.5, (potential_threat + active_threat) / max(1.0, float(w.my_total)))
            if mtype == "CAPTURE":
                if threat_total > 0.3: bonus -= (threat_total * 20) ** 2
                if min_ships_to_own_at(w, tgt.id, total_turns, extra, sim_until=total_turns + 10) > req_cost * 1.5: bonus -= 150.0
            if threat_total > 0.4: bonus -= (travel_turns / 10.0) ** 2 * 10.0
            if mtype in ("ATTACK", "CAPTURE"):
                best_hop = None
                for relay in w.my_planets:
                    if relay.id in (src.id, tgt.id) or relay.id in all_locks: continue
                    if dist(relay.x, relay.y, tgt.x, tgt.y) >= dist(src.x, src.y, tgt.x, tgt.y) or f_scores.get(relay.id, 0) <= f_scores.get(src.id, 0): continue
                    bnow = int(src.ships) - spent.get(src.id, 0) - keep
                    if bnow < 1: continue
                    aim1 = w.aim(src, relay, bnow)
                    if aim1 and is_path_safe(w, src.id, src.x, src.y, aim1[0], bnow, relay.id):
                        t1, s_relay_t1 = aim1[1], w.timelines[relay.id]["ships_at"].get(aim1[1], relay.ships) + bnow
                        pos_t1 = w.predict_pos(relay.id, t1)
                        if pos_t1:
                            relay_f = Planet(relay.id, relay.owner, pos_t1[0], pos_t1[1], relay.radius, s_relay_t1, relay.production)
                            aim2 = w.aim(relay_f, tgt, s_relay_t1)
                            if aim2 and is_path_safe(w, relay.id, relay_f.x, relay_f.y, aim2[0], s_relay_t1, tgt.id, t1):
                                if t1 + aim2[1] < total_turns and s_relay_t1 >= min_ships_to_own_at(w, tgt.id, t1 + aim2[1], extra) + w.total_keep(relay.id):
                                    if best_hop is None or (t1 + aim2[1]) < best_hop[0]: best_hop = (t1 + aim2[1], relay, aim1[0], bnow, t1)
                if best_hop:
                    raw_missions.append((src.id, best_hop[1].id, best_hop[2], best_hop[4], best_hop[3], profit, position_bonus(w, best_hop[1], phase), 0.0, 0.0, "STAGED_RELAY", 0, tgt.id, w.step + best_hop[4])); continue
            raw_missions.append((src.id, tgt.id, angle, travel_turns, req_cost, profit, bonus, ships_at_arrival, threat_total, mtype, wait_turns))
            if mtype != "DEFEND": max_profit_from[src.id] = max(max_profit_from[src.id], profit)
    for src in w.my_planets:
        if src.id in all_locks: continue
        bud = int(src.ships) - spent.get(src.id, 0) - w.total_keep(src.id)
        if allow_rear_push and bud >= REAR_MIN_SHIPS and dist_to_front:
            candidates = [p for p in w.my_planets if dist_to_front.get(p.id, 0) < dist_to_front.get(src.id, 0) - 5 and f_scores.get(p.id, 0) > f_scores.get(src.id, 0)]
            if candidates:
                relay = min(candidates, key=lambda c: dist(src.x, src.y, c.x, c.y)); aim_r = w.aim(src, relay, bud)
                if aim_r and is_path_safe(w, src.id, src.x, src.y, aim_r[0], bud, relay.id):
                    raw_missions.append((src.id, relay.id, aim_r[0], aim_r[1], bud, max_profit_from[relay.id], position_bonus(w, relay, phase), w.timelines[relay.id]["ships_at"].get(aim_r[1], relay.ships), 0.0, "RELAY", 0))
    if not raw_missions: return []
    def get_stats(arr, baseline, prior_std):
        combined = arr + [baseline, baseline - prior_std, baseline + prior_std]; mean = sum(combined) / len(combined)
        std = math.sqrt(sum((x - mean)**2 for x in combined) / len(combined)); return mean, std
    m_p, s_p = get_stats([m[5] for m in raw_missions], 0.0, 20.0); m_t, s_t = get_stats([m[3] + m[10]*1.5 for m in raw_missions], 35.0, 15.0)
    m_b, s_b = get_stats([m[6] for m in raw_missions], 0.0, 5.0); m_c, s_c = get_stats([(m[7] - m[4]) for m in raw_missions], 0.0, 15.0); m_th, s_th = get_stats([m[8] for m in raw_missions], 0.1, 0.3)
    missions = []
    for r in raw_missions:
        sid, tid, ang, trn, cst, prf, bon, shp, thrt, mtyp, wait = r[:11]; rfid, rlock = (r[11], r[12]) if len(r) > 11 else (None, None)
        eff_t = trn + wait * 1.5; np = (prf - m_p) / s_p; nt = (eff_t - m_t) / s_t; nb = (bon - m_b) / s_b; nc = (abs(shp - cst) - m_c) / s_c; nth = (thrt - m_th) / s_th
        score = (ALPHA * np + BETA * nc - GAMMA * nt + DELTA * nb - EPSILON * nth)
        debug = {"raw": {"profit": prf, "cost": cst, "turns": trn, "wait": wait, "eff_t": eff_t, "bonus": bon, "threat": thrt, "ships_at": shp}, "norm": {"p": np, "c_diff": nc, "t": nt, "b": nb, "th": nth}, "weights": {"A": ALPHA, "B": BETA, "G": GAMMA, "D": DELTA, "E": EPSILON}}
        missions.append(Mission(src_id=sid, tgt_id=tid, angle=ang, turns=trn, cost=cst, score=score, mtype=mtyp, wait_turns=wait, relay_final_id=rfid, relay_unlock_turn=rlock, debug_info=debug))
    missions.sort(key=lambda m: -m.score); return missions

def knapsack_select(missions: list[Mission], budgets: dict[int, int], w: WorldModel) -> list[Mission]:
    rem_bud, chosen_tgts, chosen_srcs, selected = dict(budgets), set(), set(), []
    for m in missions:
        if m.src_id in chosen_srcs or m.tgt_id in chosen_tgts: continue
        if m.mtype == "STAGED_RELAY" and (m.tgt_id in chosen_srcs or m.relay_final_id in chosen_tgts): continue
        if m.wait_turns > 0:
            logger.info(f"[RESERVE] {m.src_id} for {m.tgt_id} in {m.wait_turns} turns | score={m.score:.3f}")
            chosen_srcs.add(m.src_id); chosen_tgts.add(m.tgt_id)
            if m.mtype == "STEAL": STEAL_LOCKS[m.src_id] = w.step + m.wait_turns
            continue
        if rem_bud.get(m.src_id, 0) < m.cost: continue
        selected.append(m); rem_bud[m.src_id] -= m.cost; chosen_tgts.add(m.tgt_id); chosen_srcs.add(m.src_id)
        if m.mtype == "STAGED_RELAY":
            chosen_srcs.add(m.tgt_id)
            if m.relay_final_id is not None: chosen_tgts.add(m.relay_final_id)
    return selected

def execute_missions(selected: list[Mission], w: WorldModel, spent: dict, moves: list, targeted: set, planned_commitments: list):
    for m in selected:
        available = w.attack_budget(m.src_id, spent)
        if available < m.cost:
            logger.info(f"[HOLD_DEFENSE] Turn {w.step}: skip {m.src_id}->{m.tgt_id}; budget={available}, cost={m.cost}"); continue
        logger.info(f"[{m.mtype}] Turn {w.step}: {m.src_id} -> {m.tgt_id} | score={m.score:.3f}")
        if m.mtype == "STAGED_RELAY": logger.info(f"  > RELAY: final={m.relay_final_id}, lock {m.tgt_id} until turn {m.relay_unlock_turn}")
        logger.info(f"  > RAW : profit={m.debug_info['raw']['profit']:.1f}, cost={m.cost}, turns={m.turns}, wait={m.wait_turns}, bonus={m.debug_info['raw']['bonus']:.1f}, threat={m.debug_info['raw']['threat']:.2f}, ships_at={m.debug_info['raw']['ships_at']:.1f}")
        moves.append([m.src_id, float(m.angle), int(m.cost)]); spent[m.src_id] = spent.get(m.src_id, 0) + m.cost; targeted.add(m.tgt_id)
        if m.mtype == "STAGED_RELAY":
            if m.relay_unlock_turn is not None: RELAY_LOCKS[m.tgt_id] = max(RELAY_LOCKS.get(m.tgt_id, 0), m.relay_unlock_turn)
            if m.relay_final_id is not None: targeted.add(m.relay_final_id)
        planned_commitments.append((m.turns, w.player, m.tgt_id, m.cost))

def phase_comet_evac(w: WorldModel, spent: dict, moves: list, planned_commitments: list):
    locks = active_locks(w.step)
    for src in w.my_planets:
        if src.id in locks or src.id not in w.comet_ids: continue
        life = comet_life(src.id, w.comet_map)
        if 0 < life < COMET_EVAC_TURNS:
            bud = w.attack_budget(src.id, spent)
            if bud < 1: continue
            others = [p for p in w.my_planets if p.id != src.id]
            if not others: continue
            tgt = min(others, key=lambda p: dist(src.x, src.y, p.x, p.y)); aim_r = w.aim(src, tgt, bud)
            if aim_r and is_path_safe(w, src.id, src.x, src.y, aim_r[0], bud, tgt.id):
                logger.info(f"[EVAC] Turn {w.step}: Comet {src.id} -> {tgt.id} (EVAC) | ships={bud}, life={life}")
                moves.append([src.id, float(aim_r[0]), bud]); spent[src.id] = spent.get(src.id, 0) + bud; planned_commitments.append((aim_r[1], w.player, tgt.id, bud))

def phase_tactical_evac(w: WorldModel, spent: dict, moves: list, planned_commitments: list):
    locks = active_locks(w.step)
    for src in w.my_planets:
        if src.id in locks or spent.get(src.id, 0) > 0: continue
        enemies_targeting, soon_arrival = set(), False
        for eta, o, s in w.arrivals.get(src.id, []):
            if o not in (-1, w.player):
                enemies_targeting.add(o)
                if eta <= 10: soon_arrival = True
        if len(enemies_targeting) >= 2 and soon_arrival:
            bud = w.attack_budget(src.id, spent)
            if bud < 1: continue
            others = [p for p in w.my_planets if p.id != src.id]
            if not others: continue
            safe_others = [p for p in others if w.timelines[p.id].get("fall_turn") is None]
            target_list = safe_others if safe_others else others; tgt = min(target_list, key=lambda p: dist(src.x, src.y, p.x, p.y)); aim_r = w.aim(src, tgt, bud)
            if aim_r and is_path_safe(w, src.id, src.x, src.y, aim_r[0], bud, tgt.id):
                logger.info(f"[TACTICAL_EVAC] Turn {w.step}: {src.id} -> {tgt.id} | ships={bud}, reason=GANGED")
                moves.append([src.id, float(aim_r[0]), bud]); spent[src.id] = spent.get(src.id, 0) + bud; planned_commitments.append((aim_r[1], w.player, tgt.id, bud))

def finalize(moves, w):
    used, final = defaultdict(int), []
    for src_id, angle, ships in moves:
        p = w.planet_by_id.get(src_id)
        if not p: continue
        send = min(int(ships), int(p.ships) - used[src_id])
        if send >= 1: final.append([src_id, float(angle), int(send)]); used[src_id] += send
    return final

def _get(obs, key, default=None): return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)

def update_params(step, initial_by_id, planets):
    global ALPHA, BETA, GAMMA, DELTA, EPSILON, MAX_ARRIVAL_TURNS
    tp = len(set(p.owner for p in initial_by_id.values() if p.owner != -1))
    if tp == 0: tp = len(set(p.owner for p in planets if p.owner != -1))
    is_1v1 = (tp <= 2); early_limit = 55 if is_1v1 else 15; is_early = (step <= early_limit)
    if globals().get("TUNE_MODE", False):
        if is_early:
            ALPHA = globals().get("EARLY_ALPHA", ALPHA); BETA = globals().get("EARLY_BETA", BETA); GAMMA = globals().get("EARLY_GAMMA", GAMMA)
            DELTA = globals().get("EARLY_DELTA", DELTA); EPSILON = globals().get("EARLY_EPSILON", EPSILON); MAX_ARRIVAL_TURNS = globals().get("EARLY_MAX_ARRIVAL_TURNS", MAX_ARRIVAL_TURNS)
        else:
            ALPHA = globals().get("LATE_ALPHA", ALPHA); BETA = globals().get("LATE_BETA", BETA); GAMMA = globals().get("LATE_GAMMA", GAMMA)
            DELTA = globals().get("LATE_DELTA", DELTA); EPSILON = globals().get("LATE_EPSILON", EPSILON); MAX_ARRIVAL_TURNS = globals().get("LATE_MAX_ARRIVAL_TURNS", MAX_ARRIVAL_TURNS)
        return
    if is_1v1:
        if is_early: ALPHA, BETA, GAMMA, DELTA, EPSILON, MAX_ARRIVAL_TURNS = 1.86, 0.71, 0.82, 2.34, 2.38, 21
        else: ALPHA, BETA, GAMMA, DELTA, EPSILON, MAX_ARRIVAL_TURNS = 3.52, 0.42, 1.36, 2.05, 2.27, 28
    else:
        if is_early: ALPHA, BETA, GAMMA, DELTA, EPSILON, MAX_ARRIVAL_TURNS = 3.61, 0.63, 4.56, 4.39, 4.69, 26
        else: ALPHA, BETA, GAMMA, DELTA, EPSILON, MAX_ARRIVAL_TURNS = 3.42, 0.77, 4.26, 4.12, 1.56, 34

def build_world(obs):
    player, step = _get(obs, "player", 0), _get(obs, "step", 0) or 0; raw_p, raw_f = _get(obs, "planets", []) or [], _get(obs, "fleets", []) or []
    ang_vel, raw_init = _get(obs, "angular_velocity", 0.03) or 0.0, _get(obs, "initial_planets", []) or []; comets, c_ids = _get(obs, "comets", []) or [], _get(obs, "comet_planet_ids", []) or []
    planets, fleets = [Planet(*p) for p in raw_p], [Fleet(*f) for f in raw_f]; init_map = {Planet(*p).id: Planet(*p) for p in raw_init}; update_params(step, init_map, planets)
    return WorldModel(player, step, planets, fleets, init_map, ang_vel, comets, c_ids)

def agent(obs, config=None):
    t0 = time.perf_counter(); w = build_world(obs)
    if not w.my_planets: return []
    phase = detect_phase(w); logger.info(f"--- Turn {w.step} | phase={phase} | ships={w.my_total}/{w.enemy_total} | my_prod={w.my_prod} ---")
    spent, moves, targeted, planned_commitments = {}, [], set(), []
    if (time.perf_counter() - t0) < DEADLINE_SOFT:
        budgets = {p.id: w.attack_budget(p.id, spent) for p in w.my_planets}; missions = build_missions(w, spent, targeted, planned_commitments, phase)
        execute_missions(knapsack_select(missions, budgets, w), w, spent, moves, targeted, planned_commitments)
    if (time.perf_counter() - t0) < DEADLINE_SOFT:
        phase_comet_evac(w, spent, moves, planned_commitments); phase_tactical_evac(w, spent, moves, planned_commitments)
    final_moves = finalize(moves, w); t_elapsed = time.perf_counter() - t0; logger.info(f"[TIME] Turn {w.step} took {t_elapsed:.4f}s"); return final_moves
__all__ = ["agent"]