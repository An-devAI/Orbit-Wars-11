# Score: 918.7
from __future__ import annotations
import math
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field

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
NEUTRAL_MARGIN       = 3
ENEMY_MARGIN         = 4
PROACTIVE_RATIO      = 0.22

# Rear push
REAR_MIN_SHIPS   = 25
REAR_SEND_RATIO  = 0.8
REAR_MAX_TURNS   = 40
REAR_DIST_RATIO  = 1.25

# Comet evacuation
COMET_EVAC_TURNS = 6

# Timing
DEADLINE_SOFT    = 0.82

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
    # 1. Tính góc trực tiếp và khoảng cách tới mục tiêu
    direct = math.atan2(y2 - y1, x2 - x1)
    
    # 2. KIỂM TRA LỖ HỔNG CHÍ MẠNG: Dùng chính x2, y2 làm điểm dừng của tia quét
    sun_obs_r = SUN_R + SUN_MARGIN
    if seg_min_dist(x1, y1, x2, y2, SUN_X, SUN_Y) >= sun_obs_r:
        # Nếu tia từ bắn (x1,y1) đến ĐÍCH (x2,y2) không chạm mặt trời -> Bay thẳng!
        return direct
        
    # 3. Nếu tia trực tiếp bị chặn bởi Mặt Trời, tìm góc tiếp tuyến để né
    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)
    
    # Bug an toàn: Nếu x1, y1 nằm ngay bên trong mặt trời (d < r) -> math.asin sẽ báo lỗi
    if d <= sun_obs_r: 
        return direct 
        
    safe_orad = sun_obs_r + 0.1 # Cộng thêm 0.1 margin để không bị sượt sát sàn sạt
    half = math.asin(min(1.0, safe_orad / d))
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    
    # 2 góc tiếp tuyến để "ôm cua" Mặt Trời (Cua trái và Cua phải)
    cand1 = to_sun + half
    cand2 = to_sun - half
    
    # 4. Chọn góc rẽ ngắn nhất so với góc trực tiếp ban đầu
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
                pos = get_planet_pos(p.id, t-1, ang_vel, initial_by_id, comet_map, planet_by_id)
                if pos is None: continue
                if path_hits_planet(cur_fx, cur_fy, new_fx, new_fy, pos[0], pos[1], p.radius):
                    hit_p = p; break
            
            if hit_p:
                arrivals[hit_p.id].append((t, f.owner, int(f.ships)))
                break
            cur_fx, cur_fy = new_fx, new_fy
            if not (-50 <= cur_fx <= 150 and -50 <= cur_fy <= 150): break
    return arrivals

def is_path_safe(w, src, angle, ships, tgt_id):
    spd = fleet_speed(ships)
    dx, dy = math.cos(angle) , math.sin(angle)
    cur_x, cur_y = src.x, src.y
    for t in range(1, 150):
        next_x, next_y = cur_x + dx * spd, cur_y + dy * spd
        if seg_min_dist(cur_x, cur_y, next_x, next_y, SUN_X, SUN_Y) < SUN_R + SUN_MARGIN:
            return False
        hit_p = None
        for p in w.planets:
            if p.id == src.id: continue
            pos = w.predict_pos(p.id, t-1)
            if pos is None: continue
            if path_hits_planet(cur_x, cur_y, next_x, next_y, pos[0], pos[1], p.radius):
                hit_p = p; break
        if hit_p: return hit_p.id == tgt_id
        cur_x, cur_y = next_x, next_y
        if not (-20 <= cur_x <= 120 and -20 <= cur_y <= 120): break
    return False


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

        self._shot_cache = {}
        self._planet_list = planets

    def predict_pos(self, pid, t):
        return get_planet_pos(pid, t, self.ang_vel, self.initial_by_id, self.comet_map, self.planet_by_id)

    def aim(self, src, tgt, ships):
        key = (src.id, tgt.id, int(ships))
        if key in self._shot_cache: return self._shot_cache[key]

        spd = fleet_speed(max(1, int(ships)))
        intercept_turn = None
        tx, ty = tgt.x, tgt.y

        # Quét tịnh tiến để tìm chính xác turn đánh chặn khớp với tọa độ (Tuyệt đối không bị trượt do float)
        for t in range(1, 150):
            pos = self.predict_pos(tgt.id, t - 1)
            if pos is None: break # Hết đường (Comet biến mất)
            
            # Nếu cự ly từ điểm bắn tới vị trí tương lai <= quãng đường tàu bay được trong t turns
            if dist(src.x, src.y, pos[0], pos[1]) - tgt.radius <= t * spd:
                intercept_turn = t
                tx, ty = pos[0], pos[1]
                break

        if intercept_turn is None:
            self._shot_cache[key] = None
            return None

        direct_angle = math.atan2(ty - src.y, tx - src.x)
        
        # Chỉ truyền tọa độ, không truyền list hành tinh vào nữa
        safe_ang = safe_angle(src.x, src.y, tx, ty) 
        
        angle_diff = abs((safe_ang - direct_angle + math.pi) % (2 * math.pi) - math.pi)
        target_angular_width = math.atan2(tgt.radius, max(1.0, dist(src.x, src.y, tx, ty)))

        # Siết margin siêu chặt (0.3 cho sao chổi, 0.45 cho hành tinh) để bảo vệ các phát bắn chi viện xa
        margin = 0.3 if tgt.id in self.comet_ids else 0.45
        if angle_diff > target_angular_width * margin:
            self._shot_cache[key] = None
            return None

        res = (safe_ang, intercept_turn, tx, ty)
        self._shot_cache[key] = res
        return res

    def keep_needed(self, pid): return self.timelines[pid]["keep_needed"]
    def holds_full(self, pid): return self.timelines[pid]["holds_full"]
    def incoming_friendly(self, pid): return sum(s for _, o, s in self.arrivals.get(pid, []) if o == self.player)

    def proactive_keep(self, pid):
        p, best = self.planet_by_id[pid], 0
        for en in self.enemy_planets:
            aim_r = self.aim(en, p, max(1, int(en.ships)))
            if aim_r:
                _, eta, _, _ = aim_r
                if eta <= PROACTIVE_HORIZON: best = max(best, int(en.ships * PROACTIVE_RATIO))
        return best

    def total_keep(self, pid): return max(self.keep_needed(pid), self.proactive_keep(pid))
    def attack_budget(self, pid, spent):
        return max(0, int(self.planet_by_id[pid].ships) - spent.get(pid, 0) - self.total_keep(pid))


# Phase detection

def detect_phase(w: WorldModel) -> str:
    pr = w.my_prod / w.enemy_prod if w.enemy_prod > 0 else 999
    sr = w.my_total / w.enemy_total if w.enemy_total > 0 else 999
    if w.is_very_late: return "endgame"
    if w.enemy_planets and (pr >= 3.5 and sr >= 2.5) or w.is_finishing: return "smash"
    under_threat = [p for p in w.my_planets if not w.holds_full(p.id)]
    if under_threat and sr < 1.3: return "defend"
    if w.is_opening and len(w.my_planets) < 4 and w.neutral_planets: return "expand"
    if pr >= 2.0 and sr >= 1.5 and w.enemy_planets: return "aggressive"
    if w.neutral_planets and len(w.my_planets) < 7: return "grow"
    return "attack" if w.enemy_planets else "grow"

ALPHA, BETA, GAMMA, DELTA, MAX_ARRIVAL_TURNS = 1.0, 0.10, 0.70, 8.0, 20

@dataclass
class Mission:
    src_id: int
    tgt_id: int
    angle: float
    turns: int
    cost: int
    score: float


def min_ships_to_own_at(w: WorldModel, tgt_id: int, t_arrival: int, extra_arrivals: list) -> int:
    tgt, base_arrivals = w.planet_by_id[tgt_id], list(w.arrivals.get(tgt_id, []))
    def owns_after(send: int) -> bool:
        arrivals = base_arrivals + extra_arrivals + [(t_arrival, w.player, send)]
        o, g, by_turn = tgt.owner, float(tgt.ships), defaultdict(list)
        for eta, own, s in arrivals:
            eta_i = max(1, int(math.ceil(eta)))
            if eta_i <= t_arrival + 1 and s > 0: by_turn[eta_i].append((eta_i, own, int(s)))
        for t in range(1, t_arrival + 2):
            if o != -1: g += tgt.production
            grp = by_turn.get(t, [])
            if grp: o, g = resolve_combat(o, g, grp)
        return o == w.player and g >= 1
    if owns_after(0): return 0
    lo, hi = 1, 999
    while lo < hi:
        mid = (lo + hi) // 2
        if owns_after(mid): hi = mid
        else: lo = mid + 1
    return lo if owns_after(lo) else 9999

def position_bonus(w: WorldModel, tgt: Planet, phase: str) -> float:
    bonus = tgt.production * 3.0
    if tgt.owner == w.player and not w.holds_full(tgt.id): bonus += 20.0
    if tgt.owner not in (-1, w.player):
        bonus += 10.0
        if w.is_finishing: bonus += 15.0
        if w.owner_ships.get(tgt.owner, 999) < 60: bonus += 10.0
    if dist(tgt.x, tgt.y, SUN_X, SUN_Y) + tgt.radius >= ROTATION_LIMIT: bonus += 4.0
    if phase == "smash" and tgt.owner not in (-1, w.player): bonus *= 1.5
    elif phase == "expand" and tgt.owner == -1: bonus *= 1.3
    elif phase == "aggressive" and tgt.owner not in (-1, w.player): bonus *= 1.4
    friendly_in = w.incoming_friendly(tgt.id)
    if friendly_in > 0: bonus -= friendly_in * 0.05
    return bonus

def build_missions(w: WorldModel, spent: dict, targeted: set, planned_commitments: list, phase: str) -> list[Mission]:
    potential_pool = [p for p in w.planets if not (p.owner == w.player and w.holds_full(p.id))]
    pool = []
    for p in potential_pool:
        if p.id in targeted: continue
        if p.owner != w.player:
             arrivals = w.arrivals.get(p.id, [])
             f_arrivals = [a for a in arrivals if a[1] == w.player]
             if f_arrivals:
                 max_t = max(a[0] for a in f_arrivals)
                 if min_ships_to_own_at(w, p.id, max_t, []) == 0:
                     continue
        pool.append(p)

    missions = []
    for src in w.my_planets:
        budget = w.attack_budget(src.id, spent)
        if budget < 3: continue
        for tgt in pool:
            if tgt.id == src.id: continue
            aim_r = w.aim(src, tgt, budget)
            if not aim_r: continue
            angle, turns, tx, ty = aim_r
            if turns > MAX_ARRIVAL_TURNS: continue
            if w.is_late and turns > w.remaining - 3: continue
            
            extra = [(eta, o, s) for eta, o, tgt_ref, s in planned_commitments if tgt_ref == tgt.id]
            base_cost = min_ships_to_own_at(w, tgt.id, turns, extra)
            if base_cost == 0 or base_cost >= 9999 or base_cost > budget: continue
            
            tl = w.timelines[tgt.id]
            owner_at_arrival = tl["owner_at"].get(turns, tl["owner_at"].get(tl["horizon"], tgt.owner))
            
            if owner_at_arrival not in (-1, w.player):
                cost = budget
            else:
                cost = base_cost
                
            if not is_path_safe(w, src, angle, cost, tgt.id): continue
            
            better_to_wait = False
            curr_speed = fleet_speed(cost)
            approx_dist = turns * curr_speed
            ships_at = w.timelines[src.id]["ships_at"]
            for dt in range(1, min(15, turns)):
                future_bud = ships_at.get(dt, 0) - w.total_keep(src.id) - spent.get(src.id, 0)
                if future_bud > cost:
                    f_turns = math.ceil(approx_dist / fleet_speed(future_bud))
                    if dt + f_turns < turns:
                        better_to_wait = True
                        break
            if better_to_wait:
                continue
            
            life = 999
            if tgt.id in w.comet_ids:
                life = comet_life(tgt.id, w.comet_map)
                if life <= turns: continue
            
            profit = tgt.production * (min(w.remaining, life) - turns)
            if w.is_late: profit += tgt.ships * 0.4
            score = (ALPHA * profit - BETA * cost - GAMMA * turns + DELTA * position_bonus(w, tgt, phase))
            missions.append(Mission(src.id, tgt.id, angle, turns, cost, score))
            
    missions.sort(key=lambda m: -m.score)
    return missions

def knapsack_select(missions: list[Mission], budgets: dict[int, int]) -> list[Mission]:
    rem_bud, chosen_tgts, chosen_srcs, selected = dict(budgets), set(), set(), []
    for m in missions:
        if m.src_id in chosen_srcs or m.tgt_id in chosen_tgts: continue
        if rem_bud.get(m.src_id, 0) < m.cost: continue
        selected.append(m)
        rem_bud[m.src_id] -= m.cost
        chosen_tgts.add(m.tgt_id); chosen_srcs.add(m.src_id)
    return selected

def execute_missions(selected: list[Mission], w: WorldModel, spent: dict, moves: list, targeted: set, planned_commitments: list):
    for m in selected:
        print(f"[MISSION] Turn {w.step}: {m.src_id} -> {m.tgt_id} | angle={m.angle:.3f}, cost={m.cost}, turns={m.turns}")
        moves.append([m.src_id, float(m.angle), int(m.cost)])
        spent[m.src_id] = spent.get(m.src_id, 0) + m.cost
        targeted.add(m.tgt_id)
        planned_commitments.append((m.turns, w.player, m.tgt_id, m.cost))

def phase_comet_evac(w: WorldModel, spent: dict, moves: list, planned_commitments: list):
    for src in w.my_planets:
        if src.id not in w.comet_ids: continue
        life = comet_life(src.id, w.comet_map)
        if 0 < life < COMET_EVAC_TURNS:
            # Rút sạch toàn bộ quân còn lại
            bud = int(src.ships) - spent.get(src.id, 0)
            if bud < 1: continue
            
            # Tìm hành tinh đồng minh gần nhất để lánh nạn
            others = [p for p in w.my_planets if p.id != src.id]
            if not others: continue
            tgt = min(others, key=lambda p: dist(src.x, src.y, p.x, p.y))
            
            aim_r = w.aim(src, tgt, bud)
            if aim_r:
                angle, turns, _, _ = aim_r
                if is_path_safe(w, src, angle, bud, tgt.id):
                    print(f"[EVAC] Turn {w.step}: Comet {src.id} -> {tgt.id} (EVAC) | ships={bud}, life={life}")
                    moves.append([src.id, float(angle), bud])
                    spent[src.id] = spent.get(src.id, 0) + bud
                    planned_commitments.append((turns, w.player, tgt.id, bud))

def phase7_rear_push(w: WorldModel, spent: dict, moves: list, targeted: set, planned_commitments: list):
    if len(w.my_planets) < 2 or not (w.enemy_planets or w.neutral_planets): return
    if len(w.my_planets) < len(w.planets) / 5.0 and w.is_early: return
    frontier = w.enemy_planets if w.enemy_planets else w.neutral_planets
    dist_to_front = {p.id: min(dist(p.x, p.y, t.x, t.y) for t in frontier) for p in w.my_planets}
    
    for src in w.my_planets:
        bud = w.attack_budget(src.id, spent)
        if bud < REAR_MIN_SHIPS: continue
        
        send = int(bud * REAR_SEND_RATIO)
        if send < 1: continue
        
        # Tìm các hành tinh thân thiện gần tiền tuyến hơn src
        candidates = [p for p in w.my_planets if dist_to_front[p.id] < dist_to_front[src.id] - 5]
        if not candidates: continue
        
        relay = min(candidates, key=lambda c: dist(src.x, src.y, c.x, c.y))
        
        aim_r = w.aim(src, relay, send)
        if not aim_r: continue
        angle, turns, _, _ = aim_r
        if turns > REAR_MAX_TURNS or not is_path_safe(w, src, angle, send, relay.id): continue
        
        print(f"[PUSH] Turn {w.step}: {src.id} -> {relay.id} (relay) | angle={angle:.3f}, ships={send}, turns={turns}")
        moves.append([src.id, float(angle), send])
        spent[src.id] = spent.get(src.id, 0) + send
        planned_commitments.append((turns, w.player, relay.id, send))

def finalize(moves, w):
    used, final = defaultdict(int), []
    for src_id, angle, ships in moves:
        p = w.planet_by_id.get(src_id)
        if not p: continue
        send = min(int(ships), int(p.ships) - used[src_id])
        if send >= 1:
            final.append([src_id, float(angle), int(send)])
            used[src_id] += send
    return final

def _get(obs, key, default=None):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)

def build_world(obs):
    player, step = _get(obs, "player", 0), _get(obs, "step", 0) or 0
    raw_p, raw_f = _get(obs, "planets", []) or [], _get(obs, "fleets", []) or []
    ang_vel, raw_init = _get(obs, "angular_velocity", 0.03) or 0.0, _get(obs, "initial_planets", []) or []
    comets, c_ids = _get(obs, "comets", []) or [], _get(obs, "comet_planet_ids", []) or []
    planets, fleets = [Planet(*p) for p in raw_p], [Fleet(*f) for f in raw_f]
    init_map = {Planet(*p).id: Planet(*p) for p in raw_init}
    return WorldModel(player, step, planets, fleets, init_map, ang_vel, comets, c_ids)

def agent(obs, config=None):
    t0 = time.perf_counter()
    w = build_world(obs)
    if not w.my_planets: return []
    phase = detect_phase(w)
    print(f"[AGENT] Turn {w.step} | phase={phase} | my_prod={w.my_prod} | enemy_prod={w.enemy_prod} | ships={w.my_total}/{w.enemy_total}")
    spent, moves, targeted, planned_commitments = {}, [], set(), []
    if (time.perf_counter() - t0) < DEADLINE_SOFT:
        budgets = {p.id: w.attack_budget(p.id, spent) for p in w.my_planets}
        missions = build_missions(w, spent, targeted, planned_commitments, phase)
        execute_missions(knapsack_select(missions, budgets), w, spent, moves, targeted, planned_commitments)
    
    # Di tản sao chổi sắp biến mất
    if (time.perf_counter() - t0) < DEADLINE_SOFT:
        phase_comet_evac(w, spent, moves, planned_commitments)

    if (time.perf_counter() - t0) < DEADLINE_SOFT and not w.is_late and len(w.my_planets) >= 3:
        phase7_rear_push(w, spent, moves, targeted, planned_commitments)
        
    final_moves = finalize(moves, w)
    t_elapsed = time.perf_counter() - t0
    print(f"[TIME] Turn {w.step} took {t_elapsed:.4f}s")
    return final_moves

__all__ = ["agent"]
