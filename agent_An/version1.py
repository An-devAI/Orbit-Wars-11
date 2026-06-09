# Score: 793.8
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
HORIZON           = 45   
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
REAR_MIN_SHIPS   = 15
REAR_SEND_RATIO  = 0.55
REAR_MAX_TURNS   = 40
REAR_DIST_RATIO  = 1.25

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

def path_blocked(x1, y1, x2, y2, planets, src_id, tgt_id):
    if path_hits_sun(x1, y1, x2, y2): return True
    for p in planets:
        if p.id == src_id or p.id == tgt_id: continue
        if path_hits_planet(x1, y1, x2, y2, p.x, p.y, p.radius): return True
    return False

def bypass_angle(x1, y1, ox, oy, ob_radius, direct_angle, margin=0.3):
    d = math.hypot(x1 - ox, y1 - oy)
    safe_r = ob_radius + margin
    if d <= safe_r: return None, None
    half = math.asin(min(1.0, safe_r / d))
    to_obs = math.atan2(oy - y1, ox - x1)
    return to_obs + half, to_obs - half

def safe_angle(x1, y1, x2, y2, planets=None, src_id=-1, tgt_id=-1):
    direct = math.atan2(y2 - y1, x2 - x1)
    obstacles = [(SUN_X, SUN_Y, SUN_R + SUN_MARGIN)]
    if planets:
        for p in planets:
            if p.id == src_id or p.id == tgt_id: continue
            obstacles.append((p.x, p.y, p.radius))

    def is_clear(angle):
        ex = x1 + math.cos(angle) * 200
        ey = y1 + math.sin(angle) * 200
        for ox, oy, orad in obstacles:
            if seg_min_dist(x1, y1, ex, ey, ox, oy) < orad: return False
        return True

    if is_clear(direct): return direct

    sorted_obs = sorted(obstacles, key=lambda o: math.hypot(x1 - o[0], y1 - o[1]))
    candidates = []
    for ox, oy, orad in sorted_obs:
        d = math.hypot(x1 - ox, y1 - oy)
        if d <= orad: continue
        half = math.asin(min(1.0, orad / d))
        to_obs = math.atan2(oy - y1, ox - x1)
        candidates.append(to_obs + half + 0.05)
        candidates.append(to_obs - half - 0.05)

    def angle_diff(a):
        dd = (a - direct) % (2 * math.pi)
        return min(dd, 2 * math.pi - dd)

    candidates.sort(key=angle_diff)
    for cand in candidates:
        if is_clear(cand): return cand
    return direct

def predict_comet_pos(pid, comets, turns):
    for g in comets:
        pids = g.get("planet_ids", [])
        if pid not in pids: continue
        idx = pids.index(pid)
        paths = g.get("paths", [])
        pi = g.get("path_index", 0)
        if idx >= len(paths): return None
        fi = pi + int(turns)
        if 0 <= fi < len(paths[idx]):
            return paths[idx][fi][0], paths[idx][fi][1]
        return None
    return None

def comet_life(pid, comets):
    for g in comets:
        pids = g.get("planet_ids", [])
        if pid not in pids: continue
        idx = pids.index(pid)
        paths = g.get("paths", [])
        pi = g.get("path_index", 0)
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
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    fall_turn = None

    for t in range(1, horizon + 1):
        if owner != -1: garrison += planet.production
        grp = by_turn.get(t, [])
        if grp:
            prev = owner
            owner, garrison = resolve_combat(owner, garrison, grp)
            if prev == player and owner != player and fall_turn is None:
                fall_turn = t
        owner_at[t] = owner
        ships_at[t] = max(0.0, garrison)

    keep_needed = 0
    holds_full = True
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
            holds_full = False
            keep_needed = int(planet.ships)

    return {
        "owner_at": owner_at, "ships_at": ships_at,
        "keep_needed": keep_needed, "fall_turn": fall_turn,
        "holds_full": holds_full, "horizon": horizon,
    }

def build_arrivals(fleets, planets, ang_vel=0.0, initial_by_id=None):
    arrivals = {p.id: [] for p in planets}
    plist = list(planets)
    
    def get_pos_at(p, t):
        if not initial_by_id or p.id not in initial_by_id: return p.x, p.y
        init = initial_by_id[p.id]
        dist_sun = math.hypot(init.x - SUN_X, init.y - SUN_Y)
        if dist_sun + init.radius >= ROTATION_LIMIT: return p.x, p.y
        cur_ang = math.atan2(p.y - SUN_Y, p.x - SUN_X)
        ang_t = cur_ang + ang_vel * t
        return SUN_X + dist_sun * math.cos(ang_t), SUN_Y + dist_sun * math.sin(ang_t)

    for f in fleets:
        spd = fleet_speed(f.ships)
        dx, dy = math.cos(f.angle), math.sin(f.angle)
        
        hit_p, hit_t = None, None
        cur_fx, cur_fy = f.x, f.y
        
        for t in range(1, 120):
            for p in plist:
                px, py = get_pos_at(p, t-1)
                new_fx, new_fy = cur_fx + dx * spd, cur_fy + dy * spd
                if path_hits_planet(cur_fx, cur_fy, new_fx, new_fy, px, py, p.radius):
                    hit_p, hit_t = p, t
                    break
                
                npx, npy = get_pos_at(p, t)
                if path_hits_planet(px, py, npx, npy, new_fx, new_fy, p.radius):
                    hit_p, hit_t = p, t
                    break
            
            if hit_p: break
            cur_fx, cur_fy = new_fx, new_fy
            if not (-50 <= cur_fx <= 150 and -50 <= cur_fy <= 150): break
                
        if hit_p:
            arrivals[hit_p.id].append((hit_t, f.owner, int(f.ships)))
    return arrivals


# WorldModel

class WorldModel:
    def __init__(self, player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids):
        self.player = player
        self.step   = step
        self.planets = planets
        self.fleets  = fleets
        self.initial_by_id = initial_by_id
        self.ang_vel = ang_vel
        self.comets  = comets
        self.comet_ids = set(comet_ids)

        self.planet_by_id    = {p.id: p for p in planets}
        self.my_planets      = [p for p in planets if p.owner == player]
        self.enemy_planets   = [p for p in planets if p.owner not in (-1, player)]
        self.neutral_planets = [p for p in planets if p.owner == -1]

        self.remaining    = max(1, TOTAL_STEPS - step)
        self.is_early     = step < EARLY_TURN
        self.is_opening   = step < OPENING_TURN
        self.is_late      = self.remaining < LATE_REM
        self.is_very_late = self.remaining < VERY_LATE_REM

        self.owner_ships = defaultdict(int)
        self.owner_prod  = defaultdict(int)
        for p in planets:
            if p.owner != -1:
                self.owner_ships[p.owner] += int(p.ships)
                self.owner_prod[p.owner]  += int(p.production)
        for f in fleets:
            self.owner_ships[f.owner] += int(f.ships)

        self.my_total    = self.owner_ships.get(player, 0)
        self.enemy_total = sum(v for k, v in self.owner_ships.items() if k != player)
        self.my_prod     = self.owner_prod.get(player, 0)
        self.enemy_prod  = sum(v for k, v in self.owner_prod.items() if k != player)

        self.arrivals = build_arrivals(fleets, planets, ang_vel, initial_by_id)
        self.timelines = {
            p.id: simulate_timeline(p, self.arrivals[p.id], player, HORIZON)
            for p in planets
        }

        total = self.my_total + self.enemy_total
        self.domination = (self.my_total - self.enemy_total) / max(1, total)
        self.is_finishing = (self.domination > 0.33 and self.my_prod > self.enemy_prod * 1.2 and step > 100)

        self._shot_cache = {}
        self._planet_list = planets

    def _is_static(self, planet):
        init = self.initial_by_id.get(planet.id)
        if init is None: return True
        return dist(init.x, init.y, SUN_X, SUN_Y) + init.radius >= ROTATION_LIMIT

    def predict_pos(self, pid, turns):
        if pid in self.comet_ids:
            pos = predict_comet_pos(pid, self.comets, turns)
            if pos: return pos
        p = self.planet_by_id[pid]
        if self._is_static(p): return p.x, p.y
        cur = math.atan2(p.y - SUN_Y, p.x - SUN_X)
        r = dist(p.x, p.y, SUN_X, SUN_Y)
        ang = cur + self.ang_vel * turns
        return SUN_X + r*math.cos(ang), SUN_Y + r*math.sin(ang)

    def aim(self, src, tgt, ships):
        key = (src.id, tgt.id, int(ships))
        if key in self._shot_cache: return self._shot_cache[key]

        spd = fleet_speed(max(1, int(ships)))
        is_comet = tgt.id in self.comet_ids
        tx, ty = tgt.x, tgt.y
        intercept_turn = None
        
        if is_comet:
            for t in range(1, 100):
                pos = self.predict_pos(tgt.id, t)
                if pos is None: break
                ctx, cty = pos
                if dist(src.x, src.y, ctx, cty) / spd <= t:
                    intercept_turn = t
                    tx, ty = ctx, cty
                    break
        else:
            for _ in range(AIM_ITERATIONS):
                d = dist(src.x, src.y, tx, ty)
                t = max(1, int(math.ceil(d / spd)))
                pos = self.predict_pos(tgt.id, t)
                if pos is None: break
                ntx, nty = pos
                if dist(ntx, nty, tx, ty) < 0.05:
                    intercept_turn = t
                    break
                tx, ty = ntx, nty
            else:
                intercept_turn = max(1, int(math.ceil(dist(src.x, src.y, tx, ty) / spd)))

        if intercept_turn is None:
            self._shot_cache[key] = None
            return None

        direct_angle = math.atan2(ty - src.y, tx - src.x)
        safe_ang = safe_angle(src.x, src.y, tx, ty, planets=self._planet_list, src_id=src.id, tgt_id=tgt.id)
        
        angle_diff = abs((safe_ang - direct_angle + math.pi) % (2 * math.pi) - math.pi)
        target_angular_width = math.atan2(tgt.radius, max(1.0, dist(src.x, src.y, tx, ty)))
        
        margin = 0.6 if is_comet else 0.85
        if angle_diff > target_angular_width * margin:
            self._shot_cache[key] = None
            return None

        res = (safe_ang, intercept_turn, tx, ty)
        self._shot_cache[key] = res
        return res

    def keep_needed(self, pid): return self.timelines[pid]["keep_needed"]
    def fall_turn(self, pid): return self.timelines[pid]["fall_turn"]
    def holds_full(self, pid): return self.timelines[pid]["holds_full"]
    def incoming_friendly(self, pid): return sum(s for _, o, s in self.arrivals.get(pid, []) if o == self.player)

    def proactive_keep(self, pid):
        p = self.planet_by_id[pid]
        best = 0
        for en in self.enemy_planets:
            aim_r = self.aim(en, p, max(1, int(en.ships)))
            if aim_r is None: continue
            _, eta, _, _ = aim_r
            if eta > PROACTIVE_HORIZON: continue
            best = max(best, int(en.ships * PROACTIVE_RATIO))
        return best

    def total_keep(self, pid):
        return max(self.keep_needed(pid), self.proactive_keep(pid))

    def attack_budget(self, pid, spent):
        available = int(self.planet_by_id[pid].ships) - spent.get(pid, 0)
        reserve = self.total_keep(pid)
        return max(0, available - reserve)


# Phase detection

def detect_phase(w: WorldModel) -> str:
    pr = w.my_prod / w.enemy_prod if w.enemy_prod > 0 else 999
    sr = w.my_total / w.enemy_total if w.enemy_total > 0 else 999
    mc = len(w.my_planets)
    ec = len(w.enemy_planets)

    if w.is_very_late: return "endgame"
    if ec > 0 and (pr >= 3.5 and sr >= 2.5) or w.is_finishing: return "smash"

    under_threat = [p for p in w.my_planets if not w.holds_full(p.id)]
    if under_threat and sr < 1.3: return "defend"

    if w.is_opening and mc < 4 and w.neutral_planets: return "expand"
    if pr >= 2.0 and sr >= 1.5 and ec > 0: return "aggressive"
    if w.neutral_planets and mc < 7: return "grow"
    if ec > 0: return "attack"
    return "grow"

# Hằng số scoring: α·Lợi nhuận - β·Cost - γ·Thời gian + δ·Vị trí
ALPHA = 1.0   
BETA  = 0.10  
GAMMA = 0.70  
DELTA = 8.0   
MAX_ARRIVAL_TURNS = 20   

@dataclass
class Mission:
    src_id:  int
    tgt_id:  int
    angle:   float
    turns:   int    
    cost:    int    
    score:   float


def min_ships_to_own_at(w: WorldModel, tgt_id: int, t_arrival: int, extra_arrivals: list) -> int:
    """Binary search tìm số tàu nhỏ nhất cần gửi để giữ/chiếm mục tiêu."""
    tgt = w.planet_by_id[tgt_id]
    base_arrivals = list(w.arrivals.get(tgt_id, []))

    def owns_after(send: int) -> bool:
        arrivals = base_arrivals + extra_arrivals + [(t_arrival, w.player, send)]
        o, g = tgt.owner, float(tgt.ships)
        by_turn = defaultdict(list)
        for eta, own, s in arrivals:
            eta_i = max(1, int(math.ceil(eta)))
            if eta_i <= t_arrival + 1 and s > 0:
                by_turn[eta_i].append((eta_i, own, int(s)))

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

    # Nếu hành tinh ta đang bị đe dọa -> Ưu tiên cao nhất để phòng thủ cứu viện
    if tgt.owner == w.player and not w.holds_full(tgt.id):
        bonus += 20.0

    if tgt.owner not in (-1, w.player):
        bonus += 10.0
        if w.is_finishing: bonus += 15.0
        if w.owner_ships.get(tgt.owner, 999) < 60: bonus += 10.0 

    if w._is_static(tgt): bonus += 4.0

    if phase == "smash" and tgt.owner not in (-1, w.player): bonus *= 1.5
    elif phase == "expand" and tgt.owner == -1: bonus *= 1.3
    elif phase == "aggressive" and tgt.owner not in (-1, w.player): bonus *= 1.4

    friendly_in = w.incoming_friendly(tgt.id)
    if friendly_in > 0: bonus -= friendly_in * 0.05

    return bonus


def build_missions(w: WorldModel, spent: dict, targeted: set, planned_commitments: list, phase: str) -> list[Mission]:
    # Gộp tất cả mục tiêu: Địch, Trung lập, và Hành tinh ta đang gặp nguy hiểm (để phòng thủ)
    pool = []
    for p in w.planets:
        if p.id in w.comet_ids: continue
        # Bỏ qua những hành tinh của ta đang an toàn tuyệt đối
        if p.owner == w.player and w.holds_full(p.id): continue 
        pool.append(p)

    missions = []

    for src in w.my_planets:
        budget = w.attack_budget(src.id, spent)
        if budget < 3: continue

        for tgt in pool:
            if tgt.id == src.id or tgt.id in targeted: continue

            aim_r = w.aim(src, tgt, budget)
            if aim_r is None: continue
            angle, turns, tx, ty = aim_r

            if turns > MAX_ARRIVAL_TURNS: continue
            if w.is_late and turns > w.remaining - 3: continue

            extra = [(eta, o, s) for eta, o, tgt_ref, s in planned_commitments if tgt_ref == tgt.id]

            cost = min_ships_to_own_at(w, tgt.id, turns, extra)
            if cost == 0 or cost >= 9999 or cost > budget: continue

            profit = tgt.production * max(1, w.remaining - turns) 
            if w.is_late: profit += tgt.ships * 0.4  
            
            pos_bonus = position_bonus(w, tgt, phase)

            score = (ALPHA * profit - BETA * cost - GAMMA * turns + DELTA * pos_bonus)

            missions.append(Mission(
                src_id=src.id, tgt_id=tgt.id,
                angle=angle, turns=turns,
                cost=cost, score=score,
            ))

    missions.sort(key=lambda m: -m.score)
    return missions


def knapsack_select(missions: list[Mission], budgets: dict[int, int]) -> list[Mission]:
    remaining_budget = dict(budgets)
    chosen_tgts, chosen_srcs, selected = set(), set(), []

    for m in missions:
        if m.src_id in chosen_srcs or m.tgt_id in chosen_tgts: continue
        if remaining_budget.get(m.src_id, 0) < m.cost: continue

        selected.append(m)
        remaining_budget[m.src_id] -= m.cost
        chosen_tgts.add(m.tgt_id)
        chosen_srcs.add(m.src_id)

    return selected

def execute_missions(selected: list[Mission], w: WorldModel, spent: dict, moves: list, targeted: set, planned_commitments: list):
    for m in selected:
        bud = w.attack_budget(m.src_id, spent)
        if m.cost > bud or m.cost < 1: continue

        moves.append([m.src_id, float(m.angle), int(m.cost)])
        spent[m.src_id] = spent.get(m.src_id, 0) + m.cost
        targeted.add(m.tgt_id)
        planned_commitments.append((m.turns, w.player, m.tgt_id, m.cost))

def commit(moves, spent, src_id, angle, send, tgt_id=None, turns=None, targeted=None, planned_commitments=None, player=0):
    send = max(1, int(send))
    if send < 1: return
    moves.append([src_id, float(angle), send])
    spent[src_id] = spent.get(src_id, 0) + send
    if targeted is not None and tgt_id is not None:
        targeted.add(tgt_id)
    if planned_commitments is not None and tgt_id is not None and turns is not None:
        planned_commitments.append((turns, player, tgt_id, send))

def phase7_rear_push(w: WorldModel, spent: dict, moves: list, targeted: set, planned_commitments: list):
    """Gom tàu từ hậu phương an toàn ra tiền tuyến."""
    if len(w.my_planets) < 2: return
    if not (w.enemy_planets or w.neutral_planets): return

    frontier = w.enemy_planets if w.enemy_planets else w.neutral_planets
    dist_to_front = {p.id: min(dist(p.x, p.y, t.x, t.y) for t in frontier) for p in w.my_planets}
    sorted_p = sorted(w.my_planets, key=lambda p: dist_to_front[p.id])
    front = sorted_p[0]

    for rear in sorted_p[2:]:
        if dist_to_front[rear.id] < dist_to_front[front.id] * REAR_DIST_RATIO: continue
        bud = w.attack_budget(rear.id, spent) - 5
        if bud < REAR_MIN_SHIPS: continue
        send = int(bud * REAR_SEND_RATIO)
        if send < REAR_MIN_SHIPS: continue
        
        a = w.aim(rear, front, send)
        if a is None: continue
        angle, turns, _, _ = a
        if turns > REAR_MAX_TURNS: continue
        commit(moves, spent, rear.id, angle, send, front.id, turns, targeted, planned_commitments, w.player)


def finalize(moves, w):
    used = defaultdict(int)
    final = []
    for src_id, angle, ships in moves:
        p = w.planet_by_id.get(src_id)
        if p is None: continue
        allowed = int(p.ships) - used[src_id]
        send = min(int(ships), allowed)
        if send >= 1:
            final.append([src_id, float(angle), int(send)])
            used[src_id] += send
    return final

# Main agent

def _get(obs, key, default=None):
    if isinstance(obs, dict): return obs.get(key, default)
    return getattr(obs, key, default)

def build_world(obs):
    player   = _get(obs, "player", 0)
    step     = _get(obs, "step", 0) or 0
    raw_p    = _get(obs, "planets", []) or []
    raw_f    = _get(obs, "fleets", []) or []
    ang_vel  = _get(obs, "angular_velocity", 0.03) or 0.0
    raw_init = _get(obs, "initial_planets", []) or []
    comets   = _get(obs, "comets", []) or []
    c_ids    = _get(obs, "comet_planet_ids", []) or []

    planets  = [Planet(*p) for p in raw_p]
    fleets   = [Fleet(*f) for f in raw_f]
    init_map = {Planet(*p).id: Planet(*p) for p in raw_init}

    return WorldModel(player, step, planets, fleets, init_map, ang_vel, comets, c_ids)


def agent(obs, config=None):
    t0 = time.perf_counter()
    def expired():
        return (time.perf_counter() - t0) >= DEADLINE_SOFT

    w = build_world(obs)
    if not w.my_planets:
        return []

    phase    = detect_phase(w)
    spent    = {}    
    moves    = []
    targeted = set() 
    planned_commitments = []  

    # 1. Mission planning
    if not expired():
        budgets = {p.id: w.attack_budget(p.id, spent) for p in w.my_planets}
        missions = build_missions(w, spent, targeted, planned_commitments, phase)
        selected = knapsack_select(missions, budgets)
        execute_missions(selected, w, spent, moves, targeted, planned_commitments)

    # 2. Rear push 
    if not expired() and not w.is_late and len(w.my_planets) >= 3:
        phase7_rear_push(w, spent, moves, targeted, planned_commitments)

    return finalize(moves, w)

__all__ = ["agent"]