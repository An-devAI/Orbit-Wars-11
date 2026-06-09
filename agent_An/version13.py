# Score: 938.2
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
NEUTRAL_MARGIN       = 3
ENEMY_MARGIN         = 4
PROACTIVE_RATIO      = 0.22

# Mission weighting params
ALPHA, BETA, GAMMA, DELTA, EPSILON, MAX_ARRIVAL_TURNS = 1.0, 0.10, 0.70, 8.0, 2.0, 20
MIN_ATTACK_SHIPS = 3

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
            # Dự đoán vị trí của hành tinh p tại thời điểm tàu bay đến quãng đường này
            pos = w.predict_pos(p.id, start_time + t - 1)
            if pos is None: continue
            if path_hits_planet(cur_x, cur_y, next_x, next_y, pos[0], pos[1], p.radius):
                hit_p = p; break
        
        if hit_p: 
            return hit_p.id == tgt_id
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

        # Pre-cache positions for turn 20 to speed up CoM and scoring
        self._pos_cache_20 = {}
        for p in planets:
            self._pos_cache_20[p.id] = get_planet_pos(p.id, 20, ang_vel, initial_by_id, self.comet_map, self.planet_by_id)

        # Tính toán Trọng tâm lãnh thổ (Center of Mass) hiện tại và 20 turn tới
        if self.my_planets:
            sum_x, sum_y, total_w = 0.0, 0.0, 0.0
            sum_xf, sum_yf = 0.0, 0.0
            for p in self.my_planets:
                w_p = max(1.0, p.production)
                sum_x += p.x * w_p
                sum_y += p.y * w_p
                total_w += w_p
                
                pf = self._pos_cache_20.get(p.id)
                if pf:
                    sum_xf += pf[0] * w_p
                    sum_yf += pf[1] * w_p
                else:
                    sum_xf += p.x * w_p
                    sum_yf += p.y * w_p
            
            self.my_com_now = (sum_x / total_w, sum_y / total_w)
            self.my_com_f20 = (sum_xf / total_w, sum_yf / total_w)
        else:
            self.my_com_now = (SUN_X, SUN_Y)
            self.my_com_f20 = (SUN_X, SUN_Y)

        self._shot_cache = {}
        self._planet_list = planets

    def predict_pos(self, pid, t):
        if t == 0: 
            p = self.planet_by_id.get(pid)
            return (p.x, p.y) if p else None
        if t == 20 and hasattr(self, "_pos_cache_20"):
            return self._pos_cache_20.get(pid)
        return get_planet_pos(pid, t, self.ang_vel, self.initial_by_id, self.comet_map, self.planet_by_id)

    def aim(self, src, tgt, ships):
        key = (src.id, tgt.id, int(ships))
        if key in self._shot_cache: return self._shot_cache[key]

        spd = fleet_speed(max(1, int(ships)))
        intercept_turn = None
        tx, ty = tgt.x, tgt.y

        # TĂNG ĐỘ PHÂN GIẢI: Quét bước 0.5 thay vì 1.0 để không bỏ lỡ hành tinh nhỏ
        for t_float in [x * 0.5 for x in range(2, 300)]:
            t_int = int(t_float)
            # Dự đoán vị trí nội suy giữa 2 turn
            pos = self.predict_pos(tgt.id, t_float)
            if pos is None: break 
            
            d = dist(src.x, src.y, pos[0], pos[1])
            
            # Điều kiện chạm: Khoảng cách bay được >= Khoảng cách tới mép hành tinh
            if d - tgt.radius <= t_float * spd:
                intercept_turn = t_float
                tx, ty = pos[0], pos[1]
                break

        if intercept_turn is None:
            self._shot_cache[key] = None
            return None

        # Tính góc bắn thẳng vào TÂM hành tinh tại thời điểm va chạm
        direct_angle = math.atan2(ty - src.y, tx - src.x)
        
        # Né mặt trời
        safe_ang = safe_angle(src.x, src.y, tx, ty) 
        
        # Kiểm tra góc né có còn chạm hành tinh không
        angle_diff = abs((safe_ang - direct_angle + math.pi) % (2 * math.pi) - math.pi)
        dist_to_tgt = dist(src.x, src.y, tx, ty)
        target_angular_width = math.atan2(tgt.radius, max(1.0, dist_to_tgt))

        # Margin an toàn: Cho phép lệch tối đa 80% bán kính (nới lỏng cho hành tinh nhỏ)
        if angle_diff > target_angular_width * 0.8:
            self._shot_cache[key] = None
            return None

        # KIỂM TRA CHỐT: Không dùng 0.95 nữa, dùng toàn bộ bán kính + 0.1 margin nhỏ
        # Mô phỏng tia bắn thực tế kéo dài
        ex = src.x + math.cos(safe_ang) * 1000
        ey = src.y + math.sin(safe_ang) * 1000
        if seg_min_dist(src.x, src.y, ex, ey, tx, ty) > tgt.radius + 0.1:
            self._shot_cache[key] = None
            return None

        res = (safe_ang, int(math.ceil(intercept_turn)), tx, ty)
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

    def total_keep(self, pid): 
        # Phòng thủ đa lớp: 
        # 1. Dự đoán theo timeline (ngắn hạn/trung hạn)
        # 2. Dự đoán chủ động (đề phòng địch đánh bất ngờ từ các hành tinh lân cận)
        # 3. Buffer an toàn tỷ lệ với Production để chống lại sự bào mòn
        base_keep = self.keep_needed(pid)
        proactive = self.proactive_keep(pid)
        safety_buffer = 2 + int(self.planet_by_id[pid].production * 0.5)
        
        return max(base_keep + safety_buffer, proactive)
    def attack_budget(self, pid, spent):
        return max(0, int(self.planet_by_id[pid].ships) - spent.get(pid, 0) - self.total_keep(pid))


# Phase detection

def detect_phase(w: WorldModel) -> str:
    return "expand" if w.step < 55 else "combat"



@dataclass
class Mission:
    src_id: int
    tgt_id: int
    angle: float
    turns: int
    cost: int
    score: float
    mtype: str = "ATTACK"
    wait_turns: int = 0
    debug_info: dict = field(default_factory=dict)


def min_ships_to_own_at(w: WorldModel, tgt_id: int, t_arrival: int, extra_arrivals: list, sim_until: int = None) -> int:
    tgt, base_arrivals = w.planet_by_id[tgt_id], list(w.arrivals.get(tgt_id, []))
    if sim_until is None: sim_until = t_arrival + 1
    
    # Thêm margin để tránh việc chiếm hụt do sai số hoặc địch phái quân phút chót
    margin = 0
    if tgt.owner == -1: margin = NEUTRAL_MARGIN
    elif tgt.owner != w.player: margin = ENEMY_MARGIN

    def owns_after(send: int) -> bool:
        # Nếu send = 0, kiểm tra xem hiện tại/tương lai có tự thắng không
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
    # Ưu tiên cực cao cho hành tinh có sản lượng lớn
    bonus = (tgt.production ** 1.5) * 5.0 
    
    if tgt.owner not in (-1, w.player):
        bonus += 10.0
        if w.is_finishing: bonus += 15.0
        if w.owner_ships.get(tgt.owner, 999) < 60: bonus += 10.0
    
    if dist(tgt.x, tgt.y, SUN_X, SUN_Y) + tgt.radius >= ROTATION_LIMIT: bonus += 4.0
    
    if phase == "expand":
        if tgt.owner == -1: bonus *= 2.0
    else: # combat
        if tgt.owner not in (-1, w.player): bonus *= 1.5
        
    friendly_in = w.incoming_friendly(tgt.id)
    if friendly_in > 0: bonus -= friendly_in * 0.05
    
    # Bonus/Penalty dựa trên việc hành tinh đang tiến gần hay lùi xa lãnh thổ (CoM)
    pos_f = w.predict_pos(tgt.id, 20)
    if pos_f:
        d_now = dist(tgt.x, tgt.y, w.my_com_now[0], w.my_com_now[1])
        d_f20 = dist(pos_f[0], pos_f[1], w.my_com_f20[0], w.my_com_f20[1])
        bonus += (d_now - d_f20) * 0.7
        
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

    allow_rear_push = True
    if len(w.my_planets) < 2 or not (w.enemy_planets or w.neutral_planets): allow_rear_push = False
    if len(w.my_planets) < len(w.planets) / 5.0 and w.is_early: allow_rear_push = False
    if w.is_late: allow_rear_push = False
    
    # Frontier logic: Tính toán giá trị chiến lược (Profit tiềm năng) của từng hành tinh 
    # Để các phi vụ Relay có thể được chấm điểm dựa trên lợi ích thực tế mà nó mang lại.
    frontier = w.enemy_planets if w.enemy_planets else w.neutral_planets
    
    # Tính Strategic Forwardness Score:
    # 1. Đo khoảng cách tới tiền tuyến
    dist_to_front = {p.id: min(dist(p.x, p.y, t.x, t.y) for t in frontier) for p in w.my_planets} if frontier else {}
    
    # 2. Score = (100 - Khoảng cách) * 10 + Sản lượng
    # Hành tinh càng gần địch (Khoảng cách nhỏ) -> Score càng cao.
    # Sản lượng là yếu tố phụ (tie-breaker) để ưu tiên các "trạm trung chuyển lớn".
    f_scores = {}
    for p in w.my_planets:
        d_front = dist_to_front.get(p.id, 100)
        f_scores[p.id] = (100.0 - d_front) * 10.0 + p.production
    
    # Bản đồ giá trị Profit tối đa có thể đạt được từ mỗi hành tinh đồng minh
    # Dùng để chấm điểm cho Relay missions.
    max_profit_from = defaultdict(float)
    
    # 0. Kiểm tra tình trạng phòng thủ tổng thể để ưu tiên
    vulnerable_friendly = []
    for p in w.my_planets:
        tl = w.timelines[p.id]
        if tl.get("fall_turn") is not None and tl["fall_turn"] <= MAX_ARRIVAL_TURNS:
            vulnerable_friendly.append(p.id)

    # 1. Thu thập dữ liệu nhiệm vụ tấn công/chiếm đóng trước
    raw_missions = []
    for src in w.my_planets:
        src_tl = w.timelines[src.id]
        keep = w.total_keep(src.id)
        
        for tgt in pool:
            if tgt.id == src.id: continue
            
            # Lần 1: Dự đoán sơ bộ
            aim_now = w.aim(src, tgt, max(1, int(src.ships))) 
            if not aim_now: continue
            angle, travel_turns, tx, ty = aim_now
            
            if travel_turns > MAX_ARRIVAL_TURNS: continue
            
            extra = [(eta, o, s) for eta, o, tgt_ref, s in planned_commitments if tgt_ref == tgt.id]
            tl = w.timelines[tgt.id]
            
            req_cost = min_ships_to_own_at(w, tgt.id, travel_turns, extra)
            if req_cost == 0 or req_cost >= 9999: continue
            
            # Tính thời gian chờ (wait_turns) lần 1
            wait_turns = 0
            needed_total = req_cost + keep
            ships_at_src = src_tl["ships_at"]
            if ships_at_src[0] < needed_total:
                for dt in range(1, 21):
                    if ships_at_src.get(dt, 0) >= needed_total:
                        wait_turns = dt; break
            
            if wait_turns > 15: continue
            total_turns = travel_turns + wait_turns

            # Lần 2: Nếu có chờ, tinh chỉnh lại Turn và Cost cho cực kỳ chính xác
            if wait_turns > 0:
                future_src_pos = w.predict_pos(src.id, wait_turns)
                if not future_src_pos: continue
                src_f = Planet(src.id, src.owner, future_src_pos[0], future_src_pos[1], src.radius, req_cost, src.production)
                aim_f = w.aim(src_f, tgt, req_cost)
                if not aim_f: continue
                angle, travel_turns, tx, ty = aim_f
                total_turns = travel_turns + wait_turns
                # Cập nhật lại cost cho ngày đến thực tế
                req_cost = min_ships_to_own_at(w, tgt.id, total_turns, extra)
                if req_cost == 0 or req_cost >= 9999: continue

            if total_turns > MAX_ARRIVAL_TURNS + 5: continue
            if w.is_late and total_turns > w.remaining - 2: continue
            
            if not is_path_safe(w, src.id, src.x, src.y, angle, req_cost, tgt.id, wait_turns): continue

            mtype = "ATTACK"
            if tgt.owner == w.player: mtype = "DEFEND"
            elif tgt.owner == -1: mtype = "CAPTURE"

            # Đảm bảo quân đi phải có ý nghĩa (floor)
            if mtype in ("ATTACK", "CAPTURE"):
                req_cost = max(req_cost, MIN_ATTACK_SHIPS)

            owner_at_arrival = tl["owner_at"].get(total_turns, tl["owner_at"].get(tl["horizon"], tgt.owner))
            ships_at_arrival = tl["ships_at"].get(total_turns, tl["ships_at"].get(tl["horizon"], tgt.ships))
            
            profit = tgt.production * (min(w.remaining, 999) - total_turns)
            if tgt.id in w.comet_ids:
                life = comet_life(tgt.id, w.comet_map)
                if life <= total_turns: continue
                profit = tgt.production * (min(w.remaining, life) - total_turns)
                if profit <= req_cost: continue

            # Surplus check
            surplus = ships_at_arrival - req_cost
            if mtype == "ATTACK" and surplus < -req_cost * 0.5 and profit < req_cost * 2:
                continue

            bonus = position_bonus(w, tgt, phase)

            # ƯU TIÊN PHÒNG THỦ: Nếu có hành tinh đồng minh sắp mất, phạt nặng việc đi chiếm mỏ trung lập
            if vulnerable_friendly:
                if mtype == "CAPTURE":
                    bonus -= 100.0
                elif mtype == "DEFEND" and tgt.id in vulnerable_friendly:
                    # Thưởng cực lớn cho việc cứu viện, tỷ lệ thuận với Production
                    # Hành tinh càng to thì càng được ưu tiên cứu trước
                    bonus += 200.0 + (tgt.production * 50.0)

            # Threat Calculation:
            # 1. Active Threat (Từ hạm đội địch đang bay tới đích)
            active_threat = 0.0
            if mtype != "DEFEND":
                for arr_eta, arr_o, arr_s in w.arrivals.get(tgt.id, []):
                    if arr_o not in (-1, w.player):
                        active_threat += arr_s * (total_turns / max(total_turns, float(arr_eta)))**2
            
            # 2. Potential Threat (Từ các hành tinh địch có thể phản ứng)
            potential_threat = 0.0
            if mtype != "DEFEND":
                for en in w.enemy_planets:
                    eta_en = math.ceil(dist(en.x, en.y, tgt.x, tgt.y) / fleet_speed(en.ships))
                    potential_threat += en.ships * (total_turns / max(total_turns, float(eta_en)))**2
            
            # 3. Total Threat normalized by our total strength
            threat_total = min(1.5, (potential_threat + active_threat) / max(1.0, float(w.my_total)))

            # Phạt nặng nếu tranh chấp mỏ trung lập (hành tinh trống) mà threat cao
            if mtype == "CAPTURE":
                if threat_total > 0.3:
                    # Phạt lũy tiến cực nặng theo threat: 0.3 -> 80, 0.5 -> 200+
                    bonus -= (threat_total * 20) ** 2
                
                # Check độ bền vững: Nếu chiếm xong mà bị cướp lại ngay trong 10 turn tới
                # (Dựa trên timeline giả lập dứt điểm tại total_turns)
                cost_to_hold = min_ships_to_own_at(w, tgt.id, total_turns, extra, sim_until=total_turns + 10)
                if cost_to_hold > req_cost * 1.5:
                    # Nếu chi phí để giữ hành tinh sau 10 turn cao hơn 1.5 lần chi phí chiếm
                    # chứng tỏ có hạm đội địch cực lớn đang bay tới ngay sau mình.
                    bonus -= 150.0

            # Phạt thêm dựa trên khoảng cách nếu Threat cao (Tránh overextend)
            if threat_total > 0.4:
                # Phạt lũy tiến theo bình phương quãng đường bay
                bonus -= (travel_turns / 10.0) ** 2 * 10.0
            # Logic 1-Hop Relay: Nếu phải chờ, check xem đi "đường vòng" qua mỏ đồng minh có nhanh hơn không
            if wait_turns > 0:
                best_hop = None
                t_direct = total_turns # Đã tính ở Lần 2 (wait + travel)
                
                # Tìm các trạm trung chuyển (relay)
                for relay in w.my_planets:
                    if relay.id == src.id or relay.id == tgt.id: continue
                    # Điều kiện 1: Relay phải "tiến gần" về phía Tgt hơn và có Forwardness Score lớn hơn nghiêm ngặt
                    if dist(relay.x, relay.y, tgt.x, tgt.y) >= dist(src.x, src.y, tgt.x, tgt.y): continue
                    if f_scores.get(relay.id, 0) <= f_scores.get(src.id, 0): continue
                    
                    # Chặng 1: Src -> Relay (Đi ngay lập tức)
                    budget_now = int(src.ships) - spent.get(src.id, 0) - keep
                    if budget_now < MIN_ATTACK_SHIPS: continue
                    
                    aim1 = w.aim(src, relay, budget_now)
                    if not aim1: continue
                    ang1, t1, _, _ = aim1
                    if not is_path_safe(w, src.id, src.x, src.y, ang1, budget_now, relay.id): continue
                    
                    # Chặng 2: Relay -> Tgt (Xuất phát khi quân của Src vừa tới Relay)
                    # Tính lượng quân tại Relay lúc t1
                    tl_relay = w.timelines[relay.id]
                    ships_at_relay_t1 = tl_relay["ships_at"].get(t1, relay.ships) + budget_now
                    
                    # Giả lập phát bắn từ Relay (vị trí tương lai) đến Tgt
                    pos_relay_t1 = w.predict_pos(relay.id, t1)
                    if not pos_relay_t1: continue
                    relay_f = Planet(relay.id, relay.owner, pos_relay_t1[0], pos_relay_t1[1], relay.radius, ships_at_relay_t1, relay.production)
                    
                    aim2 = w.aim(relay_f, tgt, ships_at_relay_t1)
                    if not aim2: continue
                    ang2, t2, _, _ = aim2
                    
                    t_relay_total = t1 + t2
                    if t_relay_total < t_direct:
                        # Check xem quân tại Relay lúc đó có đủ dứt điểm Tgt không
                        # (req_cost_relay là số quân cần từ Relay để chiếm Tgt tại thời điểm t1+t2)
                        req_cost_relay = min_ships_to_own_at(w, tgt.id, t_relay_total, extra)
                        if ships_at_relay_t1 >= req_cost_relay + w.total_keep(relay.id):
                            if best_hop is None or t_relay_total < best_hop[0]:
                                best_hop = (t_relay_total, relay, ang1, budget_now, t1)
                
                if best_hop:
                    t_relay_total, r_planet, r_ang, r_send, r_t1 = best_hop
                    # ĐỔI NHIỆM VỤ: Src -> Relay ngay lập tức, chấm điểm bằng Profit của Tgt
                    # Ghi đè các thông số để ném vào raw_missions
                    raw_missions.append((src.id, r_planet.id, r_ang, r_t1, r_send, profit, position_bonus(w, r_planet, phase), 0.0, 0.0, "RELAY", 0))
                    # Đã có phương án tốt hơn, không cần lưu mission direct (wait_turns) nữa
                    continue

            raw_missions.append((src.id, tgt.id, angle, travel_turns, req_cost, profit, bonus, ships_at_arrival, threat_total, mtype, wait_turns))
            
            # Cập nhật profit tối đa mà src có thể đạt được (Dùng cho Relay phía sau)
            if mtype != "DEFEND":
                max_profit_from[src.id] = max(max_profit_from[src.id], profit)

    # 2. Thu thập dữ liệu nhiệm vụ Relay (chấm điểm bằng profit của đích đến)
    for src in w.my_planets:
        src_tl = w.timelines[src.id]
        keep = w.total_keep(src.id)
        budget = int(src.ships) - spent.get(src.id, 0) - keep

        if allow_rear_push and budget >= REAR_MIN_SHIPS and dist_to_front:
            send = int(budget * REAR_SEND_RATIO)
            # Tìm hành tinh đồng minh ở "phía trước" mình (gần frontier hơn) và có Forwardness Score lớn hơn nghiêm ngặt
            candidates = [p for p in w.my_planets if dist_to_front.get(p.id, 0) < dist_to_front.get(src.id, 0) - 5 and f_scores.get(p.id, 0) > f_scores.get(src.id, 0)]
            if candidates:
                relay = min(candidates, key=lambda c: dist(src.x, src.y, c.x, c.y))
                aim_r = w.aim(src, relay, send)
                if aim_r:
                    angle, turns, tx, ty = aim_r
                    if turns <= REAR_MAX_TURNS and is_path_safe(w, src.id, src.x, src.y, angle, send, relay.id):
                        tl_r = w.timelines[relay.id]
                        ships_at_arrival_r = tl_r["ships_at"].get(turns, tl_r["ships_at"].get(tl_r["horizon"], relay.ships))
                        
                        # ĐIỂM NHẤN: Profit của phi vụ Relay = Profit tốt nhất mà hành tinh đích có thể thực hiện
                        relay_profit = max_profit_from[relay.id]
                        raw_missions.append((src.id, relay.id, angle, turns, send, relay_profit, position_bonus(w, relay, phase), ships_at_arrival_r, 0.0, "RELAY", 0))
            
    if not raw_missions: return []

    def get_stats(arr, baseline=0.0, prior_std=10.0):
        combined = arr + [baseline, baseline - prior_std, baseline + prior_std]
        mean = sum(combined) / len(combined)
        std = math.sqrt(sum((x - mean)**2 for x in combined) / len(combined))
        return mean, std

    # effective_turns = travel_turns + wait_turns * 1.5
    eff_turns_list = [m[3] + m[10] * 1.5 for m in raw_missions]
    mean_p, std_p = get_stats([m[5] for m in raw_missions], 0.0, 20.0)
    mean_t, std_t = get_stats(eff_turns_list, 35.0, 15.0)
    mean_b, std_b = get_stats([m[6] for m in raw_missions], 0.0, 5.0)
    mean_c_diff, std_c_diff = get_stats([(m[7] - m[4]) for m in raw_missions], 0.0, 15.0)
    mean_th, std_th = get_stats([m[8] for m in raw_missions], 0.1, 0.3)

    missions = []
    for src_id, tgt_id, angle, turns, cost, profit, bonus, ships_at_arrival, threat_total, mtype, wait_turns in raw_missions:
        eff_t = turns + wait_turns * 1.5
        norm_p = (profit - mean_p) / std_p
        norm_t = (eff_t - mean_t) / std_t
        norm_b = (bonus - mean_b) / std_b
        norm_c_diff = (abs(ships_at_arrival - cost) - mean_c_diff) / std_c_diff
        norm_th = (threat_total - mean_th) / std_th

        score = (ALPHA * norm_p + BETA * norm_c_diff - GAMMA * norm_t + DELTA * norm_b - EPSILON * norm_th)
        debug = {
            "raw": {"profit": profit, "cost": cost, "turns": turns, "wait": wait_turns, "eff_t": eff_t, "bonus": bonus, "threat": threat_total, "ships_at": ships_at_arrival},
            "norm": {"p": norm_p, "c_diff": norm_c_diff, "t": norm_t, "b": norm_b, "th": norm_th},
            "weights": {"A": ALPHA, "B": BETA, "G": GAMMA, "D": DELTA, "E": EPSILON}
        }
        missions.append(Mission(src_id, tgt_id, angle, turns, cost, score, mtype, wait_turns, debug))

    missions.sort(key=lambda m: -m.score)
    return missions

def knapsack_select(missions: list[Mission], budgets: dict[int, int]) -> list[Mission]:
    rem_bud, chosen_tgts, chosen_srcs, selected = dict(budgets), set(), set(), []
    for m in missions:
        if m.src_id in chosen_srcs or m.tgt_id in chosen_tgts: continue
        
        if m.wait_turns > 0:
            # Nếu nhiệm vụ tốt nhất là phải chờ -> "Đặt chỗ" (Reserve) hành tinh này luôn, 
            # không cho nó thực hiện các nhiệm vụ điểm thấp hơn mà đi ngay.
            logger.info(f"[RESERVE] {m.src_id} for {m.tgt_id} in {m.wait_turns} turns | score={m.score:.3f}")
            chosen_srcs.add(m.src_id)
            chosen_tgts.add(m.tgt_id)
            continue
            
        if rem_bud.get(m.src_id, 0) < m.cost: continue
        
        selected.append(m)
        rem_bud[m.src_id] -= m.cost
        chosen_tgts.add(m.tgt_id); chosen_srcs.add(m.src_id)
    return selected

def execute_missions(selected: list[Mission], w: WorldModel, spent: dict, moves: list, targeted: set, planned_commitments: list):
    for m in selected:
        logger.info(f"[{m.mtype}] Turn {w.step}: {m.src_id} -> {m.tgt_id} | score={m.score:.3f}")
        logger.info(f"  > RAW : profit={m.debug_info['raw']['profit']:.1f}, cost={m.cost}, turns={m.turns}, wait={m.wait_turns}, bonus={m.debug_info['raw']['bonus']:.1f}, threat={m.debug_info['raw']['threat']:.2f}, ships_at={m.debug_info['raw']['ships_at']:.1f}")
        logger.info(f"  > NORM: p={m.debug_info['norm']['p']:.2f}, c_diff={m.debug_info['norm']['c_diff']:.2f}, t={m.debug_info['norm']['t']:.2f}, b={m.debug_info['norm']['b']:.2f}, th={m.debug_info['norm']['th']:.2f}")
        logger.info(f"  > WGHT: A={m.debug_info['weights']['A']}, B={m.debug_info['weights']['B']}, G={m.debug_info['weights']['G']}, D={m.debug_info['weights']['D']}, E={m.debug_info['weights']['E']}")
        
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
                if is_path_safe(w, src.id, src.x, src.y, angle, bud, tgt.id):
                    logger.info(f"[EVAC] Turn {w.step}: Comet {src.id} -> {tgt.id} (EVAC) | ships={bud}, life={life}")
                    moves.append([src.id, float(angle), bud])
                    spent[src.id] = spent.get(src.id, 0) + bud
                    planned_commitments.append((turns, w.player, tgt.id, bud))

def phase_tactical_evac(w: WorldModel, spent: dict, moves: list, planned_commitments: list):
    for src in w.my_planets:
        # Nếu đã phái quân đi làm nhiệm vụ khác rồi thì thôi
        if spent.get(src.id, 0) > 0: continue
        
        # Đếm số kẻ địch khác nhau đang nhắm vào hành tinh này và check xem có ai sắp tới không
        enemies_targeting = set()
        soon_arrival = False
        for eta, o, s in w.arrivals.get(src.id, []):
            if o not in (-1, w.player):
                enemies_targeting.add(o)
                if eta <= 10: soon_arrival = True
        
        tl = w.timelines[src.id]
        # Điều kiện rút quân SIẾT CHẶT: 
        # CHỈ rút khi bị ít nhất 2 kẻ địch khác nhau tấn công (Hội đồng)
        # VÀ ít nhất 1 trong số đó sẽ tới trong vòng 10 turn (Nguy cơ cận kề)
        is_being_ganged = (len(enemies_targeting) >= 2 and soon_arrival)
        
        if is_being_ganged:
            bud = int(src.ships)
            if bud < 1: continue
            
            # Tìm hành tinh đồng minh "an toàn" gần nhất
            others = [p for p in w.my_planets if p.id != src.id]
            if not others: continue
            
            # Ưu tiên các hành tinh đang KHÔNG bị tấn công
            safe_others = [p for p in others if w.timelines[p.id].get("fall_turn") is None]
            target_list = safe_others if safe_others else others
            tgt = min(target_list, key=lambda p: dist(src.x, src.y, p.x, p.y))
            
            aim_r = w.aim(src, tgt, bud)
            if aim_r:
                angle, turns, _, _ = aim_r
                if is_path_safe(w, src.id, src.x, src.y, angle, bud, tgt.id):
                    logger.info(f"[TACTICAL_EVAC] Turn {w.step}: {src.id} -> {tgt.id} | ships={bud}, reason={'GANGED' if is_being_ganged else 'FALLING'}")
                    moves.append([src.id, float(angle), bud])
                    spent[src.id] = spent.get(src.id, 0) + bud
                    planned_commitments.append((turns, w.player, tgt.id, bud))

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

def update_params(step, initial_by_id, planets):
    global ALPHA, BETA, GAMMA, DELTA, EPSILON, MAX_ARRIVAL_TURNS, PROACTIVE_RATIO, REAR_SEND_RATIO, MIN_ATTACK_SHIPS

    total_players = len(set(p.owner for p in initial_by_id.values() if p.owner != -1))
    if total_players == 0:
        total_players = len(set(p.owner for p in planets if p.owner != -1))
        
    is_1v1 = (total_players <= 2)
    early_limit = 55 if is_1v1 else 15
    is_early = (step <= early_limit)

    if globals().get("TUNE_MODE", False):
        if is_early:
            ALPHA = globals().get("EARLY_ALPHA", ALPHA)
            BETA = globals().get("EARLY_BETA", BETA)
            GAMMA = globals().get("EARLY_GAMMA", GAMMA)
            DELTA = globals().get("EARLY_DELTA", DELTA)
            EPSILON = globals().get("EARLY_EPSILON", EPSILON)
            MAX_ARRIVAL_TURNS = globals().get("EARLY_MAX_ARRIVAL_TURNS", MAX_ARRIVAL_TURNS)
            PROACTIVE_RATIO = globals().get("EARLY_PROACTIVE_RATIO", PROACTIVE_RATIO)
            REAR_SEND_RATIO = globals().get("EARLY_REAR_SEND_RATIO", REAR_SEND_RATIO)
            MIN_ATTACK_SHIPS = globals().get("EARLY_MIN_ATTACK_SHIPS", MIN_ATTACK_SHIPS)
        else:
            ALPHA = globals().get("LATE_ALPHA", ALPHA)
            BETA = globals().get("LATE_BETA", BETA)
            GAMMA = globals().get("LATE_GAMMA", GAMMA)
            DELTA = globals().get("LATE_DELTA", DELTA)
            EPSILON = globals().get("LATE_EPSILON", EPSILON)
            MAX_ARRIVAL_TURNS = globals().get("LATE_MAX_ARRIVAL_TURNS", MAX_ARRIVAL_TURNS)
            PROACTIVE_RATIO = globals().get("LATE_PROACTIVE_RATIO", PROACTIVE_RATIO)
            REAR_SEND_RATIO = globals().get("LATE_REAR_SEND_RATIO", REAR_SEND_RATIO)
            MIN_ATTACK_SHIPS = globals().get("LATE_MIN_ATTACK_SHIPS", MIN_ATTACK_SHIPS)
        return

    if is_1v1:
        if is_early:
            # --- SETUP 1: 1vs1, Đầu game ---
            ALPHA             = 1.8634157945851026
            BETA              = 0.7081663340762594
            GAMMA             = 0.8186077136748544
            DELTA             = 2.339245579230789
            EPSILON           = 2.381493448973844
            MAX_ARRIVAL_TURNS = 21
            PROACTIVE_RATIO   = 0.1
            REAR_SEND_RATIO   = 0.7945290798169734
            MIN_ATTACK_SHIPS  = 6
        else:
            # --- SETUP 2: 1vs1, Cuối game ---
            ALPHA             = 3.516556601994333
            BETA              = 0.4200563456732971
            GAMMA             = 1.361963066325989
            DELTA             = 2.0533274161533184
            EPSILON           = 2.2749124814062673
            MAX_ARRIVAL_TURNS = 28
            PROACTIVE_RATIO   = 0.21    
            REAR_SEND_RATIO   = 0.6678874501770248
            MIN_ATTACK_SHIPS  = 5

    else: # CHẾ ĐỘ 1VS3 (FFA) - BỘ THAM SỐ ĐỈNH CAO (TRIAL 21 - VALUE 5084)
        if is_early:
            # --- SETUP 3: 1vs3, Đầu game ---
            ALPHA             = 3.606190089226139
            BETA              = 0.6337511711715067
            GAMMA             = 2.561815554988392
            DELTA             = 3.394687881618263
            EPSILON           = 4.6890256709017155  # Con số kỷ lục!
            MAX_ARRIVAL_TURNS = 26
            PROACTIVE_RATIO   = 0.0
            REAR_SEND_RATIO   = 0.7196435003696126
            MIN_ATTACK_SHIPS  = 9
        else:
            # --- SETUP 4: 1vs3, Giữa/Cuối game ---
            ALPHA             = 3.4158722385107865
            BETA              = 0.7678952314002715
            GAMMA             = 2.258013898449506
            DELTA             = 3.1197874573250473
            EPSILON           = 1.5649124176773959  # Giảm mạnh để tấn công
            MAX_ARRIVAL_TURNS = 34
            PROACTIVE_RATIO   = 0.0
            REAR_SEND_RATIO   = 0.7347637665689817
            MIN_ATTACK_SHIPS  = 8

def build_world(obs):
    player, step = _get(obs, "player", 0), _get(obs, "step", 0) or 0
    raw_p, raw_f = _get(obs, "planets", []) or [], _get(obs, "fleets", []) or []
    ang_vel, raw_init = _get(obs, "angular_velocity", 0.03) or 0.0, _get(obs, "initial_planets", []) or []
    comets, c_ids = _get(obs, "comets", []) or [], _get(obs, "comet_planet_ids", []) or []
    planets, fleets = [Planet(*p) for p in raw_p], [Fleet(*f) for f in raw_f]
    init_map = {Planet(*p).id: Planet(*p) for p in raw_init}
    
    update_params(step, init_map, planets)
    
    return WorldModel(player, step, planets, fleets, init_map, ang_vel, comets, c_ids)

def agent(obs, config=None):
    t0 = time.perf_counter()
    w = build_world(obs)
    if not w.my_planets: return []
    phase = detect_phase(w)
    logger.info(f"--- Turn {w.step} | phase={phase} | ships={w.my_total}/{w.enemy_total} | my_prod={w.my_prod} ---")
    
    spent, moves, targeted, planned_commitments = {}, [], set(), []
    if (time.perf_counter() - t0) < DEADLINE_SOFT:
        budgets = {p.id: w.attack_budget(p.id, spent) for p in w.my_planets}
        missions = build_missions(w, spent, targeted, planned_commitments, phase)
        execute_missions(knapsack_select(missions, budgets), w, spent, moves, targeted, planned_commitments)
    
    # Di tản sao chổi sắp biến mất hoặc rút quân chiến thuật
    if (time.perf_counter() - t0) < DEADLINE_SOFT:
        phase_comet_evac(w, spent, moves, planned_commitments)
        phase_tactical_evac(w, spent, moves, planned_commitments)
        
    final_moves = finalize(moves, w)
    t_elapsed = time.perf_counter() - t0
    logger.info(f"[TIME] Turn {w.step} took {t_elapsed:.4f}s")
    return final_moves

__all__ = ["agent"]
