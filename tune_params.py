import os
import gc
import sys
import logging

# TẮT TOÀN BỘ LOG RÁC CỦA KAGGLE TRƯỚC KHI IMPORT
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KAGGLE_ENVIRONMENTS_LOG_LEVEL'] = 'CRITICAL'
os.environ['LITELLM_LOG'] = 'CRITICAL'

import optuna
from kaggle_environments import make

# Cài đặt chế độ Tune TRƯỚC KHI import Bot để tắt mồm Bot
import builtins
builtins.TUNE_MODE = True

import RuleBasedBot
from enemy.enemy1 import agent as enemy_agent1
from enemy.enemy2 import agent as enemy_agent2
from enemy.enemy3 import agent as enemy_agent3
from OrbitalOverlord import agent as boss_agent

# ================= CẤU HÌNH LOGGING CHUẨN =================
# Xóa mọi cấu hình log cũ
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_file = "tuning_results.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Tuner")

optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
# ==========================================================

def objective(trial):
    early_params = {
        "ALPHA": trial.suggest_float("EARLY_ALPHA", 0.5, 5.0),
        "BETA": trial.suggest_float("EARLY_BETA", 0.1, 4.0),
        "GAMMA": trial.suggest_float("EARLY_GAMMA", 0.1, 2.0),
        "DELTA": trial.suggest_float("EARLY_DELTA", 0.5, 6.0),
        "EPSILON": trial.suggest_float("EARLY_EPSILON", 0.1, 5.0)
    }
    late_params = {
        "ALPHA": trial.suggest_float("LATE_ALPHA", 0.5, 5.0),
        "BETA": trial.suggest_float("LATE_BETA", 0.1, 5.0),
        "GAMMA": trial.suggest_float("LATE_GAMMA", 0.1, 2.0),
        "DELTA": trial.suggest_float("LATE_DELTA", 0.5, 6.0),
        "EPSILON": trial.suggest_float("LATE_EPSILON", 0.1, 5.0)
    }

    # Nạp tham số vào Bot
    for k, v in early_params.items(): setattr(RuleBasedBot, f"EARLY_{k}", v)
    for k, v in late_params.items(): setattr(RuleBasedBot, f"LATE_{k}", v)

    total_score = 0
    opponents = [enemy_agent1, enemy_agent2, enemy_agent3]
    num_matches = len(opponents) * 2
    
    logger.info(f"--- Đang chạy Trial {trial.number} (Mô phỏng {num_matches} trận 1vs1) ---")
    
    for i in range(num_matches):
        env = make("orbit_wars", debug=False)
        
        opponent = opponents[i // 2]
        me_idx = i % 2
        
        agents = [RuleBasedBot.agent, opponent]
        if me_idx == 1:
            agents = [opponent, RuleBasedBot.agent]
        
        # Chạy game
        env.run(agents)
            
        # Tính điểm
        last_state = env.steps[-1]
        ships = [0, 0]
        for p in last_state[0].observation.planets:
            if p[1] != -1: ships[p[1]] += p[5]
        for f in last_state[0].observation.fleets:
            ships[f[1]] += f[6]
            
        my_ships = ships[me_idx]
        enemy_ships = ships[1 - me_idx]
        
        if my_ships > enemy_ships:
            match_score = 1000 + (my_ships - enemy_ships)
            res = "THẮNG"
        else:
            match_score = (my_ships / max(1.0, enemy_ships)) * 500
            res = "THUA"
            
        enemy_name = f"enemy{i//2 + 1}"
        total_score += match_score
        logger.info(f"  > Trận {i+1}/{num_matches} vs {enemy_name}: {res} | Lính của mình: {my_ships:.0f} vs {enemy_name}: {enemy_ships:.0f}")
        
        # GIẢI PHÓNG RAM SAU MỖI TRẬN
        del env
        del last_state
        gc.collect() 
            
    avg_score = total_score / num_matches
    logger.info(f"==> KẾT QUẢ TRIAL {trial.number}: {avg_score:.1f} ĐIỂM\n")
    return avg_score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    logger.info("BẮT ĐẦU HUẤN LUYỆN AI... (Vui lòng đợi)")
    try:
        # Chạy thử 50 lần trước cho nhanh (mất khoảng 30 - 45 phút)
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        logger.info("Đã dừng bởi người dùng.")
        
    best = study.best_params
    logger.info("\n=================================")
    logger.info(" BỘ THAM SỐ VÔ ĐỊCH TÌM ĐƯỢC:")
    logger.info("=================================")
    logger.info(f"EARLY GAME: ALPHA={best['EARLY_ALPHA']:.2f}, BETA={best['EARLY_BETA']:.2f}, GAMMA={best['EARLY_GAMMA']:.2f}, DELTA={best['EARLY_DELTA']:.2f}, EPSILON={best['EARLY_EPSILON']:.2f}")
    logger.info(f"LATE GAME : ALPHA={best['LATE_ALPHA']:.2f}, BETA={best['LATE_BETA']:.2f}, GAMMA={best['LATE_GAMMA']:.2f}, DELTA={best['LATE_DELTA']:.2f}, EPSILON={best['LATE_EPSILON']:.2f}")