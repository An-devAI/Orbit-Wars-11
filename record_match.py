from kaggle_environments import make
from RuleBasedBot import agent as rule_based_agent
import OrbitalOverlord
from enemy.enemy2 import agent as enemy_agent2
from enemy.enemy3 import agent as enemy_agent3
from enemy.enemy1 import agent as enemy_agent1
from enemy.enemy4 import agent as enemy_agent4

import os
import re

def record_and_save():
    print("Running match: RuleBasedBot vs Random...")
    # Sử dụng môi trường gốc để đảm bảo tính ổn định
    env = make("orbit_wars", debug=True)

    def do_nothing_agent(obs):
        return []
    
    # Chạy trận đấu giữa Bot của bạn và bot Random
    env.run([enemy_agent4, OrbitalOverlord.agent])  # Thêm bot thứ 4 để tránh lỗi "Not enough agents"
    
    # Xuất ra file HTML
    output_path = "replay.html"
    print("Generating HTML...")
    html_content = env.render(mode="html", width=800, height=600)
    
    # Đọc JS tùy chỉnh (đã thêm ID hành tinh)
    custom_js_path = "custom_env/orbit_wars/orbit_wars.js"
    if os.path.exists(custom_js_path):
        print("Applying custom renderer with Planet IDs...")
        with open(custom_js_path, "r", encoding="utf-8") as f:
            custom_js = f.read()
        
        # Regex để tìm và thay thế renderer trong HTML của Kaggle
        # Tìm đoạn: window.kaggle.renderer = async function renderer(context) { ... }
        pattern = r"(window\.kaggle\.renderer = )(async function renderer\(context\) \{.*?\n\})"
        
        # Kiểm tra xem có tìm thấy pattern không
        if re.search(pattern, html_content, re.DOTALL):
            # Thay thế bằng JS mới
            replacement = r"\1" + custom_js
            html_content = re.sub(pattern, replacement, html_content, flags=re.DOTALL)
            print("Successfully injected custom renderer.")
        else:
            print("Warning: Could not find renderer pattern in HTML. Planet IDs might not be visible.")
    else:
        print(f"Warning: Custom JS not found at {custom_js_path}. Using default renderer.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Match finished! Replay saved to: {os.path.abspath(output_path)}")
    print("Bạn hãy mở file 'replay.html' bằng trình duyệt (Chrome, Edge,...) để xem.")

if __name__ == "__main__":
    record_and_save()
