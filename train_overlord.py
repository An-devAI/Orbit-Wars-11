import json
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from OrbitalOverlord import (
    OrbitalOverlordConfig, 
    OverlordRuntime, 
    OverlordMemory,
    extract_candidate_features, 
    get_all_candidates,
    PRESET_FFA_4P,
    CandidateEvaluatorNet
)
from orbit_lite.adapter import single_obs_to_tensor
from orbit_lite.movement import PlanetMovement
from orbit_lite.obs import parse_obs

def extract_data_from_replay(json_path, target_submission_id=None):
    """Bóc tách Feature/Label từ file Replay Kaggle (.json)."""
    print(f"Đang xử lý: {json_path}")
    with open(json_path, 'r') as f:
        replay = json.load(f)
    
    # 1. Tìm player_id của Submission mục tiêu (53402535)
    top_player_id = 0
    if target_submission_id is not None:
        try:
            # Thông tin agents thường nằm trong 'info' -> 'agents'
            agents = replay.get('info', {}).get('agents', [])
            for i, agent in enumerate(agents):
                if str(agent.get('submissionId')) == str(target_submission_id):
                    top_player_id = i
                    print(f"-> Đã xác định: {agent.get('teamName')} (Index {i}) là chuyên gia.")
                    break
        except Exception:
            pass

    memory = OverlordMemory()
    all_features = []
    all_labels = []
    
    steps = replay.get('steps', replay)
    player_count = len(steps[0]) if isinstance(steps[0], list) else 4
    print(f"-> Trận đấu {player_count} người chơi.")

    for step_idx in range(len(steps) - 1):
        step_data = steps[step_idx]
        p_info = step_data[top_player_id] if isinstance(step_data, list) else step_data
        raw_obs = p_info.get('observation', p_info)
        
        next_step_data = steps[step_idx + 1]
        next_p_info = next_step_data[top_player_id] if isinstance(next_step_data, list) else next_step_data
        actual_actions = next_p_info.get('action', [])
        
        if not actual_actions: continue
            
        try:
            obs_tensors = single_obs_to_tensor(raw_obs, player_id=top_player_id)
        except Exception: continue

        config = PRESET_FFA_4P if player_count >= 4 else OrbitalOverlordConfig()
        candidates = get_all_candidates(obs_tensors, config, player_count, memory)
        num_cand = candidates.source_slots.shape[0]
        if num_cand == 0: continue
        
        obs = parse_obs(obs_tensors)
        movement = memory.movement
        garrison_status = movement.garrison_status(max_horizon=int(config.horizon))
        step_ratio = float(obs_tensors["step"]) / 500.0
        
        features = extract_candidate_features(
            obs, movement, candidates, garrison_status, 
            movement.planet_prod, player_count, step_ratio
        )
        
        labels = torch.zeros(num_cand)
        for act in actual_actions:
            if not isinstance(act, list) or len(act) < 2: continue
            act_src, act_angle = int(act[0]), float(act[1])
            src_match = (candidates.source_slots == act_src)
            if src_match.any():
                angle_diff = torch.abs(candidates.angle - act_angle)
                angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                best_match_idx = torch.argmin(torch.where(src_match, torch.abs(angle_diff), torch.tensor(float('inf'))))
                if src_match[best_match_idx] and torch.abs(angle_diff[best_match_idx]) < 0.2:
                    labels[best_match_idx] = 1.0
        
        all_features.append(features)
        all_labels.append(labels)

    if not all_features: return None, None
    return torch.cat(all_features), torch.cat(all_labels)

def train_imitation(X, Y, model_path="model_overlord.pth", batch_size=2048):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = ~torch.isnan(X).any(dim=1)
    X, Y = X[mask], Y[mask]
    
    # 1. Khởi tạo Model (Giả sử model của bạn chưa có lớp Sigmoid ở cuối)
    model = CandidateEvaluatorNet(input_dim=X.shape[1]).to(device)

    num_pos = (Y == 1).sum().item()
    num_neg = (Y == 0).sum().item()
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-6)], dtype=torch.float32).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    
    # =================================================================
    # BẢO VỆ RAM: TẠO DATALOADER ĐỂ TRAIN THEO TỪNG MẺ (MINI-BATCH)
    # =================================================================
    dataset = TensorDataset(X.to(torch.float32), Y.to(torch.float32).unsqueeze(1))
    # shuffle=True rất quan trọng để AI không học thuộc lòng thứ tự
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    
    print(f"Bắt đầu huấn luyện: {int(num_pos)} Nhãn 1 / {int(num_neg)} Nhãn 0")
    print(f"Sử dụng Device: {device} | Batch size: {batch_size}")
    
    model.train()
    for epoch in range(150): # Giảm số epoch xuống vì dùng Dataloader nó hội tụ nhanh hơn
        total_loss = 0.0
        
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            logits = logits.view(-1)     # Ép dự đoán về mảng 1 chiều [2048]
            batch_Y = batch_Y.view(-1)   # Ép đáp án về mảng 1 chiều [2048]
            # Logits có shape [Batch, 1], Y cũng có shape [Batch, 1]
            loss = criterion(logits, batch_Y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.6f}")
            
    # Lưu trọng số
    torch.save(model.state_dict(), model_path)
    print(f"✅ Đã huấn luyện xong! Đã lưu trọng số vào {model_path}")


if __name__ == "__main__":
    SUBMISSION_ID = 53402535
    REPLAY_DIR = "replays/"
    
    all_X = []
    all_Y = []
    
    # Quét toàn bộ file JSON trong thư mục replays
    json_files = glob.glob(os.path.join(REPLAY_DIR, "*.json"))
    print(f"Tìm thấy {len(json_files)} file replay.")
    
    for filepath in json_files:
        try:
            x, y = extract_data_from_replay(filepath, target_submission_id=SUBMISSION_ID)
            if x is not None and len(x) > 0:
                all_X.append(x)
                all_Y.append(y)
        except Exception as e:
            print(f"Lỗi khi đọc file {filepath}: {e}")
            
    if len(all_X) > 0:
        # Gộp dữ liệu của TẤT CẢ các trận đấu lại thành 1 Tensor siêu to khổng lồ
        final_X = torch.cat(all_X, dim=0)
        final_Y = torch.cat(all_Y, dim=0)
        
        print(f"Tổng hợp dữ liệu xong: X={final_X.shape}, Y={final_Y.shape}")
        
        # Bắt đầu Train
        train_imitation(final_X, final_Y)
    else:
        print("Không có dữ liệu hợp lệ để huấn luyện.")