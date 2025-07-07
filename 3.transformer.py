# ==================================================================
# Transformer 모델 단독 실행 스크립트 (오류 수정 최종본)
# ==================================================================
# 이 스크립트는 제공된 Transformer.py의 모든 로직을 포함하며,
# 단독으로 실행하여 모델 훈련, 예측, 평가, 시각화까지 수행합니다.
#
# 실행 전 준비물:
# - ETTh1.csv
# - sample_submit.csv
# ==================================================================

#
# ──────────────────────────────────────────────────────────────
# 1. 라이브러리 로드 및 랜덤 시드 고정
# ──────────────────────────────────────────────────────────────
#
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import math
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

print("✅ [1단계] 라이브러리 임포트 및 시드 고정 완료.")

#
# ──────────────────────────────────────────────────────────────
# 2. 경로 설정
# ──────────────────────────────────────────────────────────────
#
# 모든 파일이 저장되고 로드될 기본 경로
BASE_PATH = os.getcwd()
output_directory = os.path.join(BASE_PATH, "transformer_results/")
os.makedirs(output_directory, exist_ok=True)
print(f"📂 최종 결과물은 다음 경로에 저장됩니다: {output_directory}")


# 입력 파일 경로 정의
ETTH1_PATH = os.path.join(BASE_PATH, "ETTh1.csv")
SUBMIT_SAMPLE_PATH = os.path.join(BASE_PATH, "sample_submit.csv")

# 출력 파일 경로 정의
MODEL_SAVE_PATH = os.path.join(output_directory, "best_transformer_model.pth")
SUBMISSION_CSV_PATH = os.path.join(output_directory, "submission_transformer.csv")
HIST_IMG_PATH = os.path.join(output_directory, "histogram_transformer.png")

print("✅ [2단계] 모든 파일 경로 설정 완료.")

#
# ──────────────────────────────────────────────────────────────
# 3. 데이터 로드 및 특성 공학
# ──────────────────────────────────────────────────────────────
#
try:
    # 원본 데이터는 'df' 라는 변수명으로 로드합니다.
    df = pd.read_csv(ETTH1_PATH)
    submit_df_raw = pd.read_csv(SUBMIT_SAMPLE_PATH)
except FileNotFoundError as e:
    print(f"❌ 파일 오류: {e}")
    print("스크립트를 실행하기 전에 ETTh1.csv와 sample_submit.csv 파일이 있는지 확인해주세요.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

df_feat = df.copy()
df_feat['hour']    = df_feat.index.hour
df_feat['weekday'] = df_feat.index.weekday

def cyclical_encode(vals, max_val):
    radians = 2.0 * math.pi * vals / max_val
    return np.sin(radians), np.cos(radians)

sin_h, cos_h   = cyclical_encode(df_feat['hour'].values, 24)
sin_wd, cos_wd = cyclical_encode(df_feat['weekday'].values, 7)
df_feat['sin_hour'], df_feat['cos_hour'] = sin_h, cos_h
df_feat['sin_wd'], df_feat['cos_wd']     = sin_wd, cos_wd

for window in [24, 48, 168]:
    df_feat[f'OT_rollmean_{window}'] = df_feat['OT'].rolling(window).mean().bfill()
    df_feat[f'OT_rollstd_{window}']  = df_feat['OT'].rolling(window).std().bfill()
    df_feat[f'OT_lag_{window}']      = df_feat['OT'].shift(window).bfill()
    df_feat[f'OT_diff_{window}']     = df_feat['OT'] - df_feat['OT'].shift(window).bfill()

feature_cols = [
    'HUFL','HULL','MUFL','MULL','LUFL','LULL','OT',
    'sin_hour','cos_hour','sin_wd','cos_wd',
    'OT_rollmean_24','OT_rollmean_48','OT_rollmean_168',
    'OT_rollstd_24','OT_rollstd_48','OT_rollstd_168',
    'OT_lag_24','OT_lag_48','OT_lag_168',
    'OT_diff_24','OT_diff_48','OT_diff_168',
]
# 생성된 특성만 사용하도록 DataFrame 필터링
df_feat = df_feat[feature_cols]

print("✅ [3단계] 특성 공학 완료.")

#
# ──────────────────────────────────────────────────────────────
# 4. 데이터 정규화 및 분할
# ──────────────────────────────────────────────────────────────
#
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_feat)
scaled_df = pd.DataFrame(data_scaled, index=df_feat.index, columns=feature_cols)

# 스케일된 OT의 분위수 기준 임계값 계산
low_thr_scaled, high_thr_scaled = np.percentile(scaled_df['OT'], [20, 80])
print(f"📊 가중 손실(WeightedLoss) 임계값: 20%={low_thr_scaled:.3f} | 80%={high_thr_scaled:.3f}")

# 학습/검증 분할 기준 설정 (Leak 방지)
last_label_date = datetime(2018,1,31,23,0)
max_t0_date     = last_label_date - timedelta(hours=95)
df_trainval = scaled_df[:last_label_date]

input_len = 172
pred_len  = 96

print("✅ [4단계] 데이터 정규화 및 분할 완료.")


#
# ──────────────────────────────────────────────────────────────
# 5. 모델 및 데이터셋 클래스 정의
# ──────────────────────────────────────────────────────────────
#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  사용 디바이스: {device}")

class TS_Dataset(Dataset):
    def __init__(self, df, input_len, pred_len, feature_cols, max_t0_date, last_label_date):
        self.feat = df[feature_cols].values
        self.input_len, self.pred_len = input_len, pred_len
        self.feature_cols = feature_cols
        last_label_idx = df.index.get_loc(last_label_date)
        max_t0_idx = df.index.get_loc(max_t0_date)
        self.start_idx = [i for i in range(self.input_len, max_t0_idx + 1) if (i + self.pred_len - 1) <= last_label_idx]
    def __len__(self): return len(self.start_idx)
    def __getitem__(self, idx):
        i = self.start_idx[idx]
        x = self.feat[i - self.input_len : i]
        y = self.feat[i : i + self.pred_len, self.feature_cols.index('OT')]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def mark_high_windows(start_idx_list, feat_array, input_len, high_cut):
    flags = []
    for i in start_idx_list:
        win = feat_array[i - input_len : i, feature_cols.index('OT')]
        flags.append( (win >= high_cut).any() )
    return np.array(flags)

class AsymWeightedMSE(nn.Module):
    def __init__(self, low_thr, high_thr, w_low=0.2, w_high=2.0):
        super().__init__()
        self.low_thr, self.high_thr = low_thr, high_thr
        self.w_low, self.w_high = w_low, w_high
    def forward(self, pred, target):
        w = torch.ones_like(target)
        w[target < self.low_thr] = self.w_low
        w[target >= self.high_thr] = self.w_high
        return (w * (pred - target) ** 2).mean()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, feature_dim, d_model=256, nhead=4, num_layers=2, dim_feed=256, dropout=0.2, pred_len=96):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(feature_dim, d_model), nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feed, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, pred_len))
    def forward(self, x):
        x = self.layer_norm(self.input_proj(x))
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.output_fc(x[:, -1, :])

print("✅ [5단계] 모델 및 데이터셋 클래스 정의 완료.")

#
# ──────────────────────────────────────────────────────────────
# 6. 학습/검증 DataLoader 생성 + Oversampling
# ──────────────────────────────────────────────────────────────
#
full_ds   = TS_Dataset(df_trainval, input_len, pred_len, feature_cols, max_t0_date, last_label_date)
n_total   = len(full_ds)
n_train   = int(n_total * 0.8)
train_idx = list(range(n_train))
val_idx   = list(range(n_train, n_total))

train_ds  = Subset(full_ds, train_idx)
val_ds    = Subset(full_ds, val_idx)

# 고온 윈도우 플래그 계산
high_flags = mark_high_windows(train_idx, full_ds.feat, input_len, high_thr_scaled)
# 샘플 가중치: 고온=2, 나머지=1
sample_weights = np.where(high_flags, 2.0, 1.0)
# WeightedRandomSampler 정의
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoader 생성
batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

print("✅ [6단계] 가중 샘플링을 적용한 데이터 로더 생성 완료.")

#
# ──────────────────────────────────────────────────────────────
# 7. Transformer 모델 훈련
# ──────────────────────────────────────────────────────────────
#
transformer = TransformerModel(feature_dim=len(feature_cols), pred_len=pred_len).to(device)
optimizer = torch.optim.AdamW(transformer.parameters(), lr=5e-5, weight_decay=1e-5)
criterion = AsymWeightedMSE(low_thr_scaled, high_thr_scaled, w_low=0.2, w_high=2.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

num_epochs = 40
early_patience = 25
best_val_loss = float('inf')
early_stop_counter = 0

print("\n" + "="*80 + "\n▶ PART 7: Transformer 모델 훈련 시작\n" + "="*80)
for epoch in range(1, num_epochs + 1):
    transformer.train(); train_losses = []
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device); optimizer.zero_grad()
        loss = criterion(transformer(x_b), y_b); loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0); optimizer.step()
        train_losses.append(loss.item())
    avg_tr_loss = np.mean(train_losses)

    transformer.eval(); val_losses = []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            val_losses.append(criterion(transformer(x_b.to(device)), y_b.to(device)).item())
    avg_val_loss = np.mean(val_losses)

    print(f"  [Ep {epoch:02d}] Train Loss: {avg_tr_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss - 1e-6:
        best_val_loss = avg_val_loss
        torch.save(transformer.state_dict(), MODEL_SAVE_PATH)
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_patience:
            print(f"  -- 조기 종료 at epoch {epoch} --")
            break

print(f"\n✅ [7단계] 모델 훈련 완료. 최종 검증 손실: {best_val_loss:.6f}")

#
# ──────────────────────────────────────────────────────────────
# 8. 예측 및 결과 저장/평가
# ──────────────────────────────────────────────────────────────
#
print("\n" + "="*80 + "\n▶ PART 8: 예측 및 결과 평가 시작\n" + "="*80)

results = []
best_model = TransformerModel(feature_dim=len(feature_cols), pred_len=pred_len).to(device)
best_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
best_model.eval()

# 'OT' 컬럼의 원래 스케일 정보를 역변환에 사용
ot_col_index = feature_cols.index('OT')
ot_min, ot_max = scaler.data_min_[ot_col_index], scaler.data_max_[ot_col_index]

for _, row in submit_df_raw.iterrows():
    t0 = pd.to_datetime(row[0])
    input_end   = t0 - timedelta(hours=1)
    input_start = input_end - timedelta(hours=input_len - 1)

    seq = scaled_df.loc[input_start:input_end, feature_cols].values
    x_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_scaled = best_model(x_tensor).cpu().numpy().flatten()
    
    pred_unnorm = pred_scaled * (ot_max - ot_min) + ot_min
    results.append([row[0]] + pred_unnorm.tolist())

col_names = [submit_df_raw.columns[0]] + [f"T{i}" for i in range(pred_len)]
df_submit = pd.DataFrame(results, columns=col_names)
df_submit.to_csv(SUBMISSION_CSV_PATH, index=False)
print(f"✅ 예측 파일 생성 완료: {SUBMISSION_CSV_PATH}")

# --- 성능 평가 및 시각화 ---
y_pred_flat = df_submit.drop(columns=[submit_df_raw.columns[0]]).values.flatten()
submit_times = pd.to_datetime(df_submit.iloc[:,0].values)
gt_ots = []
for t in submit_times:
    gt_range = pd.date_range(start=t, periods=pred_len, freq='H')
    # --- 수정된 부분 ---
    # 'df_raw' 대신 'df'를 사용하여 정답 값을 가져옵니다.
    gt_series = df.reindex(gt_range)['OT']
    # --- 수정 끝 ---
    gt_ots.append(gt_series.values)
gt_ots_flat = np.concatenate(gt_ots)
valid_indices = ~np.isnan(gt_ots_flat)
gt_ots_clean = gt_ots_flat[valid_indices]
y_pred_clean = y_pred_flat[valid_indices]

# 성능 지표 출력
mae = mean_absolute_error(gt_ots_clean, y_pred_clean)
mse = mean_squared_error(gt_ots_clean, y_pred_clean)
rmse = np.sqrt(mse)
r2 = r2_score(gt_ots_clean, y_pred_clean)
print("\n  【 최종 성능 지표 】")
print(f"  - MSE  : {mse:.4f} (Kaggle 평가지표)")
print(f"  - MAE  : {mae:.4f}")
print(f"  - RMSE : {rmse:.4f}")
print(f"  - R²   : {r2:.4f}")

# 히스토그램 생성
plt.figure(figsize=(12, 7))
plt.hist(gt_ots_clean, bins=50, alpha=0.7, label='Ground Truth', color='royalblue', density=True)
plt.hist(y_pred_clean, bins=50, alpha=0.7, label='Prediction', color='orangered', density=True)
plt.title("Transformer Prediction vs. Ground Truth Distribution", fontsize=16)
plt.xlabel("OT Value", fontsize=12); plt.ylabel("Density", fontsize=12)
plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
plt.savefig(HIST_IMG_PATH)
plt.close()
print(f"\n✅ 비교 히스토그램 저장 완료: {HIST_IMG_PATH}")

print("\n🎉 모든 작업이 성공적으로 완료되었습니다. 🎉")