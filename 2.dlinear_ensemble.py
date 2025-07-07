
# ──────────────────────────────────────────────────────────────
# 1. 라이브러리 임포트 및 시드 고정
# ──────────────────────────────────────────────────────────────
#
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import math
import os
import matplotlib.pyplot as plt

# 재현성을 위한 시드 고정
torch.manual_seed(42)
np.random.seed(42)

print("✅ [1단계] 라이브러리 임포트 완료.")

#
# ──────────────────────────────────────────────────────────────
# 2. 경로 설정
# ──────────────────────────────────────────────────────────────
#
BASE_PATH = os.getcwd()

# === 중요: 미리 훈련된 모델 파일(.pth)들이 저장된 경로를 지정해주세요 ===
# 예시: 이전 dl3.py를 실행한 결과 폴더
PRE_TRAINED_MODEL_PATH = os.path.join(BASE_PATH, "dlinear_all_experiments_results/")

# 입력 데이터 경로
ETTH1_PATH = os.path.join(BASE_PATH, "ETTh1.csv")
SUBMIT_SAMPLE_PATH = os.path.join(BASE_PATH, "sample_submit.csv")

# 최종 결과물이 저장될 디렉토리
output_directory = os.path.join(BASE_PATH, "conditional_ensemble_result/")
os.makedirs(output_directory, exist_ok=True)
print(f"📂 훈련된 모델 로드 경로: {PRE_TRAINED_MODEL_PATH}")
print(f"📂 최종 결과물 저장 경로: {output_directory}")


#
# ──────────────────────────────────────────────────────────────
# 3. 데이터 로드 및 전처리
# ──────────────────────────────────────────────────────────────
#
try:
    df = pd.read_csv(ETTH1_PATH)
    submit_df_raw = pd.read_csv(SUBMIT_SAMPLE_PATH)
except FileNotFoundError as e:
    print(f"❌ 파일 오류: {e}")
    print("스크립트를 실행하기 전에 ETTh1.csv와 sample_submit.csv 파일이 있는지 확인해주세요.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 모델 훈련 시와 동일한 특성 공학 수행
df_feat = df.copy()
df_feat['hour'] = df_feat.index.hour
df_feat['weekday'] = df_feat.index.weekday
df_feat['day_of_year'] = df_feat.index.dayofyear

def cyclical_encode(vals, max_val):
    radians = 2.0 * math.pi * vals / max_val
    return np.sin(radians), np.cos(radians)

sin_h, cos_h = cyclical_encode(df_feat['hour'].values, 24)
sin_wd, cos_wd = cyclical_encode(df_feat['weekday'].values, 7)
sin_doy, cos_doy = cyclical_encode(df_feat['day_of_year'].values, 366)
df_feat['sin_hour'], df_feat['cos_hour'] = sin_h, cos_h
df_feat['sin_wd'], df_feat['cos_wd'] = sin_wd, cos_wd
df_feat['sin_doy'], df_feat['cos_doy'] = sin_doy, cos_doy

bins = [0, 6, 12, 18, 23]; labels = ['Night', 'Morning', 'Afternoon', 'Evening']
df_feat['time_of_day'] = pd.cut(df_feat['hour'], bins=bins, labels=labels, right=False, ordered=False)
df_feat.loc[df_feat['hour'] == 23, 'time_of_day'] = 'Evening'
df_feat['is_weekend'] = df_feat['weekday'].apply(lambda x: 1 if x >= 5 else 0)

def get_season(doy):
    if 80 <= doy < 172: return 'Spring'
    elif 172 <= doy < 264: return 'Summer'
    elif 264 <= doy < 355: return 'Autumn'
    else: return 'Winter'
df_feat['season'] = df_feat['day_of_year'].apply(get_season)
df_feat = pd.get_dummies(df_feat, columns=['time_of_day', 'season'], drop_first=True)

for window in [24, 48, 168, 336]:
    df_feat[f'OT_rollmean_{window}'] = df_feat['OT'].rolling(window, min_periods=1).mean().shift(1)
    df_feat[f'OT_rollstd_{window}'] = df_feat['OT'].rolling(window, min_periods=1).std().shift(1)
    df_feat[f'OT_lag_{window}'] = df_feat['OT'].shift(window)
    df_feat[f'OT_diff_{window}'] = df_feat['OT'].diff(periods=window)

df_feat.fillna(0, inplace=True)

base_feature_cols = ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
cyclical_cols = ['sin_hour','cos_hour','sin_wd','cos_wd', 'sin_doy', 'cos_doy']
rolling_cols = [col for col in df_feat.columns if 'OT_' in col]
ohe_cols = [col for col in df_feat.columns if 'time_of_day_' in col or 'season_' in col or 'is_weekend' in col]
feature_cols = base_feature_cols + cyclical_cols + ohe_cols + rolling_cols

# 데이터 정규화
train_end_date = "2018-01-01 00:00:00"
train_data = df_feat[df_feat.index < train_end_date]
scaler = MinMaxScaler()
scaler.fit(train_data[feature_cols])
ot_scaler = MinMaxScaler()
ot_scaler.fit(train_data[['OT']])
data_scaled = scaler.transform(df_feat[feature_cols])
scaled_df = pd.DataFrame(data_scaled, index=df_feat.index, columns=feature_cols)

print("✅ [3단계] 데이터 전처리 완료.")


#
# ──────────────────────────────────────────────────────────────
# 4. 모델 클래스 정의 및 모델 로드
# ──────────────────────────────────────────────────────────────
#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  사용 디바이스: {device}")

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        front = x[:, 0:1, :].repeat(1,(self.kernel_size-1)//2,1)
        end = x[:,-1:,:].repeat(1,(self.kernel_size-1)//2,1)
        return self.avg(torch.cat([front,x,end],dim=1).permute(0,2,1)).permute(0,2,1)

class DLinear_Leakproof(nn.Module):
    def __init__(self, input_len, pred_len, kernel_size, dropout_rate):
        super().__init__()
        self.decompsition = MovingAvg(kernel_size, stride=1)
        self.linear_seasonal = nn.Sequential(nn.Linear(input_len, pred_len), nn.Dropout(dropout_rate))
        self.linear_trend = nn.Sequential(nn.Linear(input_len, pred_len), nn.Dropout(dropout_rate))
    def forward(self, x):
        ot_index = feature_cols.index('OT')
        x_ot = x[:, :, ot_index].unsqueeze(-1)
        trend = self.decompsition(x_ot)
        seasonal = (x_ot - trend).squeeze(-1); trend = trend.squeeze(-1)
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)

# 앙상블할 '지역 전문가' 모델들
pred_len = 96
ensemble_members = {
    'high_temp_expert': {
        'exp_name': 'DLinear_best_combo',
        'params': {'input_len': 504, 'pred_len': pred_len, 'kernel_size': 25, 'dropout_rate': 0.1},
    },
    'low_temp_expert': {
        'exp_name': 'DLinear_best_combo_MAE',
        'params': {'input_len': 504, 'pred_len': pred_len, 'kernel_size': 25, 'dropout_rate': 0.1},
    },
    'guide': {
        'exp_name': 'DLinear_len336_base',
        'params': {'input_len': 336, 'pred_len': pred_len, 'kernel_size': 25, 'dropout_rate': 0.1},
    }
}

models = {}
try:
    for role, config in ensemble_members.items():
        model = DLinear_Leakproof(**config['params']).to(device)
        model_path = os.path.join(PRE_TRAINED_MODEL_PATH, f"model_{config['exp_name']}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[role] = model
        print(f"  - [{role.replace('_', ' ').title()}] 모델 로드 완료: {model_path}")
except FileNotFoundError as e:
    print(f"❌ 모델 파일 로드 실패: {e}")
    print(f"'{PRE_TRAINED_MODEL_PATH}' 경로에 훈련된 모델 파일(.pth)들이 있는지 확인해주세요.")
    exit()

print("✅ [4단계] 모든 전문가 모델 로드 완료.")


#
# ──────────────────────────────────────────────────────────────
# 5. 조건부 앙상블 수행
# ──────────────────────────────────────────────────────────────
#
print("\n" + "="*80 + "\n▶ PART 5: 조건부 앙상블 생성 시작\n" + "="*80)

SWITCHING_THRESHOLD = 7.5
HIGH_TEMP_WEIGHTS = {'high_temp_expert': 0.7, 'low_temp_expert': 0.2, 'guide': 0.1}
LOW_TEMP_WEIGHTS = {'high_temp_expert': 0.2, 'low_temp_expert': 0.7, 'guide': 0.1}

ensemble_results = []
for _, row in submit_df_raw.iterrows():
    t0 = pd.to_datetime(row[submit_df_raw.columns[0]])
    predictions = {}

    for role, model in models.items():
        input_len = ensemble_members[role]['params']['input_len']
        input_start = t0 - timedelta(hours=input_len); input_end = t0 - timedelta(hours=1)
        seq = scaled_df.loc[input_start:input_end].values
        x_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled = model(x_tensor).cpu().numpy().flatten()
        predictions[role] = ot_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    guide_preds = predictions['guide']
    final_pred = np.zeros(pred_len)
    for i in range(pred_len):
        weights = HIGH_TEMP_WEIGHTS if guide_preds[i] >= SWITCHING_THRESHOLD else LOW_TEMP_WEIGHTS
        final_pred[i] = sum(predictions[role][i] * weights[role] for role in weights)
    ensemble_results.append([row[0]] + final_pred.tolist())

col_names=[submit_df_raw.columns[0]] + [f"T{i}" for i in range(pred_len)]
conditional_ensemble_df = pd.DataFrame(ensemble_results, columns=col_names)

print("✅ [5단계] 조건부 앙상블 예측 결과 생성 완료.")

#
# ──────────────────────────────────────────────────────────────
# 6. 최종 결과 저장 및 평가
# ──────────────────────────────────────────────────────────────
#
print("\n" + "="*80 + "\n▶ PART 6: 최종 결과 저장 및 평가 시작\n" + "="*80)

submission_filename = os.path.join(output_directory, "submission_conditional_ensemble.csv")
conditional_ensemble_df.to_csv(submission_filename, index=False)
print(f"✅ 조건부 앙상블 예측 파일 저장 완료: {submission_filename}")

# --- 성능 평가 ---
y_pred_flat = conditional_ensemble_df.drop(columns=[submit_df_raw.columns[0]]).values.flatten()
submit_times = pd.to_datetime(conditional_ensemble_df[submit_df_raw.columns[0]].values)
gt_ots = []
for t in submit_times:
    gt_range = pd.date_range(start=t, periods=pred_len, freq='H')
    gt_series = df.reindex(gt_range)['OT']
    gt_ots.append(gt_series.values)
gt_ots_flat = np.concatenate(gt_ots)
valid_indices = ~np.isnan(gt_ots_flat)
gt_ots_clean = gt_ots_flat[valid_indices]
y_pred_clean = y_pred_flat[valid_indices]

mse = mean_squared_error(gt_ots_clean, y_pred_clean)
mae = mean_absolute_error(gt_ots_clean, y_pred_clean)
rmse = np.sqrt(mse)
r2 = r2_score(gt_ots_clean, y_pred_clean)

print("\n  【 조건부 앙상블 최종 성능 지표 】")
print(f"  - MSE  : {mse:.4f} (Kaggle 평가지표)")
print(f"  - MAE  : {mae:.4f}")
print(f"  - RMSE : {rmse:.4f}")
print(f"  - R²   : {r2:.4f}")

# --- 시각화 ---
plt.figure(figsize=(12, 7))
plt.hist(gt_ots_clean, bins=50, alpha=0.7, label='Ground Truth', color='royalblue')
plt.hist(y_pred_clean, bins=50, alpha=0.7, label='Conditional Ensemble Prediction', color='green')
plt.title("Conditional Ensemble Prediction vs. Ground Truth Distribution", fontsize=16)
plt.xlabel("OT Value", fontsize=12); plt.ylabel("Frequency", fontsize=12)
plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
hist_filename = os.path.join(output_directory, "hist_conditional_ensemble.png")
plt.savefig(hist_filename)
plt.close()
print(f"\n✅ 비교 히스토그램 저장 완료: {hist_filename}")

print("\n🎉 모든 작업이 성공적으로 완료되었습니다. 🎉")