# ==================================================================
# dl3.py 단독 실행 스크립트 (경로 설정 수정 버전)
# ==================================================================
# 원본 코드에서 파일 경로 부분만 실행 위치 기준으로 변경한 버전입니다.
#
# 실행 전 준비물:
# - 이 스크립트와 동일한 폴더에 ETTh1.csv, sample_submit.csv 파일 위치
# ==================================================================

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
# 2. 파일 경로 및 디렉토리 설정 (상대 경로 방식으로 수정)
# ──────────────────────────────────────────────────────────────
#
# --- 수정된 부분 시작 ---
# 현재 스크립트가 실행되는 디렉토리를 기준으로 경로를 설정합니다.
BASE_PATH = os.getcwd()
data_path = os.path.join(BASE_PATH, "ETTh1.csv")
sample_submission_path = os.path.join(BASE_PATH, "sample_submit.csv")

# 모든 결과가 저장될 디렉토리 ('dlinear_all_experiments_results')를 생성합니다.
output_directory = os.path.join(BASE_PATH, "dlinear_all_experiments_results/")
os.makedirs(output_directory, exist_ok=True)

print(f"📂 모든 출력 파일은 다음 경로에 저장됩니다: {output_directory}")
# --- 수정된 부분 끝 ---


#
# ──────────────────────────────────────────────────────────────
# 3. 데이터 불러오기 및 특성 공학 (*** 지능형 특성 추가 ***)
# ──────────────────────────────────────────────────────────────
#
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

df_feat = df.copy()
# --- 기존 특성 공학 ---
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

# --- (신규) 지능형 특성 공학 ---
# 1. 시간대 특성
bins = [0, 6, 12, 18, 23]
labels = ['Night', 'Morning', 'Afternoon', 'Evening']
df_feat['time_of_day'] = pd.cut(df_feat['hour'], bins=bins, labels=labels, right=False, ordered=False)
df_feat.loc[df_feat['hour'] == 23, 'time_of_day'] = 'Evening' # 23시를 저녁으로 포함

# 2. 주말 특성
df_feat['is_weekend'] = df_feat['weekday'].apply(lambda x: 1 if x >= 5 else 0)

# 3. 계절 특성 (북반구 기준)
def get_season(doy):
    if 80 <= doy < 172: return 'Spring'
    elif 172 <= doy < 264: return 'Summer'
    elif 264 <= doy < 355: return 'Autumn'
    else: return 'Winter'
df_feat['season'] = df_feat['day_of_year'].apply(get_season)

# 범주형 변수 원-핫 인코딩
df_feat = pd.get_dummies(df_feat, columns=['time_of_day', 'season'], drop_first=True)
print("✅ [3-1단계] 지능형 특성 공학(시간대, 주말, 계절) 및 원-핫 인코딩 완료.")

# --- 기존 롤링 특성 ---
for window in [24, 48, 168, 336]:
    df_feat[f'OT_rollmean_{window}'] = df_feat['OT'].rolling(window, min_periods=1).mean().shift(1)
    df_feat[f'OT_rollstd_{window}'] = df_feat['OT'].rolling(window, min_periods=1).std().shift(1)
    df_feat[f'OT_lag_{window}'] = df_feat['OT'].shift(window)
    df_feat[f'OT_diff_{window}'] = df_feat['OT'].diff(periods=window)

df_feat.fillna(0, inplace=True)

# --- 최종 특성 리스트 업데이트 ---
# 기존 특성에서 원본 범주형 변수 제거 및 원-핫 인코딩된 변수 추가
base_feature_cols = ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
cyclical_cols = ['sin_hour','cos_hour','sin_wd','cos_wd', 'sin_doy', 'cos_doy']
rolling_cols = [col for col in df_feat.columns if 'OT_' in col]
# 원-핫 인코딩으로 생성된 새로운 특성 이름들을 동적으로 추가
ohe_cols = [col for col in df_feat.columns if 'time_of_day_' in col or 'season_' in col or 'is_weekend' in col]

feature_cols = base_feature_cols + cyclical_cols + ohe_cols + rolling_cols
print("✅ [3-2단계] 전체 특성 공학 완료.")


# ──────────────────────────────────────────────────────────────
# 4. 데이터 정규화 (누수 방지)
# ──────────────────────────────────────────────────────────────
train_end_date = "2018-01-01 00:00:00"
train_data = df_feat[df_feat.index < train_end_date]

scaler = MinMaxScaler()
scaler.fit(train_data[feature_cols])
ot_scaler = MinMaxScaler()
ot_scaler.fit(train_data[['OT']])

data_scaled = scaler.transform(df_feat[feature_cols])
scaled_df = pd.DataFrame(data_scaled, index=df_feat.index, columns=feature_cols)
print(f"✅ [4단계] 누수 방지 스케일링 완료. (훈련 데이터 종료 시점: {train_end_date})")

# ──────────────────────────────────────────────────────────────
# 5. 모델 및 데이터셋 클래스 정의
# ──────────────────────────────────────────────────────────────
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
    def __init__(self, input_len, pred_len, kernel_size, dropout_rate, feature_dim, **kwargs):
        super().__init__()
        self.decompsition = MovingAvg(kernel_size, stride=1)
        # *** 입력 차원을 feature_dim으로 사용하도록 수정 ***
        self.linear_seasonal = nn.Sequential(nn.Linear(input_len, pred_len), nn.Dropout(dropout_rate))
        self.linear_trend = nn.Sequential(nn.Linear(input_len, pred_len), nn.Dropout(dropout_rate))
    def forward(self, x):
        # OT 특성의 인덱스를 동적으로 찾기
        ot_index = feature_cols.index('OT')
        x_ot = x[:, :, ot_index].unsqueeze(-1)
        trend = self.decompsition(x_ot)
        seasonal = (x_ot - trend).squeeze(-1); trend = trend.squeeze(-1)
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)

class TS_Dataset(Dataset):
    def __init__(self, df_data, input_len, pred_len, feature_names):
        self.data = df_data.values
        self.il = input_len
        self.pl = pred_len
        self.feature_names = feature_names
    def __len__(self): return len(self.data)-self.il-self.pl+1
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.il]
        # OT 특성의 인덱스를 동적으로 찾기
        ot_index = self.feature_names.index('OT')
        y = self.data[idx+self.il:idx+self.il+self.pl, ot_index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

print("✅ [5단계] 모델 및 데이터셋 클래스 정의 완료.")


# ──────────────────────────────────────────────────────────────
# 6. 실험 설정 (*** 특성 공학 실험 1개 추가 ***)
# ──────────────────────────────────────────────────────────────
pred_len = 96
feature_dim = len(feature_cols) # 특성 개수를 변수로 저장

experiment_configs = [
    # --- 기존 실험 12개 ---
    {'exp_name': 'DLinear_len96', 'criterion': nn.MSELoss()},
    {'exp_name': 'DLinear_len168', 'criterion': nn.MSELoss()},
    {'exp_name': 'DLinear_len336_base', 'criterion': nn.MSELoss()},
    {'exp_name': 'DLinear_len504', 'criterion': nn.MSELoss()},
    {'exp_name': 'DLinear_len720', 'criterion': nn.MSELoss()},
    {'exp_name': 'DLinear_ks15', 'criterion': nn.MSELoss()},
    {'exp_name': 'DLinear_ks49', 'criterion': nn.MSELoss()},
    {'exp_name': 'DLinear_drop0.2', 'criterion': nn.MSELoss()},
    {'exp_name': 'DLinear_lr5e-5', 'criterion': nn.MSELoss()},
    {'exp_name': 'DLinear_best_combo', 'criterion': nn.MSELoss()},
    {'exp_name': 'DLinear_len504_MAE', 'criterion': nn.L1Loss()},
    {'exp_name': 'DLinear_best_combo_MAE', 'criterion': nn.L1Loss()},

    # --- (신규) 특성 공학 실험 ---
    {'exp_name': 'DLinear_FeatEng_best_combo', 'criterion': nn.MSELoss()},
]

# 기본 파라미터 딕셔너리
default_params = {
    'DLinear_len96': {'input_len': 96, 'kernel_size': 25, 'dropout_rate': 0.1, 'lr': 1e-4},
    'DLinear_len168': {'input_len': 168, 'kernel_size': 25, 'dropout_rate': 0.1, 'lr': 1e-4},
    'DLinear_len336_base': {'input_len': 336, 'kernel_size': 25, 'dropout_rate': 0.1, 'lr': 1e-4},
    'DLinear_len504': {'input_len': 504, 'kernel_size': 25, 'dropout_rate': 0.1, 'lr': 1e-4},
    'DLinear_len720': {'input_len': 720, 'kernel_size': 25, 'dropout_rate': 0.1, 'lr': 1e-4},
    'DLinear_ks15': {'input_len': 336, 'kernel_size': 15, 'dropout_rate': 0.1, 'lr': 1e-4},
    'DLinear_ks49': {'input_len': 336, 'kernel_size': 49, 'dropout_rate': 0.1, 'lr': 1e-4},
    'DLinear_drop0.2': {'input_len': 336, 'kernel_size': 25, 'dropout_rate': 0.2, 'lr': 1e-4},
    'DLinear_lr5e-5': {'input_len': 336, 'kernel_size': 25, 'dropout_rate': 0.1, 'lr': 5e-5},
    'DLinear_best_combo': {'input_len': 504, 'kernel_size': 25, 'dropout_rate': 0.1, 'lr': 5e-5},
    'DLinear_len504_MAE': {'input_len': 504, 'kernel_size': 25, 'dropout_rate': 0.1, 'lr': 1e-4},
    'DLinear_best_combo_MAE': {'input_len': 504, 'kernel_size': 25, 'dropout_rate': 0.1, 'lr': 5e-5},
    'DLinear_FeatEng_best_combo': {'input_len': 504, 'kernel_size': 25, 'dropout_rate': 0.1, 'lr': 5e-5},
}

# 각 실험 설정에 기본 파라미터 병합
for config in experiment_configs:
    config.update(default_params[config['exp_name']])

print(f"✅ [6단계] 총 {len(experiment_configs)}개의 실험 설정을 완료했습니다.")


# ──────────────────────────────────────────────────────────────
# 7. DLinear 모델 훈련, 평가, 예측 통합 루프
# ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

all_results = {}

for config in experiment_configs:
    exp_name = config['exp_name']
    current_input_len = config['input_len']

    print(f"\n" + "="*80 + f"\n▶ 실험 [{exp_name}] 시작\n" + "="*80)
    param_to_print = {k: v for k, v in config.items() if k not in ['criterion', 'exp_name']}
    print(f"   - 파라미터: {param_to_print}")
    print(f"   - 손실 함수: {config['criterion'].__class__.__name__}")

    # --- 데이터로더 ---
    train_df = scaled_df[scaled_df.index < train_end_date]
    # *** TS_Dataset에 feature_cols 전달 ***
    full_dataset = TS_Dataset(train_df, current_input_len, pred_len, feature_cols)
    val_split_idx = int(len(full_dataset) * 0.8)

    train_loader = DataLoader(Subset(full_dataset, range(val_split_idx)), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, range(val_split_idx,len(full_dataset))), batch_size=32)

    # --- 모델, 옵티마이저, 손실 함수 ---
    # *** 모델 생성 시 feature_dim 전달 ***
    model = DLinear_Leakproof(**config, pred_len=pred_len, feature_dim=feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    criterion = config['criterion']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # --- 훈련 루프 ---
    best_val_loss = float('inf')
    early_stop_counter = 0
    model_path = os.path.join(output_directory, f"model_{exp_name}.pth")

    for epoch in range(1, 201):
        model.train()
        train_losses = []
        for x,y in train_loader:
            x,y=x.to(device),y.to(device);optimizer.zero_grad();p=model(x);l=criterion(p,y);l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0);optimizer.step()
            train_losses.append(l.item())
        avg_train_loss = np.mean(train_losses)

        model.eval(); val_losses=[]
        with torch.no_grad():
            for x,y in val_loader:x,y=x.to(device),y.to(device);p=model(x);val_losses.append(criterion(p,y).item())
        avg_val_loss = np.mean(val_losses)

        if (epoch-1) % 10 == 0 or epoch == 1:
            print(f"  [Ep {epoch:03d}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss, early_stop_counter = avg_val_loss, 0
            torch.save(model.state_dict(), model_path)
        else:
            early_stop_counter += 1

        if early_stop_counter >= 30:
            print(f"  조기 종료. (Epoch {epoch})")
            break
        scheduler.step()

    print(f"  최적 모델 저장 완료: {model_path} (Best Val Loss: {best_val_loss:.6f})")

    # --- 예측 파일 생성 ---
    print(f"\n▶ 예측 파일 생성 중...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    submit_df_raw = pd.read_csv(sample_submission_path)
    results = []
    for _, row in submit_df_raw.iterrows():
        t0=pd.to_datetime(row[submit_df_raw.columns[0]]); input_end,input_start=t0-timedelta(hours=1),t0-timedelta(hours=current_input_len)
        seq=scaled_df.loc[input_start:input_end].values
        x_tensor=torch.tensor(seq,dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad(): pred_scaled=model(x_tensor).cpu().numpy().flatten()
        pred_unnorm = ot_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        results.append([row[submit_df_raw.columns[0]]] + pred_unnorm.tolist())

    col_names=[submit_df_raw.columns[0]] + [f"T{i}" for i in range(pred_len)]; df_submit=pd.DataFrame(results,columns=col_names)
    submission_filename = os.path.join(output_directory, f"submission_{exp_name}.csv")
    df_submit.to_csv(submission_filename, index=False)
    print(f"✅ 예측 파일 생성 완료: {submission_filename}")

    # --- 성능 평가 및 시각화 ---
    print(f"\n▶ 성능 평가 및 시각화 중...")
    y_pred_flat = df_submit.drop(columns=[submit_df_raw.columns[0]]).values.flatten()
    submit_times = pd.to_datetime(df_submit[submit_df_raw.columns[0]].values)
    gt_ots = []
    for t in submit_times:
        gt_range = pd.date_range(start=t, periods=pred_len, freq='H')
        gt_series = df.reindex(gt_range)['OT']
        gt_ots.append(gt_series.values)
    gt_ots_flat = np.concatenate(gt_ots)
    valid_indices = ~np.isnan(gt_ots_flat)
    gt_ots_clean = gt_ots_flat[valid_indices]
    y_pred_clean = y_pred_flat[valid_indices]

    def print_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(f"  【 최종 성능 지표 】")
        print(f"  - MSE  : {mse:.4f} (Kaggle 평가지표)")
        print(f"  - MAE  : {mae:.4f}")
        print(f"  - RMSE : {rmse:.4f}")
        print(f"  - R²   : {r2:.4f}")
        return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

    metrics = print_metrics(gt_ots_clean, y_pred_clean)
    result_log = config.copy()
    result_log['criterion'] = config['criterion'].__class__.__name__
    result_log.update(metrics)
    all_results[exp_name] = result_log


    plt.figure(figsize=(12, 7))
    plt.hist(gt_ots_clean, bins=50, alpha=0.7, label='Ground Truth', color='royalblue')
    plt.hist(y_pred_clean, bins=50, alpha=0.7, label='Prediction', color='orangered')
    plt.title(f"Prediction vs. Ground Truth Distribution ({exp_name})", fontsize=16)
    plt.xlabel("OT Value", fontsize=12); plt.ylabel("Frequency", fontsize=12)
    plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    hist_filename = os.path.join(output_directory, f"hist_{exp_name}.png")
    plt.savefig(hist_filename)
    plt.close()
    print(f"✅ 비교 히스토그램 저장 완료: {hist_filename}")

# ──────────────────────────────────────────────────────────────
# 8. 최종 결과 요약
# ──────────────────────────────────────────────────────────────
print("\n" + "="*80 + "\n▶ 최종 실험 결과 요약\n" + "="*80)
results_df = pd.DataFrame.from_dict(all_results, orient='index')
results_df.index.name = 'Experiment Name'

# 보기 좋게 컬럼 순서 정리
ordered_cols = ['input_len', 'kernel_size', 'dropout_rate', 'lr', 'criterion', 'val_loss', 'MSE', 'MAE', 'RMSE', 'R2']
final_cols = [col for col in ordered_cols if col in results_df.columns]
results_df = results_df[final_cols]
results_df = results_df.sort_values(by='MSE')

# 소수점 정리
pd.set_option('display.float_format', '{:.4f}'.format)

print(results_df)

best_exp = results_df.index[0]
best_mse = results_df.iloc[0]['MSE']
print(f"\n🏆 최적 실험: [{best_exp}] (Test MSE: {best_mse:.4f})")
print("✅ 모든 작업이 성공적으로 완료되었습니다.")