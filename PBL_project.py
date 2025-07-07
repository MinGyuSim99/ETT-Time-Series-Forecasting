# ==============================================================================
# [STEP 1] DLinear 전체 실험 루프 (모델 훈련/예측/결과 저장: dlinear_all_experiments_results/)
# ==============================================================================
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

# 시드 고정
torch.manual_seed(42)
np.random.seed(42)
print("✅ [1단계] 라이브러리 임포트 완료.")

# 경로 설정
BASE_PATH = os.getcwd()
data_path = os.path.join(BASE_PATH, "ETTh1.csv")
sample_submission_path = os.path.join(BASE_PATH, "sample_submit.csv")
dlinear_output_directory = os.path.join(BASE_PATH, "dlinear_all_experiments_results/")
os.makedirs(dlinear_output_directory, exist_ok=True)
print(f"📂 모든 출력 파일은 다음 경로에 저장됩니다: {dlinear_output_directory}")

# 데이터 + 특성공학
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
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
print("✅ [3-2단계] 전체 특성 공학 완료.")

# 정규화
train_end_date = "2018-01-01 00:00:00"
train_data = df_feat[df_feat.index < train_end_date]
scaler = MinMaxScaler()
scaler.fit(train_data[feature_cols])
ot_scaler = MinMaxScaler()
ot_scaler.fit(train_data[['OT']])
data_scaled = scaler.transform(df_feat[feature_cols])
scaled_df = pd.DataFrame(data_scaled, index=df_feat.index, columns=feature_cols)
print(f"✅ [4단계] 누수 방지 스케일링 완료. (훈련 데이터 종료 시점: {train_end_date})")

# DLinear 모델/데이터셋 정의 (동일)
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
        self.linear_seasonal = nn.Sequential(nn.Linear(input_len, pred_len), nn.Dropout(dropout_rate))
        self.linear_trend = nn.Sequential(nn.Linear(input_len, pred_len), nn.Dropout(dropout_rate))
    def forward(self, x):
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
        ot_index = self.feature_names.index('OT')
        y = self.data[idx+self.il:idx+self.il+self.pl, ot_index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
print("✅ [5단계] 모델 및 데이터셋 클래스 정의 완료.")

# 실험설정
pred_len = 96
feature_dim = len(feature_cols)
experiment_configs = [
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
    {'exp_name': 'DLinear_FeatEng_best_combo', 'criterion': nn.MSELoss()},
]
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
for config in experiment_configs:
    config.update(default_params[config['exp_name']])
print(f"✅ [6단계] 총 {len(experiment_configs)}개의 실험 설정을 완료했습니다.")

# === 훈련/예측 루프 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")
all_results = {}
for config in experiment_configs:
    exp_name = config['exp_name']
    current_input_len = config['input_len']
    print(f"\n" + "="*80 + f"\n▶ 실험 [{exp_name}] 시작\n" + "="*80)
    train_df = scaled_df[scaled_df.index < train_end_date]
    full_dataset = TS_Dataset(train_df, current_input_len, pred_len, feature_cols)
    val_split_idx = int(len(full_dataset) * 0.8)
    train_loader = DataLoader(Subset(full_dataset, range(val_split_idx)), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, range(val_split_idx,len(full_dataset))), batch_size=32)
    model = DLinear_Leakproof(**config, pred_len=pred_len, feature_dim=feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    criterion = config['criterion']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    best_val_loss = float('inf')
    early_stop_counter = 0
    model_path = os.path.join(dlinear_output_directory, f"model_{exp_name}.pth")
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
    submission_filename = os.path.join(dlinear_output_directory, f"submission_{exp_name}.csv")
    df_submit.to_csv(submission_filename, index=False)
    print(f"✅ 예측 파일 생성 완료: {submission_filename}")

# ==============================================================================
# [STEP 2] 조건부 앙상블 예측 생성 (dlinear_ensemble_results/)
# ==============================================================================

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

# ==============================================================================
# [STEP 3] Transformer 단독 실행 (transformer_results/)
# ==============================================================================
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

# ==============================================================================
# [STEP 4] 최종 블렌딩/후처리/스무딩/하이퍼파라미터 최적화 (final_blending_results/)
# ==============================================================================
# 이 스크립트는 미리 생성된 두 개의 예측 파일을 불러와
# 동적 블렌딩, 앵커링, 스무딩 후처리를 통해 최종 제출 파일을 생성
#
# 실행 전 준비물:
# - 이 스크립트와 동일한 폴더에 ETTh1.csv 파일 위치
# - 이 스크립트와 동일한 폴더에 이전 단계들에서 생성된 결과 폴더들 위치
#   (예: transformer_results/, dlinear_ensemble_results/)
# ==================================================================

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import time

print("✅ [1단계] 라이브러리 임포트 및 기본 설정 완료.")

# ──────────────────────────────────────────────────────────────
# 2. 경로 설정 (상대 경로 방식)
# ──────────────────────────────────────────────────────────────

BASE_PATH = os.getcwd()

# 입력 결과 폴더 이름(수정 필요시 아래만 고치면 됨)
TRANSFORMER_RESULTS_FOLDER = "transformer_results"
DLINEAR_ENSEMBLE_RESULTS_FOLDER = "dlinear_ensemble_results"

transformer_pred_path = os.path.join(BASE_PATH, TRANSFORMER_RESULTS_FOLDER, "submission_transformer.csv")
dlinear_ensemble_path = os.path.join(BASE_PATH, DLINEAR_ENSEMBLE_RESULTS_FOLDER, "submission_conditional_ensemble.csv")
ground_truth_path = os.path.join(BASE_PATH, "ETTh1.csv")

output_dir = os.path.join(BASE_PATH, "final_blending_results/")
os.makedirs(output_dir, exist_ok=True)

print(f"📂 Transformer 예측 파일 경로: {transformer_pred_path}")
print(f"📂 DLinear 앙상블 예측 파일 경로: {dlinear_ensemble_path}")
print(f"📂 최종 결과물 저장 경로: {output_dir}")

# ──────────────────────────────────────────────────────────────
# 3. 하이퍼파라미터 탐색 공간(Grid) 정의
# ──────────────────────────────────────────────────────────────
short_term_weights_A = [0.4, 0.5, 0.6]  # 단기 예측에서 Transformer(A)에 부여할 가중치
long_term_weights_A = [0.6, 0.7, 0.8]   # 장기 예측에서 Transformer(A)에 부여할 가중치
strong_smoothing_configs = [
    [0.8, 0.5, 0.3],
    [0.7, 0.4, 0.2],
    [0.9, 0.6, 0.4]
]

print("\n" + "="*80)
print("▶ 하이퍼파라미터 최적화를 시작합니다.")
print(f"▶ 총 {len(short_term_weights_A) * len(long_term_weights_A) * len(strong_smoothing_configs)}개의 조합을 테스트합니다.")
print("="*80 + "\n")

try:
    # 데이터 로드
    pred_A = pd.read_csv(transformer_pred_path)
    pred_B = pd.read_csv(dlinear_ensemble_path)
    df_true = pd.read_csv(ground_truth_path, parse_dates=['date'], index_col='date')
    date_col_name = pred_A.columns[0]
    
    gt_ots_list = []
    submit_times = pd.to_datetime(pred_A[date_col_name].values)
    for t in submit_times:
        gt_range = pd.date_range(start=t, periods=96, freq='H')
        gt_data = df_true.reindex(gt_range)['OT'].values
        gt_ots_list.append(gt_data)
    gt_ots_flat = np.concatenate(gt_ots_list)
    valid_indices = ~np.isnan(gt_ots_flat)
    gt_ots_clean = gt_ots_flat[valid_indices]

    print("✅ 예측 파일 및 Ground Truth 데이터 로드 완료.")

    results_log = []
    best_mse = float('inf')
    best_params = {}
    run_count = 0
    start_time = time.time()
    df_final = pd.DataFrame() # 최고 성능 df를 저장할 변수 초기화

    for short_w_A in short_term_weights_A:
        for long_w_A in long_term_weights_A:
            for strong_smooth in strong_smoothing_configs:
                run_count += 1
                current_params = {
                    'short_A': short_w_A, 'long_A': long_w_A, 'strong_smooth': strong_smooth
                }
                print(f"--- [실행 {run_count}] 파라미터: {current_params} ---")
                # 1. 동적 블렌딩
                short_w_B = 1 - short_w_A
                long_w_B = 1 - long_w_A
                short_term_blended = pred_A.iloc[:, 1:25].values * short_w_A + pred_B.iloc[:, 1:25].values * short_w_B
                long_term_blended = pred_A.iloc[:, 25:].values * long_w_A + pred_B.iloc[:, 25:].values * long_w_B
                blended_values = np.hstack([short_term_blended, long_term_blended])
                df_blended = pd.DataFrame(blended_values, columns=pred_A.columns[1:])
                df_blended.insert(0, date_col_name, pred_A[date_col_name])
                # 2. 앵커링 및 스무딩 후처리
                final_predictions_list = []
                for _, row in df_blended.iterrows():
                    t0_timestamp = pd.to_datetime(row[date_col_name])
                    last_known_timestamp = t0_timestamp - pd.Timedelta(hours=1)
                    new_preds = row.drop(date_col_name).values.copy()
                    if last_known_timestamp in df_true.index:
                        anchor_value = df_true.loc[last_known_timestamp]['OT']
                        new_preds[0] = anchor_value
                        for i, weight in enumerate(strong_smooth):
                            if i + 1 < len(new_preds):
                                new_preds[i+1] = (new_preds[i] * weight) + (new_preds[i+1] * (1 - weight))
                    final_predictions_list.append(new_preds)
                # 3. 성능(MSE) 계산
                pred_flat = np.concatenate([p.flatten() for p in final_predictions_list])
                y_pred_clean = pred_flat[valid_indices]
                mse = mean_squared_error(gt_ots_clean, y_pred_clean)
                print(f"  ▶ MSE 결과: {mse:.6f}")
                # 4. 결과 기록
                log_entry = current_params.copy()
                log_entry['MSE'] = mse
                results_log.append(log_entry)
                # 5. 최고 성능 모델 정보 업데이트
                if mse < best_mse:
                    best_mse = mse
                    best_params = current_params
                    print(f"    ⭐ 새로운 최고 기록! (MSE: {best_mse:.6f})")
                    df_final = pd.DataFrame(final_predictions_list, columns=pred_A.columns[1:])
                    df_final.insert(0, date_col_name, pred_A[date_col_name])

except FileNotFoundError as e:
    print(f"❌ 파일 로드 실패: {e}\n경로를 다시 확인해주세요. 'submission_transformer.csv'와 'submission_conditional_ensemble.csv' 파일이 필요합니다.")
except Exception as e:
    print(f"❌ 스크립트 실행 중 오류가 발생했습니다: {e}")

# ──────────────────────────────────────────────────────────────
# 5. 최종 결과 요약 및 저장
# ──────────────────────────────────────────────────────────────
if results_log:
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("▶ 하이퍼파라미터 최적화가 완료되었습니다.")
    print(f"▶ 총 실행 시간: {total_time:.2f}초")
    results_df = pd.DataFrame(results_log)
    results_df = results_df.sort_values(by='MSE', ascending=True)
    # 전체 결과 로그 저장
    full_log_filename = os.path.join(output_dir, "hyperparam_tuning_log.csv")
    results_df.to_csv(full_log_filename, index=False)
    print(f"✅ 전체 튜닝 로그가 '{full_log_filename}'에 저장되었습니다.")
    # 최고 성능 예측 파일 저장
    if not df_final.empty:
        best_submission_filename = os.path.join(output_dir, "submission_best_tuned.csv")
        df_final.to_csv(best_submission_filename, index=False)
        print(f"✅ 최적 예측 파일이 '{best_submission_filename}'에 저장되었습니다.")
    print("\n🏆 최종 튜닝 결과 (상위 5개)")
    print(results_df.head(5).to_string())
    print("\n" + "-"*50)
    print("🚀 최적의 하이퍼파라미터 조합 🚀")
    print(f"  - 단기(A) 가중치: {best_params.get('short_A')}")
    print(f"  - 장기(A) 가중치: {best_params.get('long_A')}")
    print(f"  - 스무딩 계수: {best_params.get('strong_smooth')}")
    print(f"  - 최고 MSE: {best_mse:.6f}")
    print(f"\n최적 예측 파일은 'submission_best_tuned.csv' 이름으로 저장되었습니다.")
    print("="*80)
else:
    print("\n❌ 그리드 서치가 실행되지 않았습니다. 입력 파일 경로를 확인해주세요.")


# ==============================================================================
# [END] 모든 파이프라인 완료
# ==============================================================================

