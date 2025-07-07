# ==================================================================
# 최종 블렌딩 및 후처리(final.py) 단독 실행 스크립트 (경로 수정 최종본)
# ==================================================================
# 이 스크립트는 미리 생성된 두 개의 예측 파일을 불러와
# 동적 블렌딩, 앵커링, 스무딩 후처리를 통해 최종 제출 파일을 생성합니다.
#
# 실행 전 준비물:
# - 이 스크립트와 동일한 폴더에 ETTh1.csv 파일 위치
# - 이전 단계들에서 생성된 예측 결과 CSV 파일들
# ==================================================================

# ──────────────────────────────────────────────────────────────
# 1. 라이브러리 임포트 및 기본 설정
# ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import time

print("✅ [1단계] 라이브러리 임포트 및 기본 설정 완료.")

# ──────────────────────────────────────────────────────────────
# 2. 경로 설정 (상대 경로 방식으로 수정)
# ──────────────────────────────────────────────────────────────
BASE_PATH = os.getcwd()

TRANSFORMER_PRED_PATH = os.path.join(BASE_PATH, "transformer_results", "submission_transformer.csv")
DLINEAR_ENSEMBLE_PATH = os.path.join(BASE_PATH, "dlinear_ensemble_results", "submission_conditional_ensemble.csv")

# 정답 데이터 파일 경로 (보통 스크립트와 같은 위치)
GROUND_TRUTH_PATH = os.path.join(BASE_PATH, "ETTh1.csv")

# 최종 결과물이 저장될 디렉토리
output_dir = os.path.join(BASE_PATH, "final_blending_results/")
os.makedirs(output_dir, exist_ok=True)

print(f"📂 Transformer 예측 파일 경로: {TRANSFORMER_PRED_PATH}")
print(f"📂 DLinear 앙상블 예측 파일 경로: {DLINEAR_ENSEMBLE_PATH}")
print(f"📂 최종 결과물 저장 경로: {output_dir}")


# ──────────────────────────────────────────────────────────────
# 3. 하이퍼파라미터 탐색 공간(Grid) 정의
# ──────────────────────────────────────────────────────────────
# 블렌딩 가중치 탐색 범위
short_term_weights_A = [0.4, 0.5, 0.6]  # 단기 예측에서 Transformer(A)에 부여할 가중치
long_term_weights_A = [0.6, 0.7, 0.8]   # 장기 예측에서 Transformer(A)에 부여할 가중치

# 스무딩 가중치 탐색 범위 (가장 영향력이 큰 첫 번째, 두 번째 값 위주로 튜닝)
strong_smoothing_configs = [
    [0.8, 0.5, 0.3],
    [0.7, 0.4, 0.2],
    [0.9, 0.6, 0.4]
]

print("\n" + "="*80)
print("▶ 하이퍼파라미터 최적화를 시작합니다.")
print(f"▶ 총 {len(short_term_weights_A) * len(long_term_weights_A) * len(strong_smoothing_configs)}개의 조합을 테스트합니다.")
print("="*80 + "\n")


# ──────────────────────────────────────────────────────────────
# 4. 그리드 서치 실행 루프
# ──────────────────────────────────────────────────────────────
try:
    # 데이터 로드 (루프 시작 전 한 번만)
    pred_A = pd.read_csv(TRANSFORMER_PRED_PATH)
    pred_B = pd.read_csv(DLINEAR_ENSEMBLE_PATH)
    df_true = pd.read_csv(GROUND_TRUTH_PATH, parse_dates=['date'], index_col='date')
    date_col_name = pred_A.columns[0]
    
    # 실제 값(Ground Truth) 미리 준비
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
    df_final = pd.DataFrame()

    # 모든 하이퍼파라미터 조합에 대해 루프 실행
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
                
                # 2. 앵커링 및 스무딩 후처리
                final_predictions_list = []
                for i, row_values in enumerate(blended_values):
                    t0_timestamp = pd.to_datetime(pred_A.iloc[i, 0])
                    last_known_timestamp = t0_timestamp - pd.Timedelta(hours=1)
                    new_preds = row_values.copy()
                    
                    if last_known_timestamp in df_true.index:
                        anchor_value = df_true.loc[last_known_timestamp]['OT']
                        new_preds[0] = anchor_value
                        for j, weight in enumerate(strong_smooth):
                            if j + 1 < len(new_preds):
                                new_preds[j+1] = (new_preds[j] * weight) + (new_preds[j+1] * (1 - weight))
                    final_predictions_list.append(new_preds)

                # 3. 성능(MSE) 계산
                pred_flat = np.concatenate(final_predictions_list)
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
                    # 최고 성능일 때의 예측 파일 저장
                    df_final = pd.DataFrame(final_predictions_list, columns=pred_A.columns[1:])
                    df_final.insert(0, date_col_name, pred_A[date_col_name])

except FileNotFoundError as e:
    print(f"\n❌ 파일 로드 실패: {e}")
    print("  스크립트 상단의 경로 설정이 올바른지 다시 확인해주세요.")
    print(f"  - 확인 경로 1: {TRANSFORMER_PRED_PATH}")
    print(f"  - 확인 경로 2: {DLINEAR_ENSEMBLE_PATH}")
except Exception as e:
    print(f"❌ 스크립트 실행 중 오류가 발생했습니다: {e}")


# ──────────────────────────────────────────────────────────────
# 5. 최종 결과 요약
# ──────────────────────────────────────────────────────────────
if results_log:
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("▶ 하이퍼파라미터 최적화가 완료되었습니다.")
    print(f"▶ 총 실행 시간: {total_time:.2f}초")
    
    # 결과를 DataFrame으로 변환하고 MSE 기준으로 정렬
    results_df = pd.DataFrame(results_log)
    results_df = results_df.sort_values(by='MSE', ascending=True)
    
    # 전체 결과 로그 파일로 저장
    full_log_filename = os.path.join(output_dir, "hyperparam_tuning_log.csv")
    results_df.to_csv(full_log_filename, index=False)
    print(f"✅ 전체 튜닝 로그가 '{full_log_filename}'에 저장되었습니다.")
    
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