# CheckBit — 업비트 실시간 데이터 기반 비트코인 이상탐지

Upbit WebSocket 실시간 수집 → CSV 저장·전처리 → LSTM AutoEncoder 이상탐지 → FastAPI → 웹 대시보드 시각화 → 탐지 이력·근거·사용자 판정.

- 서비스: https://checkbit.site (Nginx: `/` 정적 파일, `/api/*` → FastAPI `127.0.0.1:8000`)
- 서버: AWS Lightsail Ubuntu · systemd `checkbit` · GitHub Actions 자동 배포 (`.github/workflows/deploy.yml`)

## 구조

```
main.py               FastAPI 앱 (수집 제어 /status /download + 라우터 연결, 시작 시 서비스 초기화)
collector.py          Upbit WebSocket 수집기 (재연결·중지·중복 방지·상태 확인)
market_dashboard.py   Upbit REST 시세 + 통계 밴드(참고용, 학습 모델 아님)  /dashboard/*
checkbit_ai/
  config.py           경로·환경변수
  rawstore.py         원시 CSV 저장 (마이그레이션, 손상 줄 복구, 잠금, 다운로드 스냅샷)
  features.py         전처리: 정제·중복 제거 → 1분 봉 → 특징량 → 정규화 → 시계열 윈도우
  lstm_ae.py          NumPy LSTM AutoEncoder (학습/추론/저장)
  baseline.py         통계 비교 모델, ROC-AUC 등 평가 함수
  train.py            오프라인 학습 CLI (보고서 생성)
  explain.py          규칙 기반 근거 설명
  store.py            SQLite 탐지 이력·사용자 판정
  service.py          실시간 추론 서비스 (1분마다 자동 분석)
  api.py              /ai/* API
  synthetic.py        테스트용 합성 데이터 생성기 (실제 데이터 아님)
index.html / app.js / style.css   대시보드 (기존 화면 + AI 이상탐지 섹션)
dev_server.py         로컬 개발용 (Nginx 구조 흉내: / 정적, /api FastAPI)
tests/                pytest (모두 합성 데이터 사용)
sample_data/          과거 저장소에 있던 실제 수집 CSV (2026-09-08 13:41~13:54, 약 13분)
models/lstm_ae/       배포용 학습 모델 (model.npz, meta.json, report.md) — 학습 후 커밋
data/                 런타임 데이터 (git 제외): 수집 CSV, checkbit.sqlite3, backup/
```

## 데이터와 특징량

수집 CSV 컬럼: `timestamp, market, trade_price, trade_volume, ask_bid, bid_price, ask_price, bid_size, ask_size, total_bid_size, total_ask_size, trade_timestamp, sequential_id`
(체결 1건당 1행, 수신 시점의 최우선 호가·총 잔량 스냅샷 포함. 마지막 두 컬럼은 이번에 추가 — 기존 11개 컬럼 순서는 동일)

전처리 (`features.py`):
1. 검증: 가격·수량 > 0, ask_bid ∈ {ASK,BID}, 시간 파싱, 다른 마켓 제외, 컬럼 수 오류 제외. 호가 0값은 "스냅샷 없음"(결측)으로 처리
2. 중복 제거: `sequential_id` (구 데이터는 완전 동일 행)
3. 시간: Upbit `trade_timestamp`(UTC ms) 우선, 구 데이터는 수신 시각
4. 1분 봉 집계: OHLC, 거래량, 거래대금, 매수/매도 체결량, 체결 건수, 분 마지막 호가 스냅샷
5. 결측 분: 5분 이하는 거래량 0·가격 유지로 보간, 그 이상 끊기면 구간(segment) 분리 — 윈도우는 구간을 넘지 않음
6. 특징량(11개, 모두 과거·현재 데이터만 사용):

| 특징량 | 의미 |
|---|---|
| log_return | 1분 로그 수익률 |
| hl_range | 1분 고저 변동폭 ln(high/low) |
| rv_10 | 최근 10분 수익률 표준편차(실현 변동성) |
| d_log_volume | 거래량 변화 (log1p 차분) |
| log_value / d_log_value | 거래대금 규모 / 변화 |
| log_trade_count | 체결 건수 |
| buy_sell_imbalance | (매수체결량−매도체결량)/합 |
| spread_bps | 호가 스프레드 (bp) |
| l1_imbalance | 최우선 호가 잔량 불균형 |
| depth_imbalance | 전체 호가 잔량 불균형 |

7. 정규화: z-score (평균·표준편차는 **학습 구간 데이터로만** 적합), ±8 클리핑
8. 시계열 윈도우: 길이 20분

## 모델

- 인코더 LSTM(24) → 마지막 은닉 상태 → 20스텝 반복 → 디코더 LSTM(24) → Linear(11). 파라미터 8,435개
- NumPy만으로 구현 (서버에 PyTorch 불필요, 추론 수 ms). BPTT 기울기는 수치 미분으로 검증(테스트 포함)
- 이상 점수 = 윈도우 **마지막 5분**의 평균 제곱 재구성 오차 (20분 전체 맥락으로 재구성, 최근 구간을 판정)
- 임계값 = 검증 점수 중앙값 + 4×MAD (검증 구간에 섞인 이상치에 둔감). 분위수(q=0.99) 방식도 선택 가능
- 분할: 시간순 70/15/15, 분할 사이 윈도우 겹침 제거(purging)
- 평가: 실제 이상 라벨이 없으므로 Accuracy/Precision/Recall은 **산출하지 않음**. 대신 재구성 오차 분포, 경보율, 통계 기준 모델 비교(Spearman, Jaccard), 합성 이상 주입 시험(ROC-AUC, 탐지율 — 인위적 교란에 대한 민감도) 기록. 사용자 판정이 쌓이면 `/ai/feedback/summary`에서 AI-사용자 일치율 확인

## 현재 배포 모델 (`models/lstm_ae/`)

- 버전 `lstm-ae-20261112T033914Z`, Upbit WebSocket으로 수집한 KRW-BTC 실데이터(2026-11-12 09:22~11:43 KST, 체결 12,542건, 1분 봉 123개)로 학습
- 윈도우 8분, 이상 점수 = 마지막 3분 재구성 오차, LSTM(16)→LSTM(16), 파라미터 4,091개, 시간순 분할 60/20/20 (학습 53 / 검증 10 / 테스트 12 윈도우)
- 임계값 2.49 (검증 중앙값 + 4·MAD), 검증·테스트 경보율 0%, 합성 교란 시험 AUC: ±2σ 0.944, ±3σ 1.000, ±5σ 1.000
- 학습 명령:
  `python -m checkbit_ai.train --csv data/upbit_live_20261112_0923.csv --csv data/btc_anomaly_dataset.csv --out models/lstm_ae --window 8 --score-steps 3 --hidden 16 --epochs 120 --split 0.6,0.2,0.2 --min-train-windows 40`
- 상세: `models/lstm_ae/report.md`. 수집 기간이 길어지면 아래 절차로 재학습하면 됩니다(기본값: 윈도우 20분, 70/15/15).

## 모델 학습 및 배포 (운영 서버에서 자동 학습하지 않음)

데이터는 서버에서 계속 수집됩니다. 최소 수 시간(권장 24시간 이상) 수집 후:

```bash
# 1) 서버 데이터 받기 (또는 scp로 data/btc_anomaly_dataset.csv 복사)
curl -o collected.csv https://checkbit.site/api/download
# 2) 로컬에서 학습 (기본: 학습 윈도우 200개 이상 필요 ≈ 5시간 이상의 연속 데이터)
pip install -r requirements.txt
python -m checkbit_ai.train --csv collected.csv --out models/lstm_ae
cat models/lstm_ae/report.md          # 학습·평가 결과 확인
# 3) 커밋·푸시 → GitHub Actions가 배포, 서비스 재시작 시 자동 로드
git add models/lstm_ae && git commit -m "Train LSTM AE model" && git push
```

서버에서 직접 학습할 경우(가벼운 모델이지만 CPU 사용, `nice` 권장):
`nice -n 10 venv/bin/python -m checkbit_ai.train --csv data/btc_anomaly_dataset.csv --out data/models/lstm_ae`
후 환경변수 `CHECKBIT_MODEL_DIR=/home/ubuntu/check_bit/data/models/lstm_ae` 설정 → 대시보드 관리자 기능의 "모델 다시 불러오기".

모델이 없으면 API·화면은 "모델 준비 안 됨"을 표시하며 분석 결과를 만들지 않습니다.

## API (Nginx 뒤에서는 `/api` 접두사)

| Method | Path | 설명 | 권한 |
|---|---|---|---|
| GET | `/status` | 수집 상태 (running, connected, receiving, rows, 마지막 체결, 재연결 수, 오류) | 공개 |
| POST | `/collector/start` · `/collector/stop` | 수집 시작/중지 (중지 시 WebSocket 실제 종료) | 관리자 |
| GET | `/download` | 수집 CSV 다운로드 (쓰기 중에도 완전한 행까지만) | 공개 |
| GET | `/dashboard/market` · `/dashboard/health` | 기존 시세·통계 밴드 | 공개 |
| GET | `/ai/model` | 모델 카드: 구조, 학습 데이터, 판정 기준, 평가 결과 | 공개 |
| GET | `/ai/status` | 모델 상태·버전·임계값, 버퍼, 최신 분석 시각, 이력 건수 | 공개 |
| POST | `/ai/analyze` | 최신 완성 1분까지로 즉시 분석 (분당 결과 1건으로 저장) | 공개(속도 제한) |
| GET | `/ai/analysis/latest` | 최근 분석 결과 | 공개 |
| GET | `/ai/timeline?minutes=180` | 1분 봉 가격·거래대금·특징량 + 이상 점수 | 공개 |
| GET | `/ai/detections?anomalies_only=true&limit=30` | 탐지 이력 | 공개 |
| GET | `/ai/detections/{id}` | 상세: 근거 설명, 기여 특징량, 사용자 판정 이력 | 공개 |
| POST | `/ai/detections/{id}/feedback` | 사용자 판정 `{"label":"normal|anomaly|hold","note":"..."}` | 공개(속도 제한) |
| GET | `/ai/feedback/summary` | AI 판정 vs 사용자 판정 비교 | 공개 |
| POST | `/ai/model/reload` | 모델 파일 다시 불러오기 + 미분석 구간 재분석 | 관리자 |

관리자 API는 `X-Admin-Token` 헤더가 환경변수 `CHECKBIT_ADMIN_TOKEN`과 일치해야 합니다. 토큰이 설정되지 않으면 관리 기능은 비활성화(403)됩니다. 모델 학습은 HTTP로 노출하지 않습니다.

## 서버 설정 (관리자가 1회 수행)

```bash
sudo systemctl edit checkbit
# [Service]
# Environment=CHECKBIT_ADMIN_TOKEN=<길고 임의의 문자열>
# Environment=CHECKBIT_COLLECTOR_AUTOSTART=1
sudo systemctl restart checkbit
```

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| CHECKBIT_ADMIN_TOKEN | (없음) | 관리 API 토큰 |
| CHECKBIT_COLLECTOR_AUTOSTART | 미설정 | `1`: 시작 시 수집, `0`: 수집 안 함, 미설정: 재시작 전 상태 복원 (`data/collector_state.json`) |
| CHECKBIT_DATA_DIR | `./data` | 수집 CSV·DB·백업 위치 |
| CHECKBIT_MODEL_DIR | `./models/lstm_ae` | 모델 위치 |
| CHECKBIT_REALTIME_ANALYSIS | `1` | 1분마다 자동 분석 |
| CHECKBIT_BUFFER_MINUTES | `720` | 메모리에 유지할 1분 봉 수 |
| CHECKBIT_RETENTION_DAYS | `30` | 이 기간이 지난 정상 판정 기록 삭제 (이상·판정 기록은 유지) |

Python 3.10 이상 필요. 추가 의존성은 `numpy` 하나입니다.

데이터 보존: 수집 CSV는 이제 git에서 제외된 `data/`에 저장됩니다(이전에는 저장소에 포함된 파일이라 배포의 `git reset --hard` 때마다 서버 수집분이 덮어써졌습니다). `deploy.yml`은 배포 전에 기존 루트 CSV를 `data/backup/`에 복사하고, 앱은 첫 시작 시 이를 `data/`로 가져옵니다.

## 로컬 실행·테스트

```bash
python -m venv .venv && .venv/Scripts/activate   # (Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
pytest -q                     # 합성 데이터로 전처리·모델·수집기(가짜 WebSocket 서버)·API 검증
python dev_server.py          # http://127.0.0.1:8080 (기본: 수집 상태 복원, 관리자 토큰은 환경변수)
python run.py                 # 기존 방식: API만 127.0.0.1:8000
```

합성 데이터로 화면 확인 (실제 데이터와 분리된 폴더 사용):

```bash
python -m checkbit_ai.synthetic --out demo/synthetic_trades.csv --minutes 900
python -m checkbit_ai.train --csv demo/synthetic_trades.csv --out demo/model --data-label synthetic
CHECKBIT_DATA_DIR=demo/data CHECKBIT_MODEL_DIR=demo/model python dev_server.py
```
합성 데이터로 학습한 모델은 대시보드에 "합성(시험용) 데이터로 학습" 경고가 표시됩니다.

## 한계

- 업비트가 과거 호가 데이터를 제공하지 않으므로 학습 데이터는 직접 수집한 기간에 한정됩니다.
- 실제 이상 라벨이 없어 지도학습 지표는 측정할 수 없습니다. 사용자 판정 데이터가 쌓이면 일부 검증이 가능합니다.
- 탐지 결과는 통계적 이상 징후이며 원인 단정·시세 예측·투자 조언이 아닙니다.
