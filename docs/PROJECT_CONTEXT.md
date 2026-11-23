# Project Context — 업비트 실시간 시장 이상탐지 에이전트 (졸업프로젝트)

> 이 파일은 claude.ai 대화에서 정리한 설계 결정과 작업 계획이다. 코드 작업 시 이 내용을 기준으로 한다.

## 1. 목표와 범위

- 업비트 실시간 데이터에서 이상 구간을 탐지하고, **에이전트가 근거를 모아 판정을 제시하면 사람이 최종 판단**하는 모니터링 시스템.
- 범위는 "불공정거래 탐지"가 아니라 **관측 가능한 시장 이상 탐지와 근거 제공**. 시세조종 같은 법적 이상은 의도로 정의되고 계정·주문 데이터가 필요하므로 판단하지 않는다.
- 문제 정의는 forecasting이 아니라 **Anomaly Detection**. 예측 모델(Chronos, TimesFM)은 예측 잔차를 내는 탐지기 도구로만 쓴다.
- 실시간 검증은 KRW-BTC, 확장은 같은 업비트 API의 알트코인. 정량 평가는 라벨 있는 공개 벤치마크(TSB-AD, UCR)로.

## 2. 이상의 정의 (3층)

| 층 | 정의 | 라벨 | 담당 |
|---|---|---|---|
| L1 통계적 사건 | 가격 점프(Lee-Mykland, FDR 통제), 거래량·체결수·스프레드 극단값 | 자동 | 트리거 |
| L2 유형 | 점프, 거래량 급증, 유동성 경색(스프레드 확대), 호가 불균형, 한쪽 쏠림 체결(buy_ratio), 체결 규모 분포 이탈 | 규칙 자동 부여 | 에이전트 |
| L3 맥락 판정 | 뉴스·공지로 설명되는 이상인가 | 사람 판정 | 사람 |

- 가격은 점프 검정, 나머지 feature는 **롤링 robust z-score**(중앙값/MAD, 현재 시점은 윈도우에서 제외).
- 전체 기간 정규화 금지(미래 정보 누수). 모든 정규화는 인과적(causal) 롤링.
- 업비트 시장경보 플래그(market_event)는 일 단위 기준이라 BTC에는 거의 안 걸림 → 알트코인 평가 기준으로만 사용, 학습에는 사용 금지.

## 3. 시스템 구조

1. **수집**: 업비트 WebSocket 체결·호가 → SQLite
2. **Feature**: 1분 집계. price_return_1m/5m/15m, volatility, volume_change_rate, volume_zscore, trade_count, average_trade_size, buy_ratio, bid_ask_spread, order_imbalance, moving_average_gap
3. **트리거**: 가벼운 탐지기(롤링 z-score, Isolation Forest, 점프 검정). recall 위주. 하나라도 넘으면 후보 구간 생성. 탐지기끼리 엇갈리면 "애매함"으로 표시
4. **에이전트**: 트리거된 구간만 비동기로 처리. LLM에 원시 숫자를 주지 말고 도구가 계산한 결과를 준다
   - 도구: feature 구간 조회, 탐지기 점수, 변동성 분위(regime), 차트 렌더링(PNG, 2단계), 과거 유사 구간+사람 판정 검색, 뉴스/공지 검색
   - 도구 입출력 형식은 도메인 무관하게 통일: 탐지기 `{score, is_anomaly, evidence}`, 맥락 `{time, source, summary, before_or_after_event}`
   - 숫자는 Dual-Reference Bucketing으로 해석 가능한 구간값으로 변환(장기 기준·단기 기준 각각 중앙값/IQR 정규화 → "평소 대비 매우 높음" 등)
   - 출력 JSON: `{is_anomaly, type, severity, confidence, score(연속), evidence[], explanation}`
5. **대시보드**: 알림 → 차트 → 이상 전후 시각이 표시된 뉴스 → 에이전트 판정과 근거(원자료로 링크) → 사람 판정 버튼. 사람 판정은 라벨 테이블에 저장

## 4. 현재 레포 상태 (2026-11-23 기준)

main(PR #1·#2, 2026-11-12 머지)과 브랜치 `feat/raw-sqlite-store`를 합친 상태다.

- **수집**: `collector.py`(UpbitCollector)가 두 곳에 함께 쓴다.
  - 체결 CSV(`data/btc_anomaly_dataset.csv`): 업비트 `trade_timestamp`·`sequential_id` 포함. 호가는 체결 수신 시점의 1단계만 붙는다.
  - 원시 SQLite(`data/raw.sqlite3`, `checkbit_ai/rawdb.py`): 전체 체결(영구 중복 제거), 15단계 호가 스냅샷(1초 샘플링), `collector_sessions`(연결 구간 = 수집 커버리지).
- **탐지**: `checkbit_ai/`의 LSTM AutoEncoder(NumPy). CSV에서 1분 봉과 특징량 11개를 만들고, 1분마다 분석해 `data/checkbit.sqlite3`에 이력과 근거(`explain.py`, 규칙 기반)를 저장한다. 사람 판정(normal/anomaly/hold)도 같은 DB에 저장한다.
  - 이 LSTM AE는 3절의 "트리거 탐지기" 중 하나로 본다.
- **운영**: AWS Lightsail(`checkbit.site`)에서 2026-11-13부터 자동 수집 중이다. 배포는 main push 시 GitHub Actions로 이루어진다. 원시 SQLite는 이 브랜치가 배포된 뒤부터 쌓인다.
- **에이전트가 쓰기 전에 알아야 할 한계**
  - 특징량 생성(`features.py`)은 5분 이하 공백을 거래량 0으로 채운다. `collector_sessions`는 아직 참조하지 않는다.
  - 호가 특징량은 체결 수신 시점의 1단계 스냅샷 1개 기준이다. 15단계·분 내부 통계는 원시 SQLite에서 새로 계산해야 한다.
  - 정규화는 학습 구간의 평균·표준편차(모델 입력용) 하나뿐이다. 롤링 robust 기준(Dual-Reference Bucketing용)은 없다.
- **에이전트(3절 4번)는 아직 없다.** 트리거 묶음(후보 구간), 도구, 판정 JSON, `/agent/verdicts`도 없다.

## 5. 작업 순서

1. ~~collector 재작성~~ → main collector에 원시 SQLite 추가 저장 (브랜치 `feat/raw-sqlite-store`). 배포 후 15단계 호가와 sessions 수집 시작
   - 후속: 서버 체결 CSV(2026-11-13~)를 `raw.sqlite3` `trades`로 가져오는 일회성 명령
2. 원시 SQLite 기반 feature 풀(6축: 가격, 변동성, 거래 활동, 유동성, 주문 흐름, 체결 구성) + 롤링 robust 정규화. 수집 공백은 `collector_sessions`로 판정
3. 트리거: LSTM AE 래핑 + 롤링 z + Lee-Mykland 점프 검정 → 후보 구간, 탐지기 불일치는 "애매함"
4. `agent/`: 도구(입출력 형식 통일), 숫자 해석층, 판정 JSON, 비동기 워커, 저장, `/agent/verdicts`. 궤적 로깅(이후 SFT/GRPO 데이터). 착수 전 결정: LLM 종류, 뉴스·공지 소스
5. 대시보드 판정 패널 + 사람 판정을 에이전트 판정(후보 구간)에 연결

## 6. 학습 계획 (나중 단계)

- 판정 모델만 학습(AnomaMind 방식). 범용 LLM이 도구 호출, 소형 모델(Qwen 7~8B, LoRA)이 판정.
- SFT 워밍업 후 GRPO(verl). H200 2대.
- reward: 형식 + 정확도(F1/IoU) − 오탐 + 도구 호출 비용 + 정답 게이트(판정이 맞을 때만 도구 reward). ADEPT식 그룹 통계 게이트는 비교 실험 대상.

## 7. 참고 논문

- AnomaMind (arXiv 2602.13807, 프리프린트, 코드 github.com/Xiaoyu-Tao/AnomaMind-TS) — 설계 골격
- TSAD-Agents (WWW 2026), SAGE (arXiv 2605.05725), Zhou & Yu (ICLR 2025), ADEPT (ICML 2026), Park et al. (NeurIPS 2025), Lee & Mykland (RFS 2008)
