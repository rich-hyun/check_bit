# Triggers (`checkbit_ai/triggers`)

버전 `tr-v1` (2026-11-23 확정). 트리거는 feature 풀(`docs/FEATURES.md`) 위에서 탐지기 3종을 매분 평가하고, 탐지된 분을 **후보 구간**으로 묶는다. 닫힌 후보 구간이 에이전트 대기열이 된다. 결과는 `data/features.sqlite3`에 저장한다.

## 실행

- 실시간: feature 서비스(`FeatureService`)가 feature를 확정한 직후 같은 스레드에서 실행한다. LSTM 결과를 기다리기 위해 **1분 늦게** 판정한다. `CHECKBIT_TRIGGERS=0`이면 끈다.
- 리플레이: `python -m checkbit_ai.triggers.replay [--from … --to …] [--report-only]`. 범위 안의 탐지기 출력을 다시 계산하고, 후보 구간은 처음부터 다시 만든다. 같은 구간이 다시 만들어지면 `agent_status`는 유지한다. 일별 리포트에서 점프 판정(Gumbel)과 BH-FDR을 비교한다.
- 실시간과 리플레이는 같은 함수(`process_triggers`, `candidates.step`)를 쓰고, 결과가 같다는 것을 테스트한다.

## 탐지기 출력 형식 (통일)

```json
{"detector": "jump | robust_z | lstm", "score": 5.2, "threshold": 4.6, "is_anomaly": true, "evidence": [...]}
```

`score / threshold`를 "임계값 대비 배수"로 쓴다(후보 강도 판정).

## 탐지기

| 탐지기 | 판정 | score / threshold | 비고 |
|---|---|---|---|
| `jump` (Lee-Mykland 2008) | L = r / √bipower_var. (\|L\| − Cₙ)/Sₙ > β*, α = 1%/일, n = 1440 | Gumbel 통계량 / β* (≈ 4.60) | c·L(≈ N(0,1)), p값을 evidence에 기록. 공백 분, bipower NaN이면 평가하지 않음 |
| `robust_z` | feature 풀의 `z`(장기, 없으면 단기)를 방향별로 \|z\| ≥ 4 | max(방향 z / 임계값) / 1.0 | 방향표는 `params.Z_DIRECTIONS`. 체결 20건 미만이면 비율 feature 제외, `low_snapshots`면 호가 feature 제외. price_return_1m, bipower_var는 제외 |
| `lstm` | 팀원 LSTM AE의 `analyses` 결과에서 score > threshold | 모델 score / 모델 threshold | 수집 공백 분의 결과는 무시. 모델이 없으면 출력 없음 |

## 후보 구간

- 탐지된 분 사이에 탐지 없는 분이 2분 이하면 하나로 묶는다. 3분 연속 탐지가 없으면 닫는다. 구간이 60분에 이르면 닫고 새로 시작한다.
- **strong**: 탐지기 2종 이상이 걸렸거나, 하나라도 임계값의 1.5배 이상. 나머지는 **ambiguous**.
- 유형 힌트(`types`): z에서 걸린 feature의 L2 유형(거래량 급증, 유동성 경색, 호가 불균형, 한쪽 쏠림 체결, 체결 규모 분포 이탈, 가격 급변), 점프, LSTM. 최종 유형은 에이전트가 판단한다.
- `lstm_minutes`: 탐지된 분 중 LSTM 결과가 있던 분 수. LSTM이 없으면 2종 이상이 될 기회가 줄어든다.

## 알려진 점검 사항 (데이터가 쌓이면)

1. **점프 거짓 경보율**: 정규분포 시뮬레이션(변동성 일정)에서 K = 120일 때 하루 거짓 경보율이 목표 1%가 아니라 약 2.2%다. σ̂ 추정 오차 때문이다(K = 240이면 1.3%, K = 480이면 1.0%). K 민감도 점검 때 실제 데이터로 K 또는 β를 정한다.
2. **robust z 임계값**: 고정 4로 시작했다. 3일 이상 쌓이면 feature별 초과 비율을 측정해 `params.Z_THRESHOLDS`로 보정한다. 보정 전에는 에이전트를 연결하지 않는다.
3. **LSTM 경보율**: AWS 운영 데이터에서 LSTM이 분석한 분의 약 27%를 이상으로 판정한다(2026-11-23 기준 16,541분 중 4,505분). 지금 설정으로는 LSTM 혼자 후보를 계속 연다. 결정 필요.
