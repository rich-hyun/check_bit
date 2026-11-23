# Feature pool (`checkbit_ai/featurepool`)

정의 버전 `fp-v1`. 이 파일은 `python -m checkbit_ai.featurepool.docs`로 생성한다 (직접 수정 금지).

## 공통 규칙

- 버킷: 업비트 `trade_timestamp`/호가 `timestamp` 기준 UTC 1분 `[t, t+60s)`. t+60s+3s에 확정.
- 수집 공백: `collector_sessions`로 덮이지 않은 시간이 3초를 넘는 분. 그 분의 모든 feature는 NaN.
- 보조 점검: 스냅샷 수가 예상치의 절반 미만이면 `feature_minutes.low_snapshots = 1` (값은 계산).
- close_t: 그 분 마지막 체결가. 체결이 없으면 최대 60분 전까지의 마지막 체결가 (공백을 넘지 않음). r_t = ln(close_t / close_(t−1)).
- 롤링 feature는 윈도우 안에 공백이 하나라도 있으면 NaN.

## 정규화

- robust z = (x − median) / max(1.4826 × MAD, MAD 하한). 현재 분은 윈도우에서 제외.
- 단기: 직전 60분 (유효값 30개 이상), 장기: 직전 1440분 (유효값 720개 이상).
- `z` = 장기 z, 장기가 NaN이면 단기 z.

## Feature

| 축 | feature | 단위 | 정의 | NaN 조건 | MAD 하한 |
|---|---|---|---|---|---|
| 가격 | `price_return_1m` | log | ln(close_t / close_{t-1}) | 양 끝 close 미정의 | 1e-05 |
| 가격 | `price_return_5m` | log | ln(close_t / close_{t-5}) | 양 끝 close 미정의 | 1e-05 |
| 가격 | `price_return_15m` | log | ln(close_t / close_{t-15}) | 양 끝 close 미정의 | 1e-05 |
| 가격 | `moving_average_gap` | ratio | close_t / mean(close_{t-29..t}) − 1 (SMA 30분) | 30분 중 close 미정의(공백 포함) | 1e-05 |
| 변동성 | `price_range_1m` | log | ln(high_t / low_t) | 체결 0건 | 1e-05 |
| 변동성 | `volatility` | log | std(r_{t-14..t}), ddof=1 (현재 포함 15개) | 15개 수익률 중 미정의 | 1e-06 |
| 변동성 | `bipower_var` | log² | Lee-Mykland σ̂²: Σ_(j=t-K+2)^(t-1) \|r_j\|\|r_(j-1)\| / (K−2), K=120 (현재 r_t 제외) | r_(t-K+1..t-1) 중 미정의 | 1e-10 |
| 거래 활동 | `volume` | BTC | Σ trade_volume | 공백 | 0.001 |
| 거래 활동 | `volume_krw` | KRW | Σ trade_price × trade_volume | 공백 | 100000 |
| 거래 활동 | `trade_count` | 건 | 체결 건수 | 공백 | 1 |
| 거래 활동 | `volume_change_rate` | log | ln(volume_t + ε) − ln(volume_(t-1) + ε), ε=0.001 BTC | t 또는 t−1 공백 | 0.01 |
| 유동성 | `spread_mean_bp` | bp | 스냅샷별 (ask₁−bid₁)/mid×10⁴ 의 평균 | 스냅샷 0개 | 0.1 |
| 유동성 | `spread_max_bp` | bp | 스냅샷별 스프레드의 최대 (1초 샘플링 범위 내) | 스냅샷 0개 | 0.1 |
| 유동성 | `spread_wide_ratio` | ratio | 스프레드 > 1틱 인 스냅샷 비율 (틱 = 그 스냅샷의 인접 단계 간격 최솟값) | 스냅샷 0개 또는 틱 미정 | 0.01 |
| 유동성 | `depth_total_15` | BTC | 스냅샷별 Σ(15단계 bid_size + ask_size)의 평균 (업비트 total_*_size 아님) | 스냅샷 0개 | 0.01 |
| 유동성 | `amihud` | log / 1억원 | \|price_return_1m\| / (volume_krw / 10⁸) | 거래대금 0 또는 수익률 미정의 | 1e-06 |
| 주문 흐름 | `buy_ratio` | ratio | BID 거래량 / 전체 거래량 (taker 기준) | 체결 0건 | 0.01 |
| 주문 흐름 | `net_taker_volume` | BTC | BID 거래량 − ASK 거래량 | 공백 | 0.001 |
| 주문 흐름 | `order_imbalance_5` | ratio | 스냅샷별 (Σbid₁₋₅ − Σask₁₋₅)/(Σbid₁₋₅ + Σask₁₋₅)의 평균 | 스냅샷 0개 | 0.01 |
| 주문 흐름 | `roll_measure` | bp | 2√(−Cov(Δp_i, Δp_(i−1))) / mean(p) × 10⁴, 직전 30분(t 포함) 체결 단위 가격 변화 | 윈도우 중 공백, 가격 변화 < 200건, 또는 Cov ≥ 0 | 0.1 |
| 체결 구성 | `average_trade_size` | BTC | volume / trade_count | 체결 0건 | 0.0001 |
| 체결 구성 | `large_trade_ratio` | ratio | 1건 금액 ≥ 10,000,000원 체결의 거래량 / 전체 거래량 | 체결 0건 | 0.01 |
| 체결 구성 | `round_number_ratio_btc` | ratio | 수량이 0.001 BTC 배수인 체결 건수 비율 | 체결 0건 | 0.01 |
| 체결 구성 | `round_number_ratio_krw` | ratio | 금액이 10,000원 배수(±1원)인 체결 건수 비율 | 체결 0건 | 0.01 |
