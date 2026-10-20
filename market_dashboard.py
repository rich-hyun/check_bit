"""Read-only Bitcoin dashboard: Upbit market feed + replaceable detection adapter.

Set DETECTION_API_URL to send normalized candles to an external model. Its response
must contain a `bands` list (one band per candle, chronological). No external
service is required for the built-in, explicitly labelled statistical mock.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import statistics
import time
from datetime import datetime, timezone
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix='/dashboard', tags=['Dashboard'])
LOG = logging.getLogger('checkbit.dashboard')
UPBIT = 'https://api.upbit.com'
PERIODS = {'1H': (1, 65), '6H': (5, 78), '24H': (15, 100)}
CACHE_SECONDS = max(10, int(os.getenv('DASHBOARD_CACHE_SECONDS', '20')))
STALE_SECONDS = 180
DETECTION_API_URL = os.getenv('DETECTION_API_URL', '').strip()
# Optionally set a secret only on the backend. Never put it in HTML or JS.
DETECTION_API_KEY = os.getenv('DETECTION_API_KEY', '').strip()
_cached: dict[str, tuple[float, dict]] = {}
_lock = asyncio.Lock()


def calc_rsi(closes: list[float], period: int = 14) -> float | None:
    """Simple RSI over the most recent 14 closes, null if insufficient data."""
    if len(closes) < period + 1:
        return None
    differences = [closes[i] - closes[i-1] for i in range(len(closes)-period, len(closes))]
    gains = sum(max(0.0, v) for v in differences) / period
    losses = sum(max(0.0, -v) for v in differences) / period
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    return round(100.0 - 100.0/(1.0 + gains/losses), 1)


def _finite_positive(value: object, field: str) -> float:
    try:
        v = float(value)
    except (ValueError, TypeError):
        raise ValueError(f'{field}: invalid number') from None
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f'{field}: must be finite and positive')
    return v


def mock_bands(candles: list[dict]) -> list[dict]:
    """CAUSAL windowed median + MAD mock; never uses future prices.

    Inner interval: 3.0 x robust sigma (at least 0.12% of center)
    Outer interval: 4.7 x robust sigma (at least 0.22% of center)
    Labels are relative to this *illustrative* method, not a trained model.
    """
    closes: list[float] = []
    output: list[dict] = []
    for candle in candles:
        price = _finite_positive(candle['close'], 'close')
        history = closes[-24:]
        center = statistics.median(history) if history else price
        if len(history) >= 3:
            mad = statistics.median([abs(x - center) for x in history])
            robust_sigma = 1.4826 * mad
        else:
            robust_sigma = 0.0
        inner = max(3.0 * robust_sigma, center * 0.0012)
        outer = max(4.7 * robust_sigma, center * 0.0022, inner * 1.4)
        lower = max(0.01, center - inner)
        upper = center + inner
        warning_lower = max(0.01, center - outer)
        warning_upper = center + outer
        status = ('anomaly' if price < warning_lower or price > warning_upper else
                  'warning' if price < lower or price > upper else 'normal')
        output.append({'timestamp': candle['timestamp'], 'price': price,
                       'lower': round(lower, 2), 'upper': round(upper, 2),
                       'warning_lower': round(warning_lower, 2),
                       'warning_upper': round(warning_upper, 2), 'status': status})
        closes.append(price)
    return output


def validate_external_bands(candles: list[dict], bands: object) -> list[dict]:
    """Validate full response before ever displaying external model bands."""
    if not isinstance(bands, list) or len(bands) != len(candles):
        raise ValueError('bands must contain exactly one record per candle')
    output = []
    for candle, band in zip(candles, bands):
        if not isinstance(band, dict) or band.get('timestamp') != candle['timestamp']:
            raise ValueError('band timestamps must match candles in order')
        wl, lo, hi, wh = (_finite_positive(band[k], k) for k in
                          ('warning_lower', 'lower', 'upper', 'warning_upper'))
        if not (wl <= lo < hi <= wh):
            raise ValueError('invalid boundary ordering')
        price = _finite_positive(candle['close'], 'close')
        # Calculate status on the server to prevent inconsistent labels.
        status = ('anomaly' if price < wl or price > wh else
                  'warning' if price < lo or price > hi else 'normal')
        output.append({'timestamp': candle['timestamp'], 'price': price,
                       'lower': lo, 'upper': hi, 'warning_lower': wl,
                       'warning_upper': wh, 'status': status})
    return output


async def detect(candles: list[dict], period: str, interval: int) -> tuple[list[dict], str, str | None]:
    if not DETECTION_API_URL:
        return mock_bands(candles), 'mock', None
    payload = {'symbol': 'KRW-BTC', 'period': period, 'interval_minutes': interval,
               'candles': candles}
    headers = {'Authorization': f'Bearer {DETECTION_API_KEY}'} if DETECTION_API_KEY else {}
    try:
        # External models are called by the server, never from the browser.
        async with httpx.AsyncClient(timeout=6.0, headers=headers) as api:
            response = await api.post(DETECTION_API_URL, json=payload)
            response.raise_for_status()
            bands = validate_external_bands(candles, response.json().get('bands'))
            return bands, 'external', None
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        LOG.warning('External detector unavailable; using visible mock fallback: %s', exc)
        return mock_bands(candles), 'mock-fallback', '모델 API 응답 오류로 가상 모델을 사용 중입니다.'


async def _load_live(period: str) -> dict:
    interval, count = PERIODS[period]
    async with httpx.AsyncClient(timeout=9.0, headers={'Accept': 'application/json',
                     'User-Agent': 'CheckBit-Dashboard/1.0'}) as api:
        t_request = api.get(f'{UPBIT}/v1/ticker', params={'markets':'KRW-BTC'})
        c_request = api.get(f'{UPBIT}/v1/candles/minutes/{interval}',
                             params={'market':'KRW-BTC', 'count':count})
        ticker_response, candle_response = await asyncio.gather(t_request, c_request)
        ticker_response.raise_for_status(); candle_response.raise_for_status()
        tickers = ticker_response.json()
        raw_candles = candle_response.json()
    if not isinstance(tickers, list) or not tickers or not isinstance(raw_candles, list) or len(raw_candles) < 15:
        raise ValueError('Upbit did not return enough market data')
    ticker = tickers[0]
    candles = []
    for row in reversed(raw_candles):
        # Upbit's UTC time string omits timezone, so explicitly append Z.
        stamp = row['candle_date_time_utc'] + 'Z'
        candles.append({'timestamp': stamp,
                        'open': _finite_positive(row['opening_price'], 'open'),
                        'high': _finite_positive(row['high_price'], 'high'),
                        'low': _finite_positive(row['low_price'], 'low'),
                        'close': _finite_positive(row['trade_price'], 'close'),
                        'volume': float(row['candle_acc_trade_volume'])})
    # The newest ticker trade may be fresher than the final minute candle.
    latest = _finite_positive(ticker['trade_price'], 'ticker_price')
    candles[-1]['close'] = latest
    bands, mode, model_error = await detect(candles, period, interval)
    signals = [dict(point) for point in bands if point['status'] != 'normal'][-5:][::-1]
    price_24 = float(ticker['signed_change_rate']) * 100.0
    asof_ms = ticker.get('timestamp')
    observed_at = (datetime.fromtimestamp(int(asof_ms)/1000, tz=timezone.utc).isoformat()
                   if isinstance(asof_ms, (float,int)) else datetime.now(timezone.utc).isoformat())
    return {'symbol':'KRW-BTC', 'currency':'KRW', 'provider':'Upbit', 'period':period,
            'interval_minutes':interval, 'fetched_at':datetime.now(timezone.utc).isoformat(),
            'observed_at':observed_at, 'stale':False, 'price':latest,
            'change_24h_pct':round(price_24, 2),
            'volume_24h_krw':float(ticker['acc_trade_price_24h']),
            'high_24h':float(ticker['high_price']), 'low_24h':float(ticker['low_price']),
            'rsi_14':calc_rsi([c['close'] for c in candles]),
            'model': {'mode':mode,'label':('외부 탐지 모델' if mode=='external' else '가상 통계 모델'),
                      'method':'최근 24개 봉 · 중앙값/MAD 기반 예시', 'error':model_error},
            'latest_status':bands[-1]['status'], 'points':bands, 'signals':signals}


@router.get('/market')
async def market(period: Literal['1H','6H','24H'] = Query('6H'),
                 refresh: bool = Query(False)):
    now = time.monotonic()
    entry = _cached.get(period)
    if entry and now - entry[0] < (5 if refresh else CACHE_SECONDS):
        return entry[1]
    async with _lock:
        entry = _cached.get(period)
        if entry and time.monotonic() - entry[0] < (5 if refresh else CACHE_SECONDS):
            return entry[1]
        try:
            result = await _load_live(period)
            _cached[period] = (time.monotonic(), result)
            return result
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            LOG.warning('Market feed failed for %s: %s', period, exc)
            if entry and time.monotonic() - entry[0] < STALE_SECONDS:
                return {**entry[1], 'stale':True}
            raise HTTPException(status_code=503, detail='실제 시세 제공처에 연결할 수 없습니다. 잠시 후 다시 시도하세요.') from None


@router.get('/health')
async def dashboard_health():
    return {'ok':True, 'source':'Upbit public quotation REST', 'cache_seconds':CACHE_SECONDS,
            'model_mode':'external' if DETECTION_API_URL else 'mock'}
