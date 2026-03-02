import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from openai import OpenAI

MODEL_CANDIDATES = ["gpt-5.2", "gpt-4.1", "gpt-4o"]

st.set_page_config(page_title="QQQ Option Dashboard", layout="wide")
st.title("QQQ Real-time Option Viewer and AI Briefing")

api_key = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None
if not api_key:
    st.warning("OPENAI_API_KEY is not set. Charts will load, but AI briefing is disabled.")


def generate_briefing_with_fallback(_client: OpenAI, prompt_text: str):
    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            response = _client.responses.create(
                model=model_name,
                input=[
                    {
                        "role": "system",
                        "content": "You are a Korean market-briefing assistant for general stock traders. Be specific and practical.",
                    },
                    {"role": "user", "content": prompt_text},
                ],
                max_output_tokens=900,
            )
            text = (response.output_text or "").strip()
            if text:
                return text, model_name, None
            last_error = f"Model {model_name} returned empty output."
        except Exception as error:  # noqa: BLE001
            last_error = f"{model_name}: {error}"
    return "", None, last_error


def is_yf_rate_limit_error(error: Exception) -> bool:
    return error.__class__.__name__ == "YFRateLimitError"


@st.cache_data(ttl=600)
def get_stock_snapshot(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    # Use lighter requests first to reduce rate-limit pressure on Streamlit Cloud.
    hist = ticker.history(period="1d", interval="5m", prepost=True)
    if hist.empty:
        hist = ticker.history(period="5d", interval="1d")
    if hist.empty:
        return None, [], None

    current_price = float(hist["Close"].iloc[-1])
    expirations = list(ticker.options)

    ts = pd.Timestamp(hist.index[-1])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    return current_price, expirations, ts.tz_convert("UTC").isoformat()


@st.cache_data(ttl=600)
def get_option_chain(ticker_symbol: str, expiration_date: str):
    ticker = yf.Ticker(ticker_symbol)
    opt = ticker.option_chain(expiration_date)
    return opt.calls.copy(), opt.puts.copy()


@st.cache_data(ttl=900)
def get_recent_touch_count(ticker_symbol: str, level: float, band_pct: float = 0.0015):
    ticker = yf.Ticker(ticker_symbol)
    intraday = ticker.history(period="5d", interval="5m")
    if intraday.empty:
        return 0

    lower = level * (1 - band_pct)
    upper = level * (1 + band_pct)
    touches = intraday[(intraday["Low"] <= upper) & (intraday["High"] >= lower)]
    return int(len(touches))


def calc_max_pain(calls: pd.DataFrame, puts: pd.DataFrame):
    if calls.empty or puts.empty:
        return None

    strikes = sorted(set(calls["strike"].tolist()) | set(puts["strike"].tolist()))
    if not strikes:
        return None

    best_strike = None
    min_pain = float("inf")

    for strike in strikes:
        call_pain = ((strike - calls["strike"]).clip(lower=0) * calls["openInterest"]).sum()
        put_pain = ((puts["strike"] - strike).clip(lower=0) * puts["openInterest"]).sum()
        total_pain = float(call_pain + put_pain)

        if total_pain < min_pain:
            min_pain = total_pain
            best_strike = float(strike)

    return best_strike


def get_option_last_trade_utc(calls: pd.DataFrame, puts: pd.DataFrame):
    timestamps = pd.concat(
        [
            pd.to_datetime(calls.get("lastTradeDate"), utc=True, errors="coerce"),
            pd.to_datetime(puts.get("lastTradeDate"), utc=True, errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()

    if timestamps.empty:
        return None

    return pd.Timestamp(timestamps.max()).isoformat()


def infer_downside_targets(puts: pd.DataFrame, put_wall: float):
    below = puts[puts["strike"] < put_wall].copy()
    if below.empty:
        return round(put_wall * 0.99, 2), round(put_wall * 0.98, 2), "fallback"

    grouped = (
        below.groupby("strike", as_index=False)["openInterest"]
        .sum()
        .sort_values("strike", ascending=False)
    )

    threshold = grouped["openInterest"].quantile(0.75)
    strong = grouped[grouped["openInterest"] >= threshold].sort_values("strike", ascending=False)

    if len(strong) >= 2:
        target_1 = float(strong.iloc[0]["strike"])
        target_2 = float(strong.iloc[1]["strike"])
        return target_1, target_2, "strong_put_oi_nodes"

    if len(grouped) >= 2:
        target_1 = float(grouped.iloc[0]["strike"])
        target_2 = float(grouped.iloc[1]["strike"])
        return target_1, target_2, "nearest_put_oi_nodes"

    target_1 = float(grouped.iloc[0]["strike"])
    target_2 = round(target_1 * 0.99, 2)
    return target_1, target_2, "single_put_oi_node"


def fmt_ts_pair(iso_utc: str):
    if not iso_utc:
        return "N/A", "N/A"

    ts_utc = pd.Timestamp(iso_utc).tz_convert("UTC")
    ts_et = ts_utc.tz_convert("US/Eastern")
    ts_kst = ts_utc.tz_convert("Asia/Seoul")
    return ts_et.strftime("%Y-%m-%d %H:%M:%S ET"), ts_kst.strftime("%Y-%m-%d %H:%M:%S KST")


def classify_touch_frequency(touch_count_5d: int):
    if touch_count_5d >= 40:
        return "High", "Frequent retests suggest support may be weakening."
    if touch_count_5d >= 20:
        return "Medium", "Retests are accumulating; watch for support fatigue."
    return "Low", "Retest count is low; no strong support-fatigue signal yet."


def classify_max_pain_gravity(current_price: float, max_pain: float | None, put_wall: float, call_wall: float, selected_date: str):
    if max_pain is None:
        return "Unknown", "Max pain not available, cannot assess pull likelihood."

    dist_pct = abs(current_price - max_pain) / current_price * 100
    between_walls = put_wall <= current_price <= call_wall

    today_et = pd.Timestamp.now(tz="US/Eastern").date()
    expiry_date = pd.to_datetime(selected_date).date()
    is_expiry_today = expiry_date == today_et

    if is_expiry_today and dist_pct <= 1.0 and between_walls:
        return "High", "Expiry-day pull toward max pain is relatively likely."
    if dist_pct <= 2.0 and between_walls:
        return "Medium", "Pull toward max pain is possible while price stays between walls."
    return "Low", "Current location suggests weaker max-pain pull conditions."


ticker_symbol = "QQQ"
try:
    current_price, expirations, spot_ts_utc = get_stock_snapshot(ticker_symbol)
except Exception as error:  # noqa: BLE001
    if is_yf_rate_limit_error(error):
        st.error("Yahoo Finance 요청 한도에 도달했습니다. 잠시 후 다시 시도해 주세요.")
        st.info("Streamlit Cloud에서 요청이 몰리면 일시적으로 차단될 수 있습니다. 1~5분 후 새로고침해 보세요.")
    else:
        st.error(f"주가 데이터를 불러오는 중 오류가 발생했습니다: {error}")
    st.stop()

if current_price is None or not expirations:
    st.error("Failed to load stock/option data from Yahoo Finance.")
    st.stop()

col1, col2 = st.columns([1, 3])
with col1:
    selected_date = st.selectbox("Select expiry", expirations)
    st.caption(f"AI model order: {' -> '.join(MODEL_CANDIDATES)}")
    st.metric(label=f"{ticker_symbol} spot", value=f"${current_price:.2f}")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

try:
    calls, puts = get_option_chain(ticker_symbol, selected_date)
except Exception as error:  # noqa: BLE001
    if is_yf_rate_limit_error(error):
        st.error("옵션 체인 조회가 Yahoo Finance 요청 한도에 걸렸습니다. 잠시 후 다시 시도해 주세요.")
    else:
        st.error(f"옵션 체인을 불러오는 중 오류가 발생했습니다: {error}")
    st.stop()

min_strike = current_price * 0.90
max_strike = current_price * 1.10
calls = calls[(calls["strike"] >= min_strike) & (calls["strike"] <= max_strike)]
puts = puts[(puts["strike"] >= min_strike) & (puts["strike"] <= max_strike)]

if calls.empty or puts.empty:
    st.warning("Insufficient option data for the selected expiry.")
    st.stop()

call_wall_idx = calls["openInterest"].idxmax()
put_wall_idx = puts["openInterest"].idxmax()
call_wall_strike = float(calls.loc[call_wall_idx, "strike"])
put_wall_strike = float(puts.loc[put_wall_idx, "strike"])
call_wall_oi = int(calls.loc[call_wall_idx, "openInterest"])
put_wall_oi = int(puts.loc[put_wall_idx, "openInterest"])

max_pain = calc_max_pain(calls, puts)
try:
    touch_count_5d = get_recent_touch_count(ticker_symbol, put_wall_strike)
except Exception as error:  # noqa: BLE001
    touch_count_5d = 0
    if is_yf_rate_limit_error(error):
        st.warning("5일 터치 빈도 데이터는 요청 한도로 인해 이번 실행에서 제외되었습니다.")
    else:
        st.warning(f"5일 터치 빈도 데이터 조회 실패: {error}")
option_ts_utc = get_option_last_trade_utc(calls, puts)

put_gap_pct = ((current_price - put_wall_strike) / current_price) * 100
call_gap_pct = ((call_wall_strike - current_price) / current_price) * 100

break_trigger = round(put_wall_strike * 0.997, 2)
reclaim_trigger = round(put_wall_strike * 1.003, 2)
resistance_trigger = round(call_wall_strike * 0.997, 2)

down_target_1, down_target_2, down_source = infer_downside_targets(puts, put_wall_strike)
drop_to_t1 = ((down_target_1 - current_price) / current_price) * 100
drop_to_t2 = ((down_target_2 - current_price) / current_price) * 100

spot_et, spot_kst = fmt_ts_pair(spot_ts_utc)
opt_et, opt_kst = fmt_ts_pair(option_ts_utc)

freq_level, freq_comment = classify_touch_frequency(touch_count_5d)
mp_level, mp_comment = classify_max_pain_gravity(
    current_price=current_price,
    max_pain=max_pain,
    put_wall=put_wall_strike,
    call_wall=call_wall_strike,
    selected_date=selected_date,
)

puts_copy = puts.copy()
puts_copy["oi_negative"] = -puts_copy["openInterest"]

fig = go.Figure()
fig.add_trace(
    go.Bar(
        y=calls["strike"],
        x=calls["openInterest"],
        orientation="h",
        name="Call OI (resistance)",
        marker=dict(color="rgba(255, 215, 0, 0.72)"),
    )
)
fig.add_trace(
    go.Bar(
        y=puts_copy["strike"],
        x=puts_copy["oi_negative"],
        orientation="h",
        name="Put OI (support)",
        marker=dict(color="rgba(44, 130, 201, 0.70)"),
    )
)
fig.add_hline(
    y=current_price,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Spot ${current_price:.2f}",
    annotation_position="bottom right",
)
fig.update_layout(
    title=f"QQQ Open Interest Profile - Expiry {selected_date}",
    barmode="overlay",
    yaxis_title="Strike",
    xaxis_title="Open Interest",
    height=540,
    hovermode="y unified",
)

with col2:
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Data Timestamps")
st.caption(f"Spot update: {spot_et} / {spot_kst} | Option chain last trade: {opt_et} / {opt_kst}")

st.markdown("### Live Levels")
s1, s2, s3, s4, s5, s6 = st.columns(6)
s1.metric("Put Wall", f"${put_wall_strike:.2f}", f"{put_gap_pct:.2f}%")
s2.metric("Call Wall", f"${call_wall_strike:.2f}", f"{call_gap_pct:.2f}%")
s3.metric("5D Put Touches (5m)", f"{touch_count_5d}")
s4.metric("Max Pain", f"${max_pain:.2f}" if max_pain is not None else "N/A")
s5.metric("Break Trigger", f"${break_trigger:.2f}")
s6.metric("Reclaim Trigger", f"${reclaim_trigger:.2f}")

st.markdown("### Interpretation")
st.info(f"- 5D put-wall touch frequency: **{freq_level}**\n- Reason: {freq_comment}")
st.info(f"- Max-pain pull likelihood today: **{mp_level}**\n- Reason: {mp_comment}")

st.markdown("### Rule-based Downside Targets")
st.info(
    f"From spot ${current_price:.2f} at {spot_et}, if price breaks ${put_wall_strike:.2f} and stays below ${break_trigger:.2f}, "
    f"next put-OI supports are ${down_target_1:.2f} ({drop_to_t1:.2f}%) and ${down_target_2:.2f} ({drop_to_t2:.2f}%). "
    f"Source: {down_source}"
)

alert_lines = []
if put_gap_pct <= 1.2:
    alert_lines.append(f"- Spot is near put wall (${put_wall_strike:.2f}). Break trigger: ${break_trigger:.2f}")
if touch_count_5d >= 12:
    alert_lines.append("- Put wall has been retested frequently over the last 5 trading days.")
if current_price < put_wall_strike:
    alert_lines.append("- Spot is below put wall: treat as failed support.")
if current_price >= resistance_trigger:
    alert_lines.append(f"- Spot is entering call-wall resistance (${resistance_trigger:.2f}~${call_wall_strike:.2f}).")

if alert_lines:
    st.warning("Special conditions detected\n" + "\n".join(alert_lines))
else:
    st.success("No special condition now: price is between major walls.")

st.markdown("---")
st.subheader("AI Trading Briefing")

prompt = f"""
You are writing for general stock traders (not options specialists).
Write in Korean only, with clear action-oriented guidance.
Avoid options strategy jargon.

[Data]
- Symbol: QQQ
- Spot: ${current_price:.2f}
- Spot timestamp (ET/KST): {spot_et} / {spot_kst}
- Option chain last trade (ET/KST): {opt_et} / {opt_kst}
- Expiry: {selected_date}
- Support zone (put wall): ${put_wall_strike:.2f}
- Resistance zone (call wall): ${call_wall_strike:.2f}
- Break trigger: ${break_trigger:.2f}
- Reclaim trigger: ${reclaim_trigger:.2f}
- Upper resistance zone: ${resistance_trigger:.2f} ~ ${call_wall_strike:.2f}
- Downside targets: ${down_target_1:.2f} / ${down_target_2:.2f}
- Downside percentages vs current spot: {drop_to_t1:.2f}% / {drop_to_t2:.2f}%
- 5D support retest count: {touch_count_5d} ({freq_level})
- Max pain: ${max_pain if max_pain is not None else 0:.2f} ({mp_level})

[Output format]
1) One-line conclusion
2) Data timestamp note
3) Bullish/rebound plan: entry, stop, take-profit
4) Bearish/breakdown plan: trigger, target1 action, target2 action
5) Three no-trade conditions
6) Three checklist items
7) Final one-line risk warning

[Rules]
- Include explicit price levels in every major instruction.
- If mentioning percentages, state they are based on spot ${current_price:.2f} at {spot_et}.
- Keep sentences short and practical.
"""

if st.button("Generate AI briefing"):
    with st.spinner("Generating briefing..."):
        try:
            if client is None:
                st.error("OPENAI_API_KEY is missing. Add it in Streamlit Cloud App Settings > Secrets.")
                st.stop()

            analysis_text, used_model, model_error = generate_briefing_with_fallback(client, prompt)
            if analysis_text:
                st.success(f"Briefing generated (model: {used_model})")
                st.markdown(analysis_text)
            else:
                st.error("Failed to generate briefing from all configured models.")
                if model_error:
                    st.code(model_error, language="text")
        except Exception as error:
            st.error(f"API call failed: {error}")
