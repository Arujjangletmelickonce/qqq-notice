import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from openai import OpenAI

MODEL_NAME = "gpt-5.2"


st.set_page_config(page_title="QQQ 옵션 분석 대시보드", layout="wide")
st.title("QQQ 실시간 옵션 뷰어 & AI 브리핑")

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("OpenAI API 키가 없습니다. .streamlit/secrets.toml 파일을 확인해 주세요.")
    st.stop()


@st.cache_data(ttl=120)
def get_stock_snapshot(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1d", interval="1m", prepost=True)
    if hist.empty:
        return None, [], None

    current_price = float(hist["Close"].iloc[-1])
    expirations = list(ticker.options)

    ts = pd.Timestamp(hist.index[-1])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    return current_price, expirations, ts.tz_convert("UTC").isoformat()


@st.cache_data(ttl=120)
def get_option_chain(ticker_symbol: str, expiration_date: str):
    ticker = yf.Ticker(ticker_symbol)
    opt = ticker.option_chain(expiration_date)
    return opt.calls.copy(), opt.puts.copy()


@st.cache_data(ttl=120)
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
        return "높음", "최근 5거래일 기준 재터치가 매우 잦아 지지 소진 리스크가 큰 편입니다."
    if touch_count_5d >= 20:
        return "보통", "재터치가 누적되고 있어 지지 강도가 약해질 수 있습니다."
    return "낮음", "재터치 빈도가 낮아 아직 지지 소진 신호는 강하지 않습니다."


def classify_max_pain_gravity(current_price: float, max_pain: float | None, put_wall: float, call_wall: float, selected_date: str):
    if max_pain is None:
        return "판단불가", "Max Pain 계산값이 없어 수렴 가능성을 평가할 수 없습니다."

    dist_pct = abs(current_price - max_pain) / current_price * 100
    between_walls = put_wall <= current_price <= call_wall

    today_et = pd.Timestamp.now(tz="US/Eastern").date()
    expiry_date = pd.to_datetime(selected_date).date()
    is_expiry_today = expiry_date == today_et

    if is_expiry_today and dist_pct <= 1.0 and between_walls:
        return "높음", "만기일 당일이고 Max Pain과 거리가 가까워 핀ning/수렴 가능성이 상대적으로 높습니다."
    if dist_pct <= 2.0 and between_walls:
        return "보통", "벽 사이 구간에서 Max Pain과 거리도 크지 않아 수렴 가능성을 열어둘 수 있습니다."
    return "낮음", "현재 위치가 Max Pain과 떨어져 있거나 벽 밖이라 수렴 가능성은 상대적으로 낮습니다."


ticker_symbol = "QQQ"
current_price, expirations, spot_ts_utc = get_stock_snapshot(ticker_symbol)

if current_price is None or not expirations:
    st.error("Yahoo Finance에서 주가/옵션 데이터를 불러오지 못했습니다.")
    st.stop()

col1, col2 = st.columns([1, 3])
with col1:
    selected_date = st.selectbox("만기일 선택", expirations)
    st.caption(f"AI 모델: {MODEL_NAME} (고정)")
    st.metric(label=f"{ticker_symbol} 현재가", value=f"${current_price:.2f}")
    if st.button("데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()

calls, puts = get_option_chain(ticker_symbol, selected_date)

min_strike = current_price * 0.90
max_strike = current_price * 1.10
calls = calls[(calls["strike"] >= min_strike) & (calls["strike"] <= max_strike)]
puts = puts[(puts["strike"] >= min_strike) & (puts["strike"] <= max_strike)]

if calls.empty or puts.empty:
    st.warning("선택한 만기일의 옵션 데이터가 부족합니다.")
    st.stop()

call_wall_idx = calls["openInterest"].idxmax()
put_wall_idx = puts["openInterest"].idxmax()
call_wall_strike = float(calls.loc[call_wall_idx, "strike"])
put_wall_strike = float(puts.loc[put_wall_idx, "strike"])
call_wall_oi = int(calls.loc[call_wall_idx, "openInterest"])
put_wall_oi = int(puts.loc[put_wall_idx, "openInterest"])

max_pain = calc_max_pain(calls, puts)
touch_count_5d = get_recent_touch_count(ticker_symbol, put_wall_strike)
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
        name="Call OI (저항)",
        marker=dict(color="rgba(255, 215, 0, 0.72)"),
    )
)
fig.add_trace(
    go.Bar(
        y=puts_copy["strike"],
        x=puts_copy["oi_negative"],
        orientation="h",
        name="Put OI (지지)",
        marker=dict(color="rgba(44, 130, 201, 0.70)"),
    )
)
fig.add_hline(
    y=current_price,
    line_dash="dash",
    line_color="red",
    annotation_text=f"현재가 ${current_price:.2f}",
    annotation_position="bottom right",
)
fig.update_layout(
    title=f"QQQ 미결제약정(OI) 프로필 - 만기일 {selected_date}",
    barmode="overlay",
    yaxis_title="행사가",
    xaxis_title="미결제약정",
    height=540,
    hovermode="y unified",
)

with col2:
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### 데이터 기준 시각")
st.caption(
    f"현재가 갱신: {spot_et} / {spot_kst} | 옵션 체인 마지막 체결: {opt_et} / {opt_kst}"
)

st.markdown("### 실시간 핵심 레벨")
s1, s2, s3, s4, s5, s6 = st.columns(6)
s1.metric("Put Wall", f"${put_wall_strike:.2f}", f"{put_gap_pct:.2f}%")
s2.metric("Call Wall", f"${call_wall_strike:.2f}", f"{call_gap_pct:.2f}%")
s3.metric("5일 Put Wall 터치(5분봉)", f"{touch_count_5d}회")
s4.metric("맥스페인", f"${max_pain:.2f}" if max_pain is not None else "N/A")
s5.metric("이탈 트리거", f"${break_trigger:.2f}")
s6.metric("회복 확인", f"${reclaim_trigger:.2f}")

st.markdown("### 해석 요약")
st.info(f"- 5일 Put Wall 터치 빈도: **{freq_level}**\n- 근거: {freq_comment}")
st.info(f"- 오늘 맥스페인 수렴 가능성: **{mp_level}**\n- 근거: {mp_comment}")

st.markdown("### 룰 기반 하방 목표가 (Put OI 기반)")
st.info(
    f"기준 시각({spot_et} / {spot_kst}) 현재가 ${current_price:.2f} 대비, "
    f"Put Wall ${put_wall_strike:.2f} 이탈 후 ${break_trigger:.2f} 아래에서 유지되면 "
    f"다음 OI 지지 후보는 1차 ${down_target_1:.2f} ({drop_to_t1:.2f}%), "
    f"2차 ${down_target_2:.2f} ({drop_to_t2:.2f}%)입니다. "
    f"산출 근거: {down_source}"
)

alert_lines = []
if put_gap_pct <= 1.2:
    alert_lines.append(
        f"- 현재가가 Put Wall(${put_wall_strike:.2f})에 근접했습니다. 이탈 트리거: ${break_trigger:.2f}"
    )
if touch_count_5d >= 12:
    alert_lines.append("- 최근 5거래일 Put Wall 재터치 빈도가 높습니다.")
if current_price < put_wall_strike:
    alert_lines.append("- 현재가가 Put Wall 아래입니다. 지지 실패 구간으로 봐야 합니다.")
if current_price >= resistance_trigger:
    alert_lines.append(
        f"- 현재가가 Call Wall 저항권(${resistance_trigger:.2f}~${call_wall_strike:.2f})에 진입했습니다."
    )

if alert_lines:
    st.warning("특이사항 감지\n" + "\n".join(alert_lines))
else:
    st.success("현재는 주요 벽 사이의 중립 구간입니다.")

st.markdown("---")
st.subheader("AI 트레이딩 데스크 브리핑")

prompt = f"""
?? ?? ?? ???? ?? ??? ?? ??? ????.
?? ?? ??? ?????, ?? ?? ???? ???? ?? ????.
?? ??? ????, ?? ??? ?? ???? ????.

[?? ???]
- ??: QQQ
- ???: ${current_price:.2f}
- ??? ?? ??(ET/KST): {spot_et} / {spot_kst}
- ?? ?? ??? ??(ET/KST): {opt_et} / {opt_kst}
- ???: {selected_date}
- ?? ?? ???(Put Wall): ${put_wall_strike:.2f}
- ?? ?? ???(Call Wall): ${call_wall_strike:.2f}
- ?? ?? ??: ${break_trigger:.2f}
- ?? ?? ??: ${reclaim_trigger:.2f}
- ?? ?? ??: ${resistance_trigger:.2f} ~ ${call_wall_strike:.2f}
- ?? 1?/2? ??: ${down_target_1:.2f} / ${down_target_2:.2f}
- ?? ???(??? ??): {drop_to_t1:.2f}% / {drop_to_t2:.2f}%
- ?? 5? ?? ??? ??: {touch_count_5d}? (??: {freq_level})
- ?? ?? ??(Max Pain ??): ${max_pain if max_pain is not None else 0:.2f} (??: {mp_level})

[?? ?? - ??? ???]
1) ? ? ??
2) ?? ??? ?? ?? ?? (??/?? ??)
3) ?? ?? ?? ?? ??
- ??/?? ????: ?? ?? 1?, ?? ?? 1?, ?? ?? 1?
- ??/?? ????: ?? ?? 1?, 1? ?? ?? 1?, 2? ?? ?? 1?
4) ?? ?? ?? ?? 3? (? 1?)
5) ?? ????? 3? (? 1?)
6) ??? ??? ?? 1?

[?? ??]
- ?? ??? ??(?: ? ????, ? ??, ?? ?)? ???? ??.
- ?? ??($)? ??? ????, "??? ????/??? ???"? ??? ??.
- ??? ??? ??? "??? ${current_price:.2f}, ?? ?? {spot_et}" ???? ??? ????.
- ??? ?? ???? ???? ?? ??? ????.
"""
if st.button("실시간 시황 브리핑 생성"):
    with st.spinner("브리핑 생성 중..."):
        try:

            response = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {
                        "role": "system",
                        "content": "?? ?? ?? ???? ?? ??? ?? ??? ????. ?? ????? ????? ?? ??? ??? ????.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=900,
            )
            analysis_text = (response.output_text or "").strip()
            st.success("??? ?? ??")
            if analysis_text:
                st.markdown(analysis_text)
            else:
                st.warning("??? ???? ???? ?????. ?? ?? ??? ??? ???.")
                st.code(str(response), language="text")
        except Exception as error:
            st.error(f"API 호출 중 오류가 발생했습니다: {error}")
