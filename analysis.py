import re
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from kiwipiepy import Kiwi

from config import DESSERTS, COLORS, PERIODS

# Kiwi 형태소 분석기 단일 인스턴스 (댓글 5,600여 개 처리 속도 최적화)
_KIWI = None


def _get_kiwi() -> Kiwi:
    global _KIWI
    if _KIWI is None:
        _KIWI = Kiwi()
    return _KIWI


# ==========================================
# 1. 데이터 정제 (전처리)
# ==========================================
def clean_trend(trend: pd.DataFrame) -> pd.DataFrame:
    """트렌드 데이터의 날짜/디저트명/결측 주를 정제한다."""
    df = trend.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "dessert"])
    df = df[df["dessert"].isin(DESSERTS)]

    # 모든 디저트 x 전체 주 그리드를 맞춰 빠진 주가 있으면 0으로 보간
    all_weeks = pd.date_range(df["date"].min(), df["date"].max(), freq="W-MON")
    grid = pd.MultiIndex.from_product([all_weeks, DESSERTS], names=["date", "dessert"])
    df = df.set_index(["date", "dessert"]).reindex(grid, fill_value=0).reset_index()

    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce").fillna(0.0)
    return df.sort_values(["dessert", "date"]).reset_index(drop=True)


def clean_comments(comments: pd.DataFrame) -> pd.DataFrame:
    """홍보성 URL 댓글, 중복 문장, 비언어적 노이즈를 정제한다."""
    df = comments.copy()
    df["comment_date"] = pd.to_datetime(df["comment_date"], errors="coerce")
    df = df.dropna(subset=["comment_date", "dessert", "text"])
    df = df[df["dessert"].isin(DESSERTS)]

    # 1) URL 포함 홍보/고정 공지 댓글 제거
    df = df[~df["text"].str.contains(r"http\S+|www\.\S+", case=False, regex=True)]

    # 2) 텍스트 정규화 (멘션 재확인, 과도한 자모 반복 축약, 공백 정리)
    def _normalize(text: str) -> str:
        t = re.sub(r"@\S+", "@user", str(text))
        t = re.sub(r"([ㄱ-ㅎㅏ-ㅣ])\1{2,}", r"\1\1", t)
        return re.sub(r"\s+", " ", t).strip()

    df["text"] = df["text"].apply(_normalize)
    df = df[df["text"].str.len() >= 2]

    # 3) 동일 영상 및 동일 디저트 내 중복 댓글 제거
    if "video_id" in df.columns:
        df = df.drop_duplicates(subset=["dessert", "video_id", "text"])
    df = df.drop_duplicates(subset=["dessert", "text"])

    # 4) 혹시 남아 있을 수 있는 개인정보 컬럼 제거
    forbidden = ["authorDisplayName", "authorChannelId", "authorProfileImageUrl", "authorChannelUrl"]
    df = df.drop(columns=[c for c in forbidden if c in df.columns])

    return df.reset_index(drop=True)


# ==========================================
# 2. 유행 주기 분석 (핵심 질문 1)
# ==========================================
def smooth_trend(trend: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """디저트별 4주 이동평균(ratio_ma) 컬럼을 추가한다."""
    df = clean_trend(trend)
    df["ratio_ma"] = df.groupby("dessert")["ratio"].transform(
        lambda s: s.rolling(window=window, min_periods=1).mean()
    )
    return df


def find_peak_and_half(trend: pd.DataFrame) -> pd.DataFrame:
    """디저트별 정점 주와 정점 이후 처음으로 정점의 50% 이하가 된 반감 주를 찾는다."""
    df = trend if "ratio_ma" in trend.columns else smooth_trend(trend)
    rows = []

    for dessert in DESSERTS:
        g = df[df["dessert"] == dessert].sort_values("date")
        if g.empty:
            continue

        peak_idx = g["ratio_ma"].idxmax()
        peak_date = g.loc[peak_idx, "date"]
        peak_ratio = round(float(g.loc[peak_idx, "ratio_ma"]), 2)

        after = g[(g["date"] > peak_date) & (g["ratio_ma"] <= peak_ratio * 0.5)]
        if not after.empty:
            half_date = after.iloc[0]["date"]
            weeks_to_half = int((half_date - peak_date).days // 7)
        else:
            half_date = pd.NaT
            weeks_to_half = None

        rows.append(
            {
                "dessert": dessert,
                "peak_date": peak_date,
                "peak_ratio": peak_ratio,
                "half_date": half_date,
                "weeks_to_half": weeks_to_half,
            }
        )

    return pd.DataFrame(rows)


# ==========================================
# 3. 댓글 시기 구분 & 감성분석 (핵심 질문 2)
# ==========================================
def label_period(comments: pd.DataFrame, trend: pd.DataFrame, cycle: pd.DataFrame) -> pd.DataFrame:
    """댓글 작성일을 주 단위(월요일 시작)로 맞춰 트렌드·주기와 조인하고 시기를 라벨링한다."""
    df_c = clean_comments(comments)
    df_t = trend if "ratio_ma" in trend.columns else smooth_trend(trend)

    df_c["week_date"] = pd.to_datetime(df_c["comment_date"]).dt.to_period("W-SUN").dt.start_time

    merged = df_c.merge(
        df_t[["date", "dessert", "ratio_ma"]],
        left_on=["week_date", "dessert"],
        right_on=["date", "dessert"],
        how="left",
    ).merge(
        cycle[["dessert", "peak_date", "peak_ratio", "half_date"]],
        on="dessert",
        how="left",
    )

    def _assign(row) -> str:
        w = row["week_date"]
        peak = row["peak_date"]
        half = row["half_date"]
        ma = row["ratio_ma"]
        p_ratio = row["peak_ratio"]

        if pd.isna(peak):
            return "도입기"
        if w < peak:
            if pd.notna(ma) and ma >= p_ratio * 0.5:
                return "정점기"
            return "도입기"
        else:
            if pd.notna(half) and w >= half:
                return "감소기"
            return "정점기"

    merged["period"] = merged.apply(_assign, axis=1)
    drop_cols = ["week_date", "date", "ratio_ma", "peak_date", "peak_ratio", "half_date"]
    return merged.drop(columns=drop_cols)


def _to_lexicon_dict(lexicon) -> dict:
    """DataFrame 또는 dict 형태의 감성사전을 {단어: 점수} dict로 통일한다."""
    if isinstance(lexicon, dict):
        return lexicon
    return dict(zip(lexicon["word"].astype(str).str.strip(), lexicon["score"].astype(float)))


def classify_sentiment(text: str, lexicon) -> str:
    """Kiwi 형태소 분석 후 감성사전 점수를 합산해 긍정/부정/중립을 반환한다."""
    if not isinstance(text, str) or not text.strip():
        return "중립"

    lex_dict = _to_lexicon_dict(lexicon)
    kiwi = _get_kiwi()
    tokens = kiwi.tokenize(text)

    total_score = 0.0
    negate_next = False  # 직전 토큰이 부정어('안', '못')인지 여부

    for tok in tokens:
        form = tok.form
        if form in ("안", "못") and tok.tag in ("MAG", "VV", "VX"):
            negate_next = True
            continue

        if form in lex_dict:
            score = lex_dict[form]
            if negate_next:
                score = -score
                negate_next = False
            total_score += score
        elif tok.tag.startswith(("N", "V", "M")):
            negate_next = False

    if total_score > 0:
        return "긍정"
    elif total_score < 0:
        return "부정"
    return "중립"


def add_sentiment(comments: pd.DataFrame, lexicon) -> pd.DataFrame:
    """모든 댓글에 sentiment 컬럼을 추가한다."""
    df = comments.copy()
    lex_dict = _to_lexicon_dict(lexicon)
    df["sentiment"] = df["text"].apply(lambda t: classify_sentiment(t, lex_dict))
    return df


def sentiment_ratio(comments: pd.DataFrame) -> pd.DataFrame:
    """디저트 x 시기(period) x 감성(sentiment) 비율(%) 요약표를 반환한다."""
    counts = (
        comments.groupby(["dessert", "period", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(["dessert", "period"])["count"].transform("sum")
    counts["ratio"] = (counts["count"] / totals * 100).round(1)
    return counts


# ==========================================
# 4. 키워드 빈도 추출
# ==========================================
STOPWORDS = {
    "탕후루", "요아정", "두바이", "초콜릿", "초콜렛", "두쫀쿠", "쫀득", "쿠키", "버터떡", "버터",
    "영상", "진짜", "너무", "사람", "먹방", "오늘", "지금", "정도", "생각", "느낌", "하나",
    "이거", "저거", "그거", "유튜브", "구독", "좋아요", "댓글", "편집", "소리", "asmr", "ASMR",
    "언니", "감사", "다음", "레시피", "만들기", "재료", "우리", "제발", "이번",
}


def top_words(texts, n: int = 20) -> list[tuple[str, int]]:
    """일반명사(NNG), 고유명사(NNP), 형용사(VA) 중 2글자 이상 핵심 단어 상위 n개를 반환한다."""
    kiwi = _get_kiwi()
    counter = Counter()

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        for tok in kiwi.tokenize(text):
            if tok.tag in ("NNG", "NNP", "VA"):
                word = tok.form
                if len(word) >= 2 and word not in STOPWORDS:
                    counter[word] += 1

    return counter.most_common(n)


# ==========================================
# 5. Plotly 시각화 함수 4종
# ==========================================
def plot_trend(trend: pd.DataFrame) -> go.Figure:
    """① 디저트별 주간 관심도(4주 이동평균) 꺾은선 그래프 + 정점 표시."""
    df = trend if "ratio_ma" in trend.columns else smooth_trend(trend)
    cycle = find_peak_and_half(df)

    fig = px.line(
        df,
        x="date",
        y="ratio_ma",
        color="dessert",
        color_discrete_map=COLORS,
        labels={"date": "날짜", "ratio_ma": "관심도 (4주 이동평균)", "dessert": "디저트"},
        title="디저트별 네이버 검색 관심도 추이 (4주 이동평균)",
    )

    # 각 디저트의 정점 위치에 마커 추가
    for _, row in cycle.iterrows():
        d = row["dessert"]
        fig.add_trace(
            go.Scatter(
                x=[row["peak_date"]],
                y=[row["peak_ratio"]],
                mode="markers+text",
                marker=dict(size=10, color=COLORS.get(d, "#333333"), line=dict(width=2, color="white")),
                text=[f"{d} 정점({row['peak_ratio']})"],
                textposition="top center",
                showlegend=False,
            )
        )

    fig.update_layout(hovermode="x unified", legend_title_text="디저트")
    return fig


def plot_cycle(cycle: pd.DataFrame) -> go.Figure:
    """② 정점에서 절반(50%) 이하로 떨어지기까지 걸린 기간(주) 가로 막대 그래프."""
    df = cycle.dropna(subset=["weeks_to_half"]).sort_values("weeks_to_half", ascending=True)

    fig = px.bar(
        df,
        x="weeks_to_half",
        y="dessert",
        orientation="h",
        color="dessert",
        color_discrete_map=COLORS,
        text="weeks_to_half",
        labels={"weeks_to_half": "정점 → 절반 하락 소요 기간 (주)", "dessert": "디저트"},
        title="디저트별 유행 반감기 비교 (정점 대비 50% 이하 도달 주 수)",
    )
    fig.update_traces(texttemplate="%{text}주", textposition="outside")
    fig.update_layout(showlegend=False, xaxis_range=[0, df["weeks_to_half"].max() * 1.2])
    return fig


def plot_sentiment(ratio_df: pd.DataFrame) -> go.Figure:
    """③ 시기별 긍정·중립·부정 100% 누적 막대 그래프."""
    sentiment_colors = {"긍정": "#2A9D8F", "중립": "#B0B7C3", "부정": "#E63946"}
    df = ratio_df.copy()
    df["period"] = pd.Categorical(df["period"], categories=PERIODS, ordered=True)
    df = df.sort_values(["dessert", "period"])

    fig = px.bar(
        df,
        x="period",
        y="ratio",
        color="sentiment",
        facet_col="dessert",
        barmode="stack",
        color_discrete_map=sentiment_colors,
        category_orders={"period": PERIODS, "sentiment": ["긍정", "중립", "부정"]},
        labels={"period": "유행 시기", "ratio": "비율 (%)", "sentiment": "감성", "dessert": "디저트"},
        title="디저트·유행 시기별 댓글 감성 비율 변화",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(range=[0, 100])
    return fig


def plot_top_words(words: list[tuple[str, int]], title: str = "자주 나온 핵심 키워드 TOP 20") -> go.Figure:
    """③ 자주 나온 단어 가로 막대 그래프."""
    if not words:
        return go.Figure()

    df = pd.DataFrame(words, columns=["word", "count"]).sort_values("count", ascending=True)
    fig = px.bar(
        df,
        x="count",
        y="word",
        orientation="h",
        text="count",
        labels={"count": "출현 횟수", "word": "단어"},
        title=title,
    )
    fig.update_traces(marker_color="#457B9D", textposition="outside")
    fig.update_layout(xaxis_range=[0, df["count"].max() * 1.15])
    return fig


# ==========================================
# 직접 실행 테스트 (파이참에서 Run 눌러 확인용)
# ==========================================
if __name__ == "__main__":
    raw_trend = pd.read_csv("data/trend.csv")
    raw_comments = pd.read_csv("data/comments.csv")
    lexicon_df = pd.read_csv("data/sentiment_lexicon.csv")

    trend_df = smooth_trend(raw_trend)
    cycle_df = find_peak_and_half(trend_df)
    comments_df = label_period(raw_comments, trend_df, cycle_df)
    comments_df = add_sentiment(comments_df, lexicon_df)
    ratio_df = sentiment_ratio(comments_df)

    print("=== [1] 정제 후 댓글 수 ===")
    print(f"원본: {len(raw_comments)}건 -> 정제 후: {len(comments_df)}건")

    print("\n=== [2] 유행 주기 (정점 및 반감기) ===")
    print(cycle_df)

    print("\n=== [3] 디저트별 시기 분포 ===")
    print(pd.crosstab(comments_df["dessert"], comments_df["period"]))

    print("\n=== [4] 감성 비율 요약 ===")
    print(ratio_df.head(15))

    print("\n=== [5] 전체 상위 키워드 TOP 10 ===")
    print(top_words(comments_df["text"], n=10))