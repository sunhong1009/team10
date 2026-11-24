import streamlit as st
import pandas as pd
from analysis import (
    smooth_trend,
    find_peak_and_half,
    label_period,
    add_sentiment,
)

@st.cache_data
def load_data():
    # 1. CSVファイルの読み込み
    trend = pd.read_csv("data/trend.csv")
    comments = pd.read_csv("data/comments.csv")
    lexicon = pd.read_csv("data/sentiment_lexicon.csv")

    # 2. 日付カラムをdatetime型に変換
    trend["date"] = pd.to_datetime(trend["date"])
    comments["comment_date"] = pd.to_datetime(comments["comment_date"])

    # 3. analysis.py の関数で指標・ラベルを追加
    trend = smooth_trend(trend)                    # ratio_ma（4週移動平均）を追加
    cycle = find_peak_and_half(trend)              # ピーク・半減期の集計表を作成
    comments = label_period(comments, trend, cycle)  # period（時期）カラムを追加
    comments = add_sentiment(comments, lexicon)      # sentiment（感情）カラムを追加

    # 4. 3つのデータを返す
    return trend, cycle, comments

# 関数の呼び出し（このあとのフィルターやタブ処理で使います）
trend, cycle, comments = load_data()

from config import DESSERTS

# 共通フィルタリング関数（日付カラム名が trend は "date"、comments は "comment_date" であることに対応）
def filter_data(df, desserts, start, end, date_col="date"):
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    cond_dessert = df["dessert"].isin(desserts)
    cond_date = (df[date_col] >= start_ts) & (df[date_col] <= end_ts)
    return df[cond_dessert & cond_date]

# --- サイドバーUI ---
st.sidebar.header("필터 설정")

# 1. デザート選択（0個選択時のエラー防止）
selected_desserts = st.sidebar.multiselect("디저트 종류", DESSERTS, default=DESSERTS)
if not selected_desserts:
    st.warning("디저트를 하나 이상 고르세요")
    st.stop()

# 2. 照会期間選択（1日だけクリックした瞬間のエラー防止）
min_date = trend["date"].min().date()
max_date = trend["date"].max().date()
date_range = st.sidebar.date_input("조회 기간", (min_date, max_date))

if len(date_range) != 2:
    st.info("시작일과 종료일을 모두 선택하세요.")
    st.stop()

start_date, end_date = date_range

# 3. データの絞り込み実行（cycle は全期間計算を維持）
filtered_trend = filter_data(trend, selected_desserts, start_date, end_date, date_col="date")
filtered_comments = filter_data(comments, selected_desserts, start_date, end_date, date_col="comment_date")
filtered_cycle = cycle[cycle["dessert"].isin(selected_desserts)]

# 4. 期間フィルターの適用範囲に関する1行説明
st.caption("※ 정점과 절반 지점은 전체 기간 기준으로 계산되며, 조회 기간 필터는 그래프와 댓글 표에만 적용됩니다.")

from analysis import (
    sentiment_ratio,
    top_words,
    plot_trend,
    plot_cycle,
    plot_sentiment,
    plot_top_words,
)

# 3つのタブを作成
tab1, tab2, tab3 = st.tabs(["① 관심도 추이", "② 유행 주기", "③ 댓글 반응"])

# --- タブ1: 관심도 추이 ---
with tab1:
    fig_trend = plot_trend(filtered_trend)
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("값은 기간 중 최댓값을 100으로 둔 상댓값입니다.")

# --- タブ2: 유행 주기 ---
with tab2:
    # まだ半減していないデザート（weeks_to_half が空欄）を除外して最短を計算
    valid_cycle = filtered_cycle.dropna(subset=["weeks_to_half"])
    if not valid_cycle.empty:
        fastest_row = valid_cycle.loc[valid_cycle["weeks_to_half"].idxmin()]
        st.metric(
            label="가장 빨리 식은 디저트",
            value=fastest_row["dessert"],
            delta=f"정점 후 {fastest_row['weeks_to_half']}주 만에 절반 이하 감소",
            delta_color="inverse",
        )

    st.dataframe(filtered_cycle, use_container_width=True)
    fig_cycle = plot_cycle(filtered_cycle)
    st.plotly_chart(fig_cycle, use_container_width=True)

# --- 탭 3: 댓글 반응 ---
with tab3:
    # 1. 감성 비율 그래프 (전체 너비 사용)
    ratio_df = sentiment_ratio(filtered_comments)
    fig_sent = plot_sentiment(ratio_df)
    st.plotly_chart(fig_sent, use_container_width=True)

    # 2. 자주 나온 단어 그래프 (전체 너비 사용)
    words_list = top_words(filtered_comments["text"].dropna().tolist(), n=20)
    fig_words = plot_top_words(words_list)
    st.plotly_chart(fig_words, use_container_width=True)

    # 感情フィルター（selectbox）
    sentiment_choice = st.selectbox("감성 필터", ["전체", "긍정", "중립", "부정"])
    display_comments = filtered_comments.copy()
    if sentiment_choice != "전체":
        display_comments = display_comments[display_comments["sentiment"] == sentiment_choice]

    # 個人情報保護のため指定の4カラムのみ表示
    st.dataframe(
        display_comments[["user", "period", "text", "sentiment"]],
        use_container_width=True,
    )