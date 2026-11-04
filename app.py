"""SNS 유행 디저트 관심도 분석 앱. 실행: streamlit run app.py"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from data_collection import (
    DESSERT_KEYWORDS,
    DataCollectionError,
    collect_youtube_comments,
    fetch_naver_trend,
)

COLLECTION_START = date(2022, 1, 1)
VIDEOS_PER_DESSERT = 5
COMMENTS_PER_VIDEO = 300
NAVER_CACHE_SECONDS = 60 * 60
# 한 번 수집에 최대 575유닛(검색 5회 × 100 + 댓글 75쪽 × 1)을 쓰므로 6시간 동안 재사용한다.
YOUTUBE_CACHE_SECONDS = 6 * 60 * 60


def get_secret(name: str) -> str:
    """Streamlit Secrets에서 값을 읽는다. 파일이나 항목이 없으면 빈 문자열을 돌려준다."""
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def yesterday_in_korea() -> date:
    """배포 서버의 시간대와 관계없이 한국 기준 어제 날짜를 돌려준다."""
    korea_time = timezone(timedelta(hours=9))
    return datetime.now(korea_time).date() - timedelta(days=1)


# 인자 이름 앞의 밑줄(_)은 API 키를 Streamlit 캐시 키에 넣지 않는다는 뜻이다.
@st.cache_data(ttl=NAVER_CACHE_SECONDS, show_spinner="네이버 데이터를 수집하는 중입니다...")
def load_trend_data(_client_id: str, _client_secret: str, end_date: date) -> pd.DataFrame:
    """전체 기간의 네이버 관심도를 수집해 잠시 메모리에 보관한다."""
    return fetch_naver_trend(
        _client_id, _client_secret, COLLECTION_START.isoformat(), end_date.isoformat()
    )


@st.cache_data(ttl=YOUTUBE_CACHE_SECONDS, show_spinner="YouTube 댓글을 수집하는 중입니다...")
def load_comment_data(_api_key: str) -> pd.DataFrame:
    """디저트별 YouTube 댓글을 수집해 잠시 메모리에 보관한다."""
    return collect_youtube_comments(_api_key, VIDEOS_PER_DESSERT, COMMENTS_PER_VIDEO)


def show_sidebar(end_date: date) -> tuple[list[str], date, date]:
    """사이드바 필터를 그리고 선택한 디저트와 시작일, 종료일을 돌려준다."""
    st.sidebar.header("필터")
    desserts = list(DESSERT_KEYWORDS)
    selected = st.sidebar.multiselect("디저트 종류", desserts, default=desserts)
    period = st.sidebar.date_input(
        "조회 기간",
        value=(COLLECTION_START, end_date),
        min_value=COLLECTION_START,
        max_value=end_date,
    )
    # 달력에서 시작일만 고른 동안에는 값이 1개이므로 종료일을 끝 날짜로 둔다.
    if isinstance(period, (tuple, list)) and len(period) == 2:
        return selected, period[0], period[1]
    start = period[0] if isinstance(period, (tuple, list)) and period else COLLECTION_START
    return selected, start, end_date


def filter_by_period(
    table: pd.DataFrame, column: str, selected: list[str], start: date, end: date
) -> pd.DataFrame:
    """선택한 디저트와 기간에 해당하는 행만 남긴다."""
    dates = pd.to_datetime(table[column])
    last_moment = pd.Timestamp(end) + pd.Timedelta(days=1)
    in_period = (dates >= pd.Timestamp(start)) & (dates < last_moment)
    return table[table["디저트"].isin(selected) & in_period]


def show_trend_tab(selected: list[str], start: date, end: date, end_date: date) -> None:
    """네이버 검색 관심도 그래프와 표를 보여 준다."""
    st.caption("네이버 데이터랩 검색어 트렌드 · 값은 전체 수집 기간의 최고점을 100으로 둔 상대값")

    client_id = get_secret("NAVER_CLIENT_ID")
    client_secret = get_secret("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        st.warning("네이버 API 키가 없습니다. README의 실행 방법에 따라 키를 등록해 주세요.")
        return

    try:
        trend = load_trend_data(client_id, client_secret, end_date)
    except DataCollectionError as error:
        st.error(str(error))
        return

    visible = filter_by_period(trend, "날짜", selected, start, end)
    if visible.empty:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    wide = visible.pivot(index="날짜", columns="디저트", values="관심도")
    st.line_chart(wide)
    st.metric("표시 중인 데이터", f"{len(visible):,}행")

    table = wide.reset_index()
    table["날짜"] = table["날짜"].dt.strftime("%Y-%m-%d")
    st.dataframe(table, hide_index=True)


def show_comment_tab(selected: list[str], start: date, end: date) -> None:
    """익명 처리한 YouTube 댓글 표를 보여 준다."""
    st.caption(
        f"YouTube Data API · 디저트별 영상 {VIDEOS_PER_DESSERT}개, "
        f"영상당 댓글 최대 {COMMENTS_PER_VIDEO}개 · 제작자 댓글 제외 · 작성자는 익명 처리"
    )

    api_key = get_secret("YOUTUBE_API_KEY")
    if not api_key:
        st.warning("YouTube API 키가 없습니다. README의 실행 방법에 따라 키를 등록해 주세요.")
        return

    try:
        comments = load_comment_data(api_key)
    except DataCollectionError as error:
        st.error(str(error))
        return

    if comments.empty:
        st.info("수집된 댓글이 없습니다.")
        return

    in_condition = filter_by_period(comments, "작성일", selected, start, end)
    related_only = st.checkbox("디저트 관련 댓글만 보기", value=True)
    visible = in_condition[in_condition["관련 댓글"]] if related_only else in_condition

    # 관련 댓글이 없는 디저트도 0개로 보이도록 전체 수집 결과에서 센다.
    related_counts = (
        comments[comments["관련 댓글"]]["디저트"]
        .value_counts()
        .reindex(list(DESSERT_KEYWORDS), fill_value=0)
    )
    st.caption(
        "전체 수집 기간의 디저트별 관련 댓글: "
        + " · ".join(f"{name} {count:,}개" for name, count in related_counts.items())
    )

    if visible.empty:
        st.info("선택한 조건에 해당하는 댓글이 없습니다.")
        return

    st.metric("표시 중인 댓글", f"{len(visible):,}개 / 조건에 맞는 전체 {len(in_condition):,}개")
    table = visible.drop(columns="관련 댓글") if related_only else visible
    st.dataframe(table.sort_values("작성일", ascending=False), hide_index=True)


def main() -> None:
    """앱 화면 전체를 구성한다."""
    st.set_page_config(page_title="SNS 유행 디저트 관심도 분석", layout="wide")
    st.title("SNS 유행 디저트 관심도 분석")

    end_date = yesterday_in_korea()
    selected, start, end = show_sidebar(end_date)

    trend_tab, comment_tab = st.tabs(["네이버 검색 관심도", "YouTube 댓글"])
    with trend_tab:
        show_trend_tab(selected, start, end, end_date)
    with comment_tab:
        show_comment_tab(selected, start, end)


main()
