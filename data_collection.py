"""네이버 데이터랩과 YouTube에서 데이터를 수집하는 함수 모음."""

from __future__ import annotations

import html
import re

import pandas as pd
import requests

NAVER_TREND_URL = "https://naverapihub.apigw.ntruss.com/search-trend/v1/search"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
REQUEST_TIMEOUT = 15

# 분석할 디저트와 네이버 검색어 묶음. 네이버는 한 번에 5개 묶음까지 요청할 수 있다.
DESSERT_KEYWORDS = {
    "탕후루": ["탕후루"],
    "요아정": ["요아정", "요거트 아이스크림의 정석"],
    "두바이 초콜릿": ["두바이 초콜릿", "두바이초콜릿"],
    "두쫀쿠": ["두쫀쿠", "두바이 쫀득 쿠키", "두바이쫀득쿠키"],
    "버터떡": ["버터떡", "상하이 버터떡"],
}

# 디저트 이름이 없어도 먹거리 반응으로 볼 수 있는 표현. 결과를 보며 팀에서 고쳐 나간다.
FOOD_REACTION_WORDS = [
    "맛", "먹", "달달", "달콤", "달아요", "달아서", "달다", "디저트", "간식",
    "식감", "바삭", "쫀득", "쫄깃", "느끼", "비싸", "가격", "사먹", "사러", "사왔",
    "줄서", "웨이팅", "유행", "레시피", "재료", "초콜릿", "쿠키", "아이스크림",
    "요거트", "설탕", "과일", "피스타치오", "카다이프", "버터",
]

# 디저트 이름 없이 이런 표현이 있으면 영상·출연자·행사에 대한 반응으로 본다.
VIDEO_REACTION_WORDS = [
    "언니", "영상", "구독", "채널", "유튜버", "먹방", "당첨", "이벤트", "설문",
    "목소리", "노래", "표정", "브금", "썸네일",
]

# 검색 1회는 결과 수와 관계없이 100유닛이므로 넉넉히 받아 제목으로 거른다.
SEARCH_CANDIDATES = 25

TREND_COLUMNS = ["날짜", "디저트", "관심도"]
COMMENT_COLUMNS = ["디저트", "작성자", "댓글", "작성일", "좋아요 수", "영상 제목", "관련 댓글"]


class DataCollectionError(Exception):
    """화면에 그대로 보여 줘도 안전한 수집 오류. API 키를 담지 않는다."""


class YouTubeApiError(DataCollectionError):
    """YouTube가 돌려준 오류. reason으로 오류 종류를 구분한다."""

    def __init__(self, message: str, reason: str = ""):
        super().__init__(message)
        self.reason = reason


def fetch_naver_trend(
    client_id: str,
    client_secret: str,
    start_date: str,
    end_date: str,
    time_unit: str = "week",
) -> pd.DataFrame:
    """디저트 5개의 검색 관심도를 한 번의 요청으로 받아 표로 돌려준다.

    한 요청 안의 값만 같은 기준(최고점 = 100)이므로 5개를 함께 요청한다.
    """
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Content-Type": "application/json",
    }
    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": time_unit,
        "keywordGroups": [
            {"groupName": name, "keywords": keywords}
            for name, keywords in DESSERT_KEYWORDS.items()
        ],
    }

    try:
        response = requests.post(
            NAVER_TREND_URL, headers=headers, json=body, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException:
        raise DataCollectionError("네이버 서버에 연결하지 못했습니다.") from None

    if response.status_code != 200:
        raise DataCollectionError(_naver_error_message(response.status_code))

    return _parse_naver_trend(response.json())


def _naver_error_message(status_code: int) -> str:
    """네이버 응답 코드를 사용자가 이해할 수 있는 문장으로 바꾼다."""
    messages = {
        401: "네이버 API 키가 틀렸거나 Search Trend를 신청하지 않았습니다.",
        403: "NAVER API HUB 앱에서 Search Trend 사용 설정을 확인해 주세요.",
        429: "네이버 API 호출 한도를 넘었습니다. 잠시 후 다시 시도해 주세요.",
    }
    default = f"네이버 API 요청에 실패했습니다. (응답 코드 {status_code})"
    return messages.get(status_code, default)


def _parse_naver_trend(payload: dict) -> pd.DataFrame:
    """네이버 응답을 날짜·디저트·관심도 표로 바꾼다.

    검색량이 없는 구간은 응답에서 빠질 수 있으므로 0으로 채운다.
    """
    rows = [
        {"날짜": point["period"], "디저트": result["title"], "관심도": point["ratio"]}
        for result in payload.get("results", [])
        for point in result.get("data", [])
    ]
    if not rows:
        return pd.DataFrame(columns=TREND_COLUMNS)

    table = pd.DataFrame(rows)
    table["날짜"] = pd.to_datetime(table["날짜"])
    wide = table.pivot(index="날짜", columns="디저트", values="관심도")
    wide = wide.reindex(columns=list(DESSERT_KEYWORDS)).fillna(0.0)
    long_table = wide.reset_index().melt(
        id_vars="날짜", var_name="디저트", value_name="관심도"
    )
    return long_table.sort_values(["날짜", "디저트"], ignore_index=True)


def _youtube_get(url: str, params: dict) -> dict:
    """YouTube API를 호출해 JSON을 돌려준다. 오류 문구에 키가 드러나지 않게 한다."""
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        # 원래 오류에는 키가 포함된 주소가 들어 있을 수 있어 그대로 보여 주지 않는다.
        raise DataCollectionError("YouTube 서버에 연결하지 못했습니다.") from None

    if response.status_code == 200:
        return response.json()

    reason = _youtube_error_reason(response)
    if reason == "quotaExceeded":
        message = "YouTube API의 하루 사용량을 넘었습니다. 내일 다시 시도해 주세요."
    elif response.status_code in (400, 401, 403):
        message = (
            "YouTube API 키와 'YouTube Data API v3' 사용 설정을 확인해 주세요. "
            f"(응답 코드 {response.status_code}, 사유 {reason or '알 수 없음'})"
        )
    else:
        message = f"YouTube API 요청에 실패했습니다. (응답 코드 {response.status_code})"
    raise YouTubeApiError(message, reason)


def _youtube_error_reason(response: requests.Response) -> str:
    """YouTube 오류 응답에서 reason 값을 꺼낸다. 없으면 빈 문자열을 돌려준다."""
    try:
        errors = response.json()["error"]["errors"]
        return errors[0].get("reason", "")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return ""


def search_videos(api_key: str, query: str, max_videos: int) -> list[dict]:
    """검색어로 한국 영상을 찾아 영상 ID와 제목 목록을 돌려준다. 호출당 100유닛."""
    payload = _youtube_get(
        YOUTUBE_SEARCH_URL,
        {
            "key": api_key,
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_videos,
            "order": "relevance",
            "regionCode": "KR",
            "relevanceLanguage": "ko",
        },
    )
    return [
        {
            "video_id": item["id"]["videoId"],
            "title": html.unescape(item["snippet"]["title"]),
            "channel_id": item["snippet"].get("channelId"),
        }
        for item in payload.get("items", [])
    ]


def _compact(text: str) -> str:
    """띄어쓰기 차이를 무시하도록 공백을 없애고 소문자로 바꾼다."""
    return re.sub(r"\s+", "", text).lower()


def mentions_dessert(text: str, dessert: str) -> bool:
    """글에 해당 디저트의 검색어가 들어 있는지 확인한다."""
    compact_text = _compact(text)
    return any(_compact(word) in compact_text for word in DESSERT_KEYWORDS[dessert])


def select_related_videos(videos: list[dict], dessert: str, limit: int) -> list[dict]:
    """제목에 디저트 이름이 있는 영상만 검색 순서대로 limit개까지 고른다."""
    return [video for video in videos if mentions_dessert(video["title"], dessert)][:limit]


def is_related_comment(text: str, dessert: str) -> bool:
    """디저트에 대한 댓글인지 낱말 규칙으로 판단한다.

    디저트 이름이 있으면 관련 댓글이다. 이름이 없으면 먹거리 반응 표현이 있고
    영상·출연자·행사에 대한 표현이 없을 때만 관련 댓글로 본다.
    """
    if mentions_dessert(text, dessert):
        return True
    # '까먹다'의 '먹'이 먹거리 표현으로 잘못 잡히지 않게 먼저 지운다.
    compact_text = _compact(text).replace("까먹", "")
    has_food_word = any(word in compact_text for word in FOOD_REACTION_WORDS)
    has_video_word = any(word in compact_text for word in VIDEO_REACTION_WORDS)
    return has_food_word and not has_video_word


def fetch_video_comments(api_key: str, video_id: str, max_comments: int) -> list[dict]:
    """영상의 댓글을 최대 max_comments개 받는다. 페이지당 1유닛.

    댓글이 막혔거나 없어진 영상은 빈 목록을 돌려주어 수집을 계속한다.
    """
    comments: list[dict] = []
    page_token = None

    while len(comments) < max_comments:
        params = {
            "key": api_key,
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "order": "relevance",
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            payload = _youtube_get(YOUTUBE_COMMENTS_URL, params)
        except YouTubeApiError as error:
            if error.reason in ("commentsDisabled", "videoNotFound"):
                return comments
            raise

        for item in payload.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                {
                    "author_channel_id": snippet.get("authorChannelId", {}).get("value"),
                    "text": snippet.get("textDisplay", ""),
                    "published_at": snippet.get("publishedAt"),
                    "like_count": snippet.get("likeCount", 0),
                }
            )

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return comments


def clean_comment_text(text: str) -> str:
    """댓글 본문의 줄바꿈을 정리하고 본문 속 @닉네임을 가린다."""
    text = re.sub(r"@[^\s@]+", "@user", text)
    return re.sub(r"\s+", " ", text).strip()


def anonymize_comments(comments: list[dict]) -> list[dict]:
    """작성자 채널 ID를 user_001 형식으로 바꾸고, 원래 ID와 닉네임은 남기지 않는다.

    같은 작성자는 같은 번호를 받고, 채널 ID가 없는 댓글은 새 번호를 받는다.
    """
    author_numbers: dict[str, int] = {}
    next_number = 1
    anonymized = []

    for comment in comments:
        channel_id = comment.get("author_channel_id")
        if channel_id in author_numbers:
            number = author_numbers[channel_id]
        else:
            number = next_number
            next_number += 1
            if channel_id:
                author_numbers[channel_id] = number

        visible = {key: value for key, value in comment.items() if key != "author_channel_id"}
        visible["user_id"] = f"user_{number:03d}"
        visible["text"] = clean_comment_text(comment.get("text", ""))
        anonymized.append(visible)

    return anonymized


def collect_youtube_comments(
    api_key: str, videos_per_dessert: int, comments_per_video: int
) -> pd.DataFrame:
    """디저트별 영상을 검색해 댓글을 모으고 익명화한 표를 돌려준다."""
    collected: list[dict] = []

    for dessert in DESSERT_KEYWORDS:
        candidates = search_videos(api_key, f"{dessert} 후기", SEARCH_CANDIDATES)
        for video in select_related_videos(candidates, dessert, videos_per_dessert):
            for comment in fetch_video_comments(
                api_key, video["video_id"], comments_per_video
            ):
                # 영상 제작자가 직접 단 안내·홍보 댓글은 소비자 반응이 아니므로 뺀다.
                author = comment["author_channel_id"]
                if author and author == video["channel_id"]:
                    continue
                collected.append(
                    {**comment, "dessert": dessert, "video_title": video["title"]}
                )

    rows = [row for row in anonymize_comments(collected) if row["text"]]
    for row in rows:
        row["is_related"] = is_related_comment(row["text"], row["dessert"])
    if not rows:
        return pd.DataFrame(columns=COMMENT_COLUMNS)

    table = pd.DataFrame(rows).rename(
        columns={
            "dessert": "디저트",
            "user_id": "작성자",
            "text": "댓글",
            "published_at": "작성일",
            "like_count": "좋아요 수",
            "video_title": "영상 제목",
            "is_related": "관련 댓글",
        }
    )
    # YouTube 시각은 UTC이므로 한국 시각으로 바꾼다.
    table["작성일"] = (
        pd.to_datetime(table["작성일"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.tz_localize(None)
    )
    return table[COMMENT_COLUMNS]
