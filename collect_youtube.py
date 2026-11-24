import os
import re
import requests
import pandas as pd
from dotenv import load_dotenv
from config import DESSERTS

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")


def search_videos(query, published_after, published_before, max_results=8):
    """특정 기간 내 한국어 영상 검색 (search.list: 1회당 100단위 소모)"""
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "regionCode": "KR",
        "relevanceLanguage": "ko",
        "publishedAfter": f"{published_after}T00:00:00Z",
        "publishedBefore": f"{published_before}T23:59:59Z",
        "maxResults": max_results,
        "key": API_KEY,
    }
    res = requests.get(url, params=params)
    res.raise_for_status()

    videos = []
    for item in res.json().get("items", []):
        snippet = item["snippet"]
        videos.append({
            "video_id": item["id"]["videoId"],
            "video_title": snippet["title"],
            "video_date": snippet["publishedAt"][:10],  # YYYY-MM-DD
        })
    return videos


def get_comments(video_info, dessert, max_comments=100):
    """영상 1개의 최상위 댓글 최대 100개 수집 (commentThreads.list: 1회당 1단위)"""
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_info["video_id"],
        "maxResults": max_comments,
        "textFormat": "plainText",
        "order": "relevance",
        "key": API_KEY,
    }
    res = requests.get(url, params=params)

    # 댓글 사용 중지(commentsDisabled) 등 오류 영상은 건너뜀
    if res.status_code != 200:
        print(f"  [건너뜀] {video_info['video_id']} (응답 코드: {res.status_code})")
        return []

    rows = []
    for item in res.json().get("items", []):
        top = item["snippet"]["topLevelComment"]["snippet"]
        channel_id = top.get("authorChannelId", {}).get("value", "unknown")
        rows.append({
            "dessert": dessert,
            "video_id": video_info["video_id"],
            "video_title": video_info["video_title"],
            "video_date": video_info["video_date"],
            "raw_channel_id": channel_id,  # 익명화 매핑 후 삭제될 임시 컬럼
            "text": top.get("textDisplay", ""),
            "likes": int(top.get("likeCount", 0)),
            "comment_date": top.get("publishedAt", "")[:10],  # YYYY-MM-DD
        })
    return rows


def anonymize(df):
    """작성자 채널 ID를 user_001 형식으로 치환하고 개인정보 및 @아이디 제거"""
    df = df.copy()

    # 등장 순서대로 user_001, user_002... 부여 (원본 매핑표 미저장)
    unique_ids = df["raw_channel_id"].drop_duplicates().tolist()
    id_map = {cid: f"user_{idx:03d}" for idx, cid in enumerate(unique_ids, start=1)}
    df["user"] = df["raw_channel_id"].map(id_map)

    # 본문 내 @아이디 가리기
    df["text"] = df["text"].astype(str).apply(lambda t: re.sub(r"@\S+", "@user", t))

    # 원본 개인정보 컬럼 제거 및 1부터 시작하는 comment_no 부여
    df = df.drop(columns=["raw_channel_id"], errors="ignore")
    df.insert(0, "comment_no", range(1, len(df) + 1))

    # 가이드 3장 약속 컬럼 순서 고정
    cols = [
        "comment_no", "dessert", "video_id", "video_title",
        "video_date", "user", "text", "likes", "comment_date"
    ]
    return df[cols]


if __name__ == "__main__":
    # 네이버 트렌드 그래프의 정점·감소기를 보고 디저트별 검색 기간(2~3구간) 설정
    SEARCH_WINDOWS = {
        "탕후루": [
            ("2023-07-01", "2023-11-30"),  # 정점기 (2023년 8~10월 피크 전후)
            ("2024-01-01", "2024-07-31"),  # 감소기
        ],
        "두바이 초콜릿": [
            ("2024-06-01", "2024-09-15"),  # 정점기 (2024년 7월 피크 전후)
            ("2024-10-01", "2025-04-30"),  # 감소기
        ],
        "요아정": [
            ("2024-06-15", "2024-10-15"),  # 정점기 (2024년 7~8월 피크 전후)
            ("2024-11-01", "2025-06-30"),  # 감소기
        ],
        "두쫀쿠": [
            ("2025-12-01", "2026-02-28"),  # 정점기 (2026년 1월 ratio 100 피크 전후)
            ("2026-03-15", "2026-10-31"),  # 감소기
        ],
        "버터떡": [
            ("2026-02-15", "2026-04-30"),  # 정점기 (2026년 3월 피크 전후)
            ("2026-05-15", "2026-11-15"),  # 감소기
        ],
    }

    all_comments = []

    for dessert in DESSERTS:
        print(f"\n=== [{dessert}] 영상 검색 및 댓글 수집 시작 ===")
        seen_video_ids = set()

        for start_dt, end_dt in SEARCH_WINDOWS[dessert]:
            videos = search_videos(dessert, start_dt, end_dt, max_results=7)
            for v in videos:
                # 중복 영상 제거
                if v["video_id"] in seen_video_ids:
                    continue
                seen_video_ids.add(v["video_id"])

                comments = get_comments(v, dessert, max_comments=100)
                all_comments.extend(comments)

        dessert_count = sum(1 for c in all_comments if c["dessert"] == dessert)
        print(f"-> [{dessert}] 수집 완료: 영상 {len(seen_video_ids)}개 / 댓글 {dessert_count}개")

    raw_df = pd.DataFrame(all_comments)
    clean_df = anonymize(raw_df)

    # 저장 전 개인정보 컬럼 없는지 눈으로 확인 (가이드 4-4 필수 체크)
    print("\n[저장 직전 최종 컬럼 확인]")
    print(clean_df.columns.tolist())

    os.makedirs("data", exist_ok=True)
    clean_df.to_csv("data/comments.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: data/comments.csv (총 {len(clean_df)}개 댓글)")

    # 디저트 x 연도별 댓글 수 확인 (가이드 4-3-5)
    clean_df["year"] = clean_df["comment_date"].str[:4]
    print("\n[디저트 × 연도별 댓글 수 분포]")
    print(pd.crosstab(clean_df["dessert"], clean_df["year"]))