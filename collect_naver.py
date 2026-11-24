import os
import requests
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from config import DESSERTS, KEYWORD_GROUPS, COLORS

# .env에서 API 키 불러오기
load_dotenv()


def get_search_trend(keyword_groups, start_date="2022-01-01", end_date="2026-11-15", time_unit="week"):
    url = "https://naverapihub.apigw.ntruss.com/search-trend/v1/search"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": os.getenv("NAVER_CLIENT_ID"),
        "X-NCP-APIGW-API-KEY": os.getenv("NAVER_CLIENT_SECRET"),
        "Content-Type": "application/json",
    }

    # 5개 디저트 묶음을 한 요청에 구성
    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": time_unit,
        "keywordGroups": [
            {"groupName": dessert, "keywords": keywords}
            for dessert, keywords in keyword_groups.items()
        ],
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    results = response.json().get("results", [])

    # 응답을 (date, dessert, ratio) 행 단위로 펼치기
    rows = []
    for item in results:
        dessert_name = item["title"]
        for point in item["data"]:
            rows.append({
                "date": point["period"],
                "dessert": dessert_name,
                "ratio": float(point["ratio"]),
            })

    df = pd.DataFrame(rows)

    # 빠진 주가 있으면 0으로 채워 5개 디저트의 행 수를 동일하게 맞춤
    all_dates = sorted(df["date"].unique())
    full_index = pd.MultiIndex.from_product([all_dates, DESSERTS], names=["date", "dessert"])
    df = df.set_index(["date", "dessert"]).reindex(full_index, fill_value=0.0).reset_index()

    return df


if __name__ == "__main__":
    trend_df = get_search_trend(KEYWORD_GROUPS, start_date="2022-01-01", end_date="2026-11-15", time_unit="week")

    # CSV 저장
    os.makedirs("data", exist_ok=True)
    trend_df.to_csv("data/trend.csv", index=False, encoding="utf-8-sig")
    print("저장 완료: data/trend.csv")

    # 검증 출력 (가이드 4-2 확인 사항)
    print("\n[디저트별 행 수 확인]")
    print(trend_df["dessert"].value_counts())
    print("\n[ratio 최댓값(100) 위치 확인]")
    print(trend_df[trend_df["ratio"] == trend_df["ratio"].max()])

    # 팀 공유용 꺾은선 그래프 띄우기
    fig = px.line(trend_df, x="date", y="ratio", color="dessert", color_discrete_map=COLORS, title="디저트 5종 검색어 트렌드")
    fig.show()