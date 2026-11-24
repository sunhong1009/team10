import pandas as pd
from analysis import clean_comments, add_sentiment, sentiment_ratio, smooth_trend, find_peak_and_half, label_period

# 1. 50개 검증셋 일치율 재측정
sample50 = pd.read_csv("data/sample50_check.csv")
lex = pd.read_csv("data/sentiment_lexicon.csv")

human_labels = [
    "긍정", "긍정", "긍정", "중립", "긍정", "긍정", "중립", "긍정", "긍정", "긍정",
    "긍정", "긍정", "부정", "중립", "긍정", "중립", "중립", "중립", "부정", "긍정",
    "긍정", "부정", "부정", "긍정", "긍정", "긍정", "부정", "긍정", "긍정", "긍정",
    "긍정", "중립", "긍정", "긍정", "긍정", "중립", "부정", "긍정", "긍정", "중립",
    "부정", "긍정", "긍정", "긍정", "긍정", "중립", "중립", "중립", "긍정", "긍정"
]

sample50_after = add_sentiment(sample50[["dessert", "text"]], lex)
sample50_after["human"] = human_labels
acc = (sample50_after["sentiment"] == sample50_after["human"]).mean() * 100
matches = (sample50_after["sentiment"] == sample50_after["human"]).sum()
print(f"보강 후 일치율: {acc:.1f}% ({matches}/50건)")

# 2. 전체 데이터 감성 비율 재산출
trend = smooth_trend(pd.read_csv("data/trend.csv"))
cycle = find_peak_and_half(trend)
comments = label_period(pd.read_csv("data/comments.csv"), trend, cycle)
comments = add_sentiment(comments, lex)
print("\n=== 보강 후 디저트·시기별 감성 비율 ===")
print(sentiment_ratio(comments).to_string(index=False))