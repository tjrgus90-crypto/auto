"""
멜론 실시간 차트의 순위와 좋아요 수 사이의 관계 분석 (선형회귀)

1. 멜론 실시간 차트(https://www.melon.com/chart/index.htm)에서 1~10위 곡 정보를 수집
2. 각 곡의 좋아요 수를 1,000개 단위로 환산 (예: 168,003 -> 168)
3. sklearn LinearRegression으로 x(순위) -> y(좋아요 수) 회귀 모델 학습
4. 실제 값(파란 점)과 회귀선(붉은 선)을 그려 png로 저장

실행 예:
    python melon_like_regression.py
    python melon_like_regression.py --output melon_regression.png
    # 크롤링이 막힌 경우 직접 수집한 값으로 실행
    python melon_like_regression.py --manual 168 150 ... --chart-time "2025.09.19 19:00"
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression

CHART_URL = "https://www.melon.com/chart/index.htm"
LIKE_URL = "https://www.melon.com/commonlike/getSongLike.json"

# 멜론은 브라우저가 아닌 요청(User-Agent 없음)을 차단하므로 헤더를 지정한다.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": CHART_URL,
}

TOP_N = 10


def fetch_chart(top_n=TOP_N):
    """실시간 차트 페이지에서 상위 top_n곡의 (순위, 곡ID, 제목, 가수)와 차트 일시를 가져온다."""
    resp = requests.get(CHART_URL, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    songs = []
    for rank, row in enumerate(soup.select("tr.lst50")[:top_n], start=1):
        songs.append({
            "rank": rank,
            "song_id": row["data-song-no"],
            "title": row.select_one("div.ellipsis.rank01 a").get_text(strip=True),
            "artist": row.select_one("div.ellipsis.rank02 a").get_text(strip=True),
        })

    # 차트 상단의 기준 일시 (예: "2025.09.19" / "19:00")
    year = soup.select_one("span.year")
    hour = soup.select_one("span.hour")
    chart_time = None
    if year is not None and hour is not None:
        chart_time = f"{year.get_text(strip=True)} {hour.get_text(strip=True)}"

    return songs, chart_time


def fetch_likes(song_ids):
    """좋아요 수는 차트 HTML에 없고 별도 JSON API로 비동기 로딩되므로 직접 호출한다."""
    resp = requests.get(
        LIKE_URL,
        params={"contsIds": ",".join(song_ids)},
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    like_map = {str(item["CONTSID"]): int(item["SUMMCNT"])
                for item in resp.json()["contsLike"]}
    return [like_map[sid] for sid in song_ids]


def train(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model


def plot_result(x, y, y_pred, chart_time, output):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="blue", label="Actual")
    plt.plot(x, y_pred, color="red", label="Linear Regression")
    plt.title(chart_time)
    plt.xlabel("Rank")
    plt.ylabel("Likes (x1,000)")
    plt.xticks(x.ravel())
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"그래프 저장: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="멜론 실시간 차트 순위-좋아요 수 선형회귀")
    parser.add_argument("--output", default="melon_regression.png", help="저장할 그림 파일 이름")
    parser.add_argument("--manual", type=int, nargs=TOP_N, metavar="LIKES",
                        help="크롤링 대신 1~10위 좋아요 수(1,000개 단위)를 직접 입력")
    parser.add_argument("--chart-time", help='그래프 제목에 쓸 차트 일시 (예: "2025.09.19 19:00")')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.manual:
        likes_k = args.manual
        chart_time = args.chart_time
        if chart_time is None:
            sys.exit("--manual 사용 시 --chart-time 으로 차트 일시를 지정해야 합니다.")
    else:
        songs, chart_time = fetch_chart()
        likes = fetch_likes([s["song_id"] for s in songs])
        # 1,000개 단위로 기록 (168,003 -> 168)
        likes_k = [n // 1000 for n in likes]
        chart_time = args.chart_time or chart_time

        print(f"[멜론 실시간 차트 {chart_time}]")
        for s, n in zip(songs, likes):
            print(f"{s['rank']:>2}위  {s['title']} - {s['artist']}  좋아요 {n:,}")
        print()

    x = np.array(range(1, TOP_N + 1)).reshape(-1, 1)
    y = np.array(likes_k)
    print("x =", x.ravel().tolist())
    print("y =", y.tolist())

    model = train(x, y)
    y_pred = model.predict(x)

    print(f"\n기울기(coef)    : {model.coef_[0]:.4f}")
    print(f"절편(intercept) : {model.intercept_:.4f}")
    print(f"결정계수(R^2)   : {model.score(x, y):.4f}")

    plot_result(x, y, y_pred, chart_time, args.output)


if __name__ == "__main__":
    main()
