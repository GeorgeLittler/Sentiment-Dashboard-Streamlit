import time
import feedparser
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st

# Ensure VADER is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

SOURCES = {
    "BBC News (Top stories)": "http://feeds.bbci.co.uk/news/rss.xml",
    "Reuters (World)": "http://feeds.reuters.com/reuters/worldNews",
    "The Guardian (UK)": "https://www.theguardian.com/uk/rss",
}

@st.cache_data(ttl=300)  # refresh every 5 minutes
def fetch_headlines(sources: dict) -> pd.DataFrame:
    rows = []
    for name, url in sources.items():
        feed = feedparser.parse(url)
        for e in feed.entries[:50]:  # cap per source
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "")
            published = getattr(e, "published", getattr(e, "updated", ""))
            if title:
                rows.append({"source": name, "title": title, "link": link, "published": published})
    df = pd.DataFrame(rows).drop_duplicates(subset=["title"])
    return df

def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    scores = df["title"].apply(sia.polarity_scores).apply(pd.Series)
    out = pd.concat([df, scores], axis=1)
    # label class by compound threshold
    out["label"] = pd.cut(out["compound"], bins=[-1.0, -0.05, 0.05, 1.0], labels=["negative", "neutral", "positive"])
    return out

def main():
    st.set_page_config(page_title="Sentiment Dashboard (Round 1)", page_icon="🗞️", layout="wide")
    st.title("🗞️ Real‑Time Sentiment Dashboard (Round 1)")
    st.caption("Live news headlines via RSS • VADER sentiment • First cut")

    with st.sidebar:
        st.header("Sources")
        selected = {name:url for name,url in SOURCES.items() if st.checkbox(name, value=True)}
        refresh = st.button("Refresh")
        st.markdown("---")
        st.caption("Tip: Round 2 will add time series, model upgrades, and export.")

    if not selected:
        st.info("Select at least one source on the left.")
        st.stop()

    if refresh:
        # bust cache
        fetch_headlines.clear()

    with st.spinner("Fetching headlines..."):
        df = fetch_headlines(selected)
        time.sleep(0.2)

    if df.empty:
        st.warning("No headlines fetched.")
        st.stop()

    scored = score_sentiment(df)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Headlines", len(scored))
    col2.metric("Avg compound", f"{scored['compound'].mean():.3f}")
    col3.metric("Positive", int((scored['label']=='positive').sum()))
    col4.metric("Negative", int((scored['label']=='negative').sum()))

    # Distribution
    counts = scored["label"].value_counts().reindex(["positive","neutral","negative"]).fillna(0).astype(int)
    chart_df = counts.reset_index().rename(columns={"index":"sentiment","label":"count"})
    st.subheader("Sentiment distribution")
    st.bar_chart(chart_df.set_index("sentiment"))

    # Table of headlines
    st.subheader("Headlines")
    show_cols = ["source","published","compound","label","title","link"]
    st.dataframe(scored[show_cols].sort_values("compound", ascending=False), use_container_width=True)

    # Clickable links note
    st.caption("Open links from the table via the context menu (copy link) — Round 2 will add clickable cells.")

if __name__ == "__main__":
    main()
