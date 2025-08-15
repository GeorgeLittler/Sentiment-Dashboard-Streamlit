import time
from datetime import datetime, timezone
import feedparser
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
import altair as alt

# Ensure VADER is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

SOURCES = {
    "BBC News (Top stories)": "https://feeds.bbci.co.uk/news/rss.xml",
    "Reuters (World)": "https://feeds.reuters.com/reuters/worldNews",
    "The Guardian (UK)": "https://www.theguardian.com/uk/rss",
}

TITLE_MAP = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
ORDER_TITLE = ["Positive", "Neutral", "Negative"]

@st.cache_data(ttl=300)  # refresh every 5 minutes
def fetch_headlines(sources: dict) -> pd.DataFrame:
    """Fetch recent headlines from RSS sources and normalise timestamps."""
    rows = []
    for name, url in sources.items():
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
        for e in feed.entries[:50]:  # cap per source
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "")
            published = getattr(e, "published", getattr(e, "updated", ""))
            if title:
                rows.append(
                    {"source": name, "title": title, "link": link, "published": published}
                )
    df = pd.DataFrame(rows).drop_duplicates(subset=["title"])

    if not df.empty:
        dt = pd.to_datetime(df["published"], errors="coerce", utc=True)
        df["imputed_time"] = dt.isna()  # remember which ones were missing
        now_utc = pd.Timestamp.now(tz="UTC")
        df["published_dt"] = dt.fillna(now_utc)
    else:
        df["imputed_time"] = []
        df["published_dt"] = []
    return df

def score_sentiment(df: pd.DataFrame, neg_thresh: float, pos_thresh: float) -> pd.DataFrame:
    """Compute VADER scores + labels with adjustable thresholds."""
    sia = SentimentIntensityAnalyzer()
    scores = df["title"].apply(sia.polarity_scores).apply(pd.Series)
    out = pd.concat([df, scores], axis=1)

    def lab(c):
        if c <= neg_thresh:
            return "negative"
        if c >= pos_thresh:
            return "positive"
        return "neutral"

    out["label"] = out["compound"].apply(lab)
    out["Label"] = out["label"].map(TITLE_MAP)  # Title-case for display
    return out

def main():
    st.set_page_config(page_title="Real-Time News Sentiment Dashboard", layout="wide")
    st.title("Real-Time News Sentiment Dashboard")
    st.caption("Live news headlines via RSS • VADER sentiment")

    with st.sidebar:
        st.header("Controls")
        selected = {name: url for name, url in SOURCES.items() if st.checkbox(name, value=True)}
        st.markdown("---")
        kw = st.text_input("Keyword Filter (optional)", placeholder="e.g. Elections, Climate, AI").strip()
        st.markdown("---")
        st.subheader("Sentiment Thresholds")
        col_a, col_b = st.columns(2)
        neg_thresh = col_a.slider("Negative ≤", min_value=-1.00, max_value=0.00, value=-0.05, step=0.01)
        pos_thresh = col_b.slider("Positive ≥", min_value=0.00, max_value=1.00, value=0.05, step=0.01)

        st.subheader("Time Window & Granularity")
        lookback_hours = st.slider("Lookback (hours)", 1, 72, 24)
        bin_size = st.selectbox("Bin Size", ["1min", "5min", "15min", "30min", "1H"], index=1)
        exclude_imputed = st.checkbox("Exclude undated items (imputed 'now' timestamps)", value=True)

        st.markdown("---")
        refresh = st.button("Fetch Latest Headlines")
        st.caption("Cache auto-refreshes every 5 min. Use ‘Fetch Latest Headlines’ to update immediately.")

    if not selected:
        st.info("Select at least one source on the left.")
        st.stop()

    if refresh:
        fetch_headlines.clear()

    with st.spinner("Fetching headlines…"):
        df = fetch_headlines(selected)
        time.sleep(0.15)

    if df.empty:
        st.warning("No headlines fetched.")
        st.stop()

    # Keyword filter (case-insensitive)
    if kw:
        mask = df["title"].str.contains(kw, case=False, na=False)
        df = df[mask]
        if df.empty:
            st.warning(f"No headlines matched **{kw}**.")
            st.stop()

    scored = score_sentiment(df, neg_thresh=neg_thresh, pos_thresh=pos_thresh)

    # Header metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Headlines", len(scored))
    col2.metric("Avg Compound", f"{scored['compound'].mean():.3f}")
    col3.metric("Positive", int((scored['label'] == 'positive').sum()))
    col4.metric("Neutral", int((scored['label'] == 'neutral').sum()))
    col5.metric("Negative", int((scored['label'] == 'negative').sum()))

    # Distribution (Altair) — Title Case labels
    st.subheader("Sentiment Distribution")
    counts = scored["Label"].value_counts().reindex(ORDER_TITLE).fillna(0).astype(int)
    chart_df = counts.rename_axis("Sentiment").reset_index(name="Headlines")
    bar = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Sentiment:N", sort=ORDER_TITLE, title="Sentiment"),
        y=alt.Y("Headlines:Q", title="Headlines"),
        tooltip=["Sentiment", "Headlines"]
    )
    st.altair_chart(bar, use_container_width=True)

    # Time series — windowed, binned, smoothed (Title Case axes)
    st.subheader("Mean Sentiment Over Time (by Source)")
    now_utc = pd.Timestamp.now(tz="UTC")
    cutoff = now_utc - pd.Timedelta(hours=lookback_hours)

    tmp = scored[(scored["published_dt"] >= cutoff) & (scored["published_dt"] <= now_utc)].copy()
    if exclude_imputed and "imputed_time" in tmp.columns:
        tmp = tmp[~tmp["imputed_time"]]

    if not tmp.empty:
        tmp["bucket"] = tmp["published_dt"].dt.floor(bin_size)
        ts = tmp.groupby(["source", "bucket"], as_index=False)["compound"].mean()
        ts["smoothed"] = ts.groupby("source")["compound"].transform(lambda s: s.rolling(3, min_periods=1).mean())

        if not ts.empty and ts["bucket"].notna().any():
            line = alt.Chart(ts).mark_line(point=True).encode(
                x=alt.X("bucket:T", title="Published (UTC)"),
                y=alt.Y("smoothed:Q", title="Mean Compound (Smoothed)"),
                color=alt.Color("source:N", title="Source"),
                tooltip=["source", "bucket:T", alt.Tooltip("compound:Q", format=".3f", title="Mean Compound")]
            )
            st.altair_chart(line, use_container_width=True)
        else:
            st.caption("No valid timestamps available to plot.")
    else:
        st.caption("No recent data to plot for the selected window.")

    # Per-source breakdown
    st.subheader("Per-Source Breakdown")
    by_src = scored.groupby("source").agg(
        Headlines=("title", "count"),
        Avg_Compound=("compound", "mean"),
        Positives=("label", lambda s: (s == "positive").sum()),
        Negatives=("label", lambda s: (s == "negative").sum()),
    ).reset_index().sort_values("Avg_Compound", ascending=False)
    st.dataframe(by_src.rename(columns={"source": "Source"}), use_container_width=True)

    # Table of headlines (show Title Case label)
    st.subheader("Headlines")
    show_cols = ["source", "published", "compound", "Label", "title", "link"]
    st.dataframe(
        scored[show_cols].sort_values("compound", ascending=False),
        use_container_width=True,
        column_config={
            "source": "Source",
            "published": "Published",
            "compound": st.column_config.NumberColumn("Compound", format="%.3f"),
            "Label": "Sentiment",
            "title": "Title",
            "link": st.column_config.LinkColumn("Link"),
        },
        hide_index=True
    )

    # Export
    csv = scored.drop(columns=["label"]).to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="sentiment_headlines.csv", mime="text/csv")

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} (UTC)")

if __name__ == "__main__":
    main()
