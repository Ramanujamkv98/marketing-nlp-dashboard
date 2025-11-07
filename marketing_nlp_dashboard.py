import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bertopic import BERTopic

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Marketing NLP Dashboard", layout="wide")

st.title("💬 Voice of Customer Analytics")
st.markdown("Upload a CSV with columns: **Date, Platform, Comment** to analyze sentiment, topics, and insights.")

# ---------- LAZY MODEL LOADERS ----------
@st.cache_resource
def load_sentiment_model():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@st.cache_resource
def load_summarizer():
    from transformers import pipeline
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_topic_model():
    return BERTopic(language="english", verbose=False)

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = {'Date', 'Platform', 'Comment'}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain columns: {', '.join(required_cols)}")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Comment'], inplace=True)
    df['Comment'] = df['Comment'].astype(str)

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Sentiment Dashboard", "🧠 Topic Modeling", "📝 Summarization"])

    # ===================================
    # TAB 1: SENTIMENT ANALYSIS
    # ===================================
    with tab1:
        st.header("📊 Sentiment Analysis")
        with st.spinner("Analyzing sentiment..."):
            sentiment_analyzer = load_sentiment_model()
            df['sentiment_result'] = df['Comment'].apply(
                lambda x: sentiment_analyzer(x[:512])[0]['label']
            )

        df['sentiment_result'] = df['sentiment_result'].map({
            'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'
        }).fillna(df['sentiment_result'])

        # KPIs
        col1, col2, col3 = st.columns(3)
        total = len(df)
        positive = (df['sentiment_result'] == 'Positive').sum()
        negative = (df['sentiment_result'] == 'Negative').sum()
        neutral = total - positive - negative
        col1.metric("Total Comments", total)
        col2.metric("Positive", f"{(positive/total)*100:.1f}%")
        col3.metric("Negative", f"{(negative/total)*100:.1f}%")

        # Charts
        fig1 = px.histogram(df, x='sentiment_result', color='sentiment_result',
                            title="Sentiment Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df, x='Platform', color='sentiment_result', barmode='group',
                            title="Sentiment by Platform")
        st.plotly_chart(fig2, use_container_width=True)

        # WordCloud
        st.subheader("🔍 WordCloud")
        text = " ".join(df['Comment'].tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig_wc, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

        # Trend
        st.subheader("⏱ Sentiment Trend Over Time")
        daily = df.groupby(['Date', 'sentiment_result']).size().reset_index(name='Count')
        fig3 = px.line(daily, x='Date', y='Count', color='sentiment_result')
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("📋 View Data"):
            st.dataframe(df)

    # ===================================
    # TAB 2: TOPIC MODELING
    # ===================================
    with tab2:
        st.header("🧠 Topic Modeling using BERTopic")
        st.markdown("Discover key discussion themes and marketing insights.")

        with st.spinner("Running BERTopic... this may take 1–2 minutes ⏳"):
            topic_model = load_topic_model()
            topics, probs = topic_model.fit_transform(df['Comment'])

        df['Topic'] = topics
        topic_info = topic_model.get_topic_info()

        st.subheader("📋 Top Topics")
        st.dataframe(topic_info[['Topic', 'Count', 'Name']].head(10))

        # Bar chart
        fig_topics = px.bar(topic_info.head(10), x='Name', y='Count',
                            title="Top Topics in Conversations")
        st.plotly_chart(fig_topics, use_container_width=True)

        # WordCloud for top topic
        st.subheader("☁️ WordCloud of Top Topic")
        if len(topic_info) > 1:
            top_topic = topic_info.iloc[1]['Topic']
            words = " ".join([w for w, _ in topic_model.get_topic(top_topic)])
            wc = WordCloud(width=800, height=400, background_color='white').generate(words)
            fig_tw, ax2 = plt.subplots()
            ax2.imshow(wc, interpolation='bilinear')
            ax2.axis("off")
            st.pyplot(fig_tw)

        with st.expander("📊 Topic Details"):
            for i in range(min(5, len(topic_info))):
                topic_name = topic_info.iloc[i]['Name']
                st.markdown(f"**{i+1}. {topic_name}**")
                st.write(topic_model.get_topic(i))

    # ===================================
    # TAB 3: SUMMARIZATION
    # ===================================
    with tab3:
        st.header("📝 Executive Summary Generator")
        st.markdown("Automatically summarize customer sentiment and trending topics for marketing reports.")

        with st.spinner("Generating insights summary..."):
            summarizer = load_summarizer()

            positive_text = " ".join(df[df['sentiment_result'] == 'Positive']['Comment'].tolist()[:100])
            negative_text = " ".join(df[df['sentiment_result'] == 'Negative']['Comment'].tolist()[:100])

            if positive_text:
                pos_summary = summarizer(positive_text, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
            else:
                pos_summary = "Not enough positive comments to summarize."

            if negative_text:
                neg_summary = summarizer(negative_text, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
            else:
                neg_summary = "Not enough negative comments to summarize."

        st.subheader("😊 Positive Sentiment Summary")
        st.write(pos_summary)

        st.subheader("😞 Negative Sentiment Summary")
        st.write(neg_summary)

        st.info("You can copy these summaries directly into your marketing report or campaign brief.")

else:
    st.info("📁 Upload your CSV to begin analysis.")
