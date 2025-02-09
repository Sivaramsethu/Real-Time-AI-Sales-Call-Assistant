import webrtc_streamer, WebRtcMode, AudioProcessorBase
from sentiment_analysis import analyze_sentiment, transcribe_with_chunks
from product_recommender import ProductRecommender
from objection_handler import ObjectionHandler
from google_sheets import fetch_call_data, store_data_in_sheet
from sentence_transformers import SentenceTransformer
import re
import uuid
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import queue
import threading

# Initialize components
objection_handler = ObjectionHandler("objections.csv")
product_recommender = ProductRecommender("recommendations.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Queue to hold transcribed text
transcription_queue = queue.Queue()

def generate_comprehensive_summary(chunks):
    # Your existing function implementation
    pass

def is_valid_input(text):
    # Your existing function implementation
    pass

def is_relevant_sentiment(sentiment_score):
    # Your existing function implementation
    pass

def calculate_overall_sentiment(sentiment_scores):
    # Your existing function implementation
    pass

def handle_objection(text):
    query_embedding = model.encode([text])
    distances, indices = objection_handler.index.search(query_embedding, 1)
    if distances[0][0] < 1.5:
        responses = objection_handler.handle_objection(text)
        return "\n".join(responses) if responses else "No objection response found."
    return "No objection response found."

class AudioProcessor(AudioProcessorBase):
    def _init_(self):
        self.sr = 16000  # Sample rate
        self.q = transcription_queue

    def recv(self, frame):
        audio_data = frame.to_ndarray()
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()  # Convert to int16 format
        
        print(f"Audio data shape: {audio_data.shape}")
        print(f"Audio data sample: {audio_data[:10]}")
        
        text = self.transcribe_audio(audio_bytes)
        if text:
            self.q.put(text)

        return frame

    def transcribe_audio(self, audio_bytes):
        try:
            chunks = transcribe_with_chunks({})
            if chunks:
                return chunks[-1][0]
        except Exception as e:
            print(f"Error transcribing audio: {e}")
        return None

def real_time_analysis():
    st.info("Listening... Say 'stop' to end the process.")

    webrtc_ctx = webrtc_streamer(
        key="real-time-audio",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if webrtc_ctx.state.playing:
        while not transcription_queue.empty():
            text = transcription_queue.get()
            st.write(f"Recognized Text: {text}")

            sentiment, score = analyze_sentiment(text)
            st.write(f"Sentiment: {sentiment} (Score: {score})")

            objection_response = handle_objection(text)
            st.write(f"Objection Response: {objection_response}")

            recommendations = []
            if is_valid_input(text) and is_relevant_sentiment(score):
                query_embedding = model.encode([text])
                distances, indices = product_recommender.index.search(query_embedding, 1)

                if distances[0][0] < 1.5:
                    recommendations = product_recommender.get_recommendations(text)

            if recommendations:
                st.write("Product Recommendations:")
                for rec in recommendations:
                    st.write(rec)

def fetch_data_and_display():
    try:
        st.header("Call Summaries and Sentiment Analysis")
        data = fetch_call_data("Real-Time Speech Analysis")  # Updated Google Sheet Name
        
        print(f"Fetched data: {data}")  # Log fetched data
        
        if data.empty:
            st.warning("No data available in the Google Sheet.")
        else:
            sentiment_counts = data['Sentiment'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sentiment Distribution")
                fig_pie = px.pie(
                    values=sentiment_counts.values, 
                    names=sentiment_counts.index, 
                    title='Call Sentiment Breakdown',
                    color_discrete_map={
                        'POSITIVE': 'green', 
                        'NEGATIVE': 'red', 
                        'NEUTRAL': 'blue'
                    }
                )
                st.plotly_chart(fig_pie)

            with col2:
                st.subheader("Sentiment Counts")
                fig_bar = px.bar(
                    x=sentiment_counts.index, 
                    y=sentiment_counts.values, 
                    title='Number of Calls by Sentiment',
                    labels={'x': 'Sentiment', 'y': 'Number of Calls'},
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'POSITIVE': 'green', 
                        'NEGATIVE': 'red', 
                        'NEUTRAL': 'blue'
                    }
                )
                st.plotly_chart(fig_bar)

            st.subheader("All Calls")
            display_data = data.copy()
            display_data['Summary Preview'] = display_data['Summary'].str[:100] + '...'
            st.dataframe(display_data[['Call ID', 'Chunk', 'Sentiment', 'Summary Preview', 'Overall Sentiment']])

            unique_call_ids = data[data['Call ID'] != '']['Call ID'].unique()
            call_id = st.selectbox("Select a Call ID to view details:", unique_call_ids)

            call_details = data[data['Call ID'] == call_id]
            if not call_details.empty:
                st.subheader("Detailed Call Information")
                st.write(f"*Call ID:* {call_id}")
                st.write(f"*Overall Sentiment:* {call_details.iloc[0]['Overall Sentiment']}")

                st.subheader("Full Call Summary")
                st.text_area("Summary:", 
                             value=call_details.iloc[0]['Summary'], 
                             height=200, 
                             disabled=True)

                st.subheader("Conversation Chunks")
                for _, row in call_details.iterrows():
                    if pd.notna(row['Chunk']):  
                        st.write(f"*Chunk:* {row['Chunk']}")
                        st.write(f"*Sentiment:* {row['Sentiment']}")
                        st.write("---")
            else:
                st.error("No details available for the selected Call ID.")
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")

def run_app():
    st.set_page_config(page_title="Sales Call Assistant", layout="wide")
    st.title("AI Sales Call Assistant")

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a mode:", ["Real-Time Call Analysis", "Dashboard"])

    if app_mode == "Real-Time Call Analysis":
        st.header("Real-Time Sales Call Analysis")
        if st.button("Start Listening"):
            real_time_analysis()

    elif app_mode == "Dashboard":
        st.header("Call Summaries and Sentiment Analysis")
        fetch_data_and_display()

if _name_ == "_main_":
    run_app()