import streamlit as st
import pandas as pd
import speech_recognition as sr
from textblob import TextBlob
import requests
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import plotly.express as px

# Set up Google Sheets credentials with google-auth
SCOPE = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
CREDENTIALS = Credentials.from_service_account_file(
    'ai-sales-call-assistant-447217-d083604834bf.json', scopes=SCOPE
)
gc = gspread.authorize(CREDENTIALS)

# Connect to Google Sheets
sheet = gc.open("Real-Time Speech Analysis").sheet1  # Replace with your Google Sheets name

# Cache function for fetching data from Google Sheets
@st.cache
def get_sheet_data():
    data = pd.DataFrame(sheet.get_all_records())
    return data

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"

# Load CRM data from file
CRM_DATA_PATH = "D:/Real-Time AI/laptop_data.csv"
crm_data = pd.read_csv(CRM_DATA_PATH)

# Sentiment analysis using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, round(polarity, 2)

# Process audio to text using SpeechRecognition
def process_audio_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""
        except sr.WaitTimeoutError:
            return ""

# Query Groq API for product recommendations
def query_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        st.error(f"Groq API Error: {response.status_code} - {response.text}")
        return "No information available"

# Recommend products based on user query
def recommend_products(query):
    recommendations = []
    if "price of" in query.lower():
        product_name = query.lower().replace("what is the price of", "").strip()
        match = crm_data[crm_data['Model'].str.lower() == product_name]
        if not match.empty:
            product = match.iloc[0]
            recommendations.append(f"The price of {product['Model']} is ₹{product['Price']}.")
    elif "gaming laptop under" in query.lower():
        price_limit = int(''.join(filter(str.isdigit, query)))
        matches = crm_data[(crm_data['Category'].str.lower() == "gaming") & (crm_data['Price'] <= price_limit)]
        if not matches.empty:
            recommendations = [f"{row['Model']} for ₹{row['Price']}" for _, row in matches.iterrows()]
    elif "should i buy" in query.lower():
        product_name = query.lower().replace("should i buy", "").strip()
        match = crm_data[crm_data['Model'].str.lower() == product_name]
        if not match.empty and match.iloc[0]['Category'].lower() != "gaming":
            alternatives = crm_data[crm_data['Category'].str.lower() == "gaming"]
            alt_recommendations = [f"{row['Model']} for ₹{row['Price']}" for _, row in alternatives.iterrows()]
            recommendations.append(f"{product_name} is not ideal for gaming. Consider: " + ", ".join(alt_recommendations))
    else:
        prompt = f"Based on the query: {query}, suggest relevant products."
        response = query_groq(prompt)
        recommendations.append(response)

    if recommendations:
        short_summary = "Here are some suggestions: " + ", ".join(recommendations[:2])
        return short_summary, recommendations[0]
    return "Sorry, I couldn't find any recommendations.", "No information available"

# Handle customer objections
def handle_objections(user_input):
    objections = {
        "too expensive": "We understand budget is important. Would you like to explore more affordable options?",
        "better deal": "Let me help you find the best deal available. Are you open to refurbished models?",
        "not sure about the specs": "Could you share more about your needs? I can recommend the right specs for you.",
    }
    for key, response in objections.items():
        if key in user_input.lower():
            return response
    return "I understand your concern. Can you tell me more about your hesitation?"

# Generate dynamic follow-up questions
def generate_questions(user_input):
    prompt = f"Based on the user input: {user_input}, generate a dynamic follow-up question to guide the sales process."
    response = query_groq(prompt)
    return response

# Store query data in Google Sheets
def store_data_in_sheets(user_input, sentiment, score, recommendation, objection, dqg_response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        sheet.append_row([timestamp, user_input, sentiment, score, recommendation, objection, dqg_response])
        st.info("Data successfully logged into Google Sheets.")
    except Exception as e:
        st.error(f"Failed to store data in Google Sheets: {e}")

# Streamlit Interface
st.title("Real-Time Sales Call Assistant")

menu = ["Query Analysis", "Dashboard"]
selection = st.sidebar.selectbox("Choose an option", menu)

if selection == "Query Analysis":
    if "running" not in st.session_state:
        st.session_state.running = False

    if st.button("Start Listening"):
        st.session_state.running = True
        st.info("Listening...")

    while st.session_state.running:
        user_input = process_audio_to_text()
        if user_input:
            if "stop" in user_input.lower():
                st.session_state.running = False
                st.info("Stopping real-time analysis.")
            else:
                sentiment, score = analyze_sentiment(user_input)
                st.write(f"Transcript: {user_input}")
                st.write(f"Sentiment: {sentiment} (Score: {score})")
                
                recommendation, summary = recommend_products(user_input)
                objection = handle_objections(user_input)
                dqg_response = generate_questions(user_input)
                
                st.write(f"Recommendation: {recommendation}")
                st.write(f"Objection Handling: {objection}")
                st.write(f"Dynamic Question: {dqg_response}")
                
                store_data_in_sheets(user_input, sentiment, score, recommendation, objection, dqg_response)

elif selection == "Dashboard":
    st.subheader("Post-Call Summary")
    with st.spinner("Loading data..."):
        data = get_sheet_data()

    if st.button("Show Enhanced Dashboard"):
        sentiment_counts = data['Sentiment'].value_counts()
        sentiment_fig = px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values,
                                title="Sentiment Distribution")
        st.plotly_chart(sentiment_fig)
