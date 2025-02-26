import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import os

# Google Gemini API setup
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")  # Store this in Streamlit secrets
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"

# Function to send request to Google Gemini API
def query_gemini_api(prompt):
    headers = {"Content-Type": "application/json"}
    params = {"key": GOOGLE_GEMINI_API_KEY}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    
    response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"API Error! Details: {response.text}"

# Mapping function for IPE Score to Job Level
def map_ipe_to_level(ipe_score):
    mapping = {
        range(40, 42): 1,
        range(42, 44): 2,
        range(44, 46): 3,
        range(46, 48): 4,
        range(48, 51): 5,
        range(51, 53): 6,
        range(53, 56): 7,
        range(56, 58): 8,
        range(58, 60): 9,
        range(60, 62): 10,
        range(62, 66): 11,
        range(66, 74): 12,
    }
    for score_range, level in mapping.items():
        if ipe_score in score_range:
            return level
    return "Unknown"

# Streamlit Web App
def main():
    st.set_page_config(page_title="Job Evaluation AI", layout="wide")
    st.title("📊 AI-Powered Job Evaluation Chatbot")
    st.write("Upload a job description and get an IPE Level & Job Architecture suggestion.")
    
    job_description = st.text_area("📌 Paste Job Description Here:")
    
    if st.button("🚀 Evaluate Job"):        
        if job_description:
            result = evaluate_job(job_description)
            st.subheader("🏆 Job Evaluation Result:")
            st.write(result)
        else:
            st.warning("⚠️ Please enter a job description.")

def evaluate_job(job_desc):
    """AI-powered function to analyze job complexity and assign an IPE level."""
    
    prompt = f"""
    You are an HR expert specializing in job evaluation using the IPE methodology and the company's Job Architecture framework.
    Your task is to carefully analyze the job description below and determine the correct IPE level.
    
    **Job Description:**
    {job_desc}
    
    **Assign the following values:**
    - **IPE Total Score (40-73 scale), considering both job responsibilities and complexity. Avoid overrating.**
    - **Mapped Job Level (1-12) using this defined structure:**
      - 40-41 → Level 1
      - 42-43 → Level 2
      - 44-45 → Level 3
      - 46-47 → Level 4
      - 48-50 → Level 5
      - 51-52 → Level 6
      - 53-55 → Level 7
      - 56-57 → Level 8
      - 58-59 → Level 9
      - 60-61 → Level 10
      - 62-65 → Level 11
      - 66-73 → Level 12
    - **Brief Justification (1-2 sentences only), explaining why the job fits the assigned level.**
    
    **Return ONLY in this format:**
    IPE Total Score: X
    Mapped Level: Y
    Justification: (Short and clear statement)
    
    Do NOT include unnecessary details, do NOT repeat the job description, and do NOT explain the process. Keep the answer concise.
    """
    
    response = query_gemini_api(prompt)
    
    # Extract IPE score from response
    try:
        ipe_score = int(response.split("IPE Total Score:")[1].split("\n")[0].strip())
        mapped_level = map_ipe_to_level(ipe_score)
        response = response.replace(f"Mapped Level: {ipe_score}", f"Mapped Level: {mapped_level}")
    except Exception:
        pass  # If parsing fails, keep the original response
    
    return response

if __name__ == "__main__":
    main()
