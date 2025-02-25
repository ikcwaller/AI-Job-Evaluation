import openai
import streamlit as st
import pandas as pd

# Load IPE criteria from the Excel file
file_path = "/mnt/data/IPE_Calculator_HFG_HH.xlsm"
xls = pd.ExcelFile(file_path)

impact_df = pd.read_excel(xls, sheet_name="Impact")
communication_df = pd.read_excel(xls, sheet_name="Communication")
innovation_df = pd.read_excel(xls, sheet_name="Innovation")
knowledge_df = pd.read_excel(xls, sheet_name="Knowledge")

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
    openai.api_key = "your_api_key"  # Replace with actual API Key
    
    prompt = f"""
    You are an HR expert in job evaluation using the IPE methodology.
    Analyze the following job description:
    """
    prompt += job_desc + """
    Based on IPE principles (Impact, Communication, Innovation, Knowledge), provide:
    - IPE Level recommendation (1-5)
    - Justification based on complexity, scope, and influence
    - Suggested Job Family & Stream (using provided Job Architecture framework)
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an AI specialized in Job Evaluation."},
                  {"role": "user", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    main()
