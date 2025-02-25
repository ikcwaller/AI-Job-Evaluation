import openai
import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# GitHub Raw URL for Excel file (Replace 'your-username' and 'your-repo-name')
github_excel_url = "https://raw.githubusercontent.com/ikcwaller/AI-Job-Evaluation/main/IPE_Calculator_HFG_HH.xlsx"

# Fetch the file from GitHub and verify it loads correctly
response = requests.get(github_excel_url)

if response.status_code == 200:
    file_content = BytesIO(response.content)
    
    try:
        xls = pd.ExcelFile(file_content)
        print("✅ File downloaded successfully and loaded into pandas.")
    except Exception as e:
        raise FileNotFoundError(f"⚠️ Error loading the Excel file. Possible issues:\n"
                                f"1. File format is incorrect (Try using .xlsx instead of .xlsm).\n"
                                f"2. File is corrupted.\n"
                                f"3. Pandas cannot read the file.\n"
                                f"Error Details: {str(e)}")
else:
    raise FileNotFoundError(f"🚨 Error: Could not download the Excel file.\n"
                            f"❌ Check your GitHub URL: {github_excel_url}\n"
                            f"❌ Status Code: {response.status_code}")

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
    client = openai.OpenAI()  # Ensure correct OpenAI client initialization
    
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
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI specialized in Job Evaluation."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    main()
