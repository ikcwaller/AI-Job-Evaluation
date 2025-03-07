import streamlit as st
import pandas as pd
import requests
import os
import json

###############################
# 1) Streamlit Config & Setup #
###############################

st.set_page_config(page_title="Job Evaluation AI", layout="wide")

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"

SHEET_ID = "1zziZhOUA9Bv3HZSROUdgqA81vWJQFUB4rL1zRohb0Wc"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet="

SHEET_NAMES = {
    "impact_contribution_table": "impact_contribution_table",
    "impact_size_table": "impact_size_table",
    "communication_table": "communication_table",
    "innovation_table": "innovation_table",
    "knowledge_table": "knowledge_table",
    # We won't use breadth_table because we map breadth directly
}

############################
# 2) Fetch Tables Function #
############################
def fetch_table(sheet_key):
    try:
        sheet_name = SHEET_NAMES.get(sheet_key, "")
        if not sheet_name:
            return None
        url = SHEET_URL + sheet_name

        # CSV from Google Sheets is usually comma-delimited with double quotes
        df = pd.read_csv(
            url,
            index_col=0,
            dtype=str,
            sep=",",
            quotechar='"',
            header=0
        )

        # Convert column/index from strings to float
        df.columns = [float(str(c)) for c in df.columns]
        df.index   = [float(str(i)) for i in df.index]

        df = df.apply(pd.to_numeric, errors='coerce')
        return df.to_dict()
    except Exception as e:
        st.error(f"Error fetching {sheet_key}: {e}")
        return None

# Load tables
impact_contribution_table = fetch_table("impact_contribution_table")
impact_size_table = fetch_table("impact_size_table")
communication_table = fetch_table("communication_table")
innovation_table = fetch_table("innovation_table")
knowledge_table = fetch_table("knowledge_table")
# Not loading "breadth_table" because we map breadth directly.

###############################
# 3) Gemini API Query Function #
###############################
def query_gemini_api(prompt):
    headers = {"Content-Type": "application/json"}
    params = {"key": GOOGLE_GEMINI_API_KEY}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            return f"API Error! Details: {resp.text}"
    except Exception as e:
        return f"Error calling Gemini API: {e}"

###############################
# 4) Clamping & Lookups       #
###############################
def clamp(value, min_val, max_val):
    try:
        f = float(value)
    except:
        f = min_val
    if f < min_val:
        f = min_val
    if f > max_val:
        f = max_val
    return f

def lookup_table(value1, value2, table, label="Lookup"):
    if not isinstance(table, dict):
        return None
    try:
        val1 = float(value1)
        val2 = float(value2)
        if val1 in table and isinstance(table[val1], dict):
            result = table[val1].get(val2, None)
            if result is None:
                st.warning(f"{label} failed: No match for ({val1}, {val2})")
            return result
        return None
    except (ValueError, TypeError):
        return None

###############################
# 5) IPE Score & Job Level    #
###############################
def calculate_ipe_score(total_points, impact, communication, innovation, teams):
    if total_points > 26 and all(x not in (None, "") for x in [impact, communication, innovation, teams]):
        return int((total_points - 26)/25 + 40)
    return ""

def map_job_level(ipe_score):
    if isinstance(ipe_score, int):
        if 40 <= ipe_score <= 41:
            return 1
        elif 42 <= ipe_score <= 43:
            return 2
        elif 44 <= ipe_score <= 45:
            return 3
        elif 46 <= ipe_score <= 47:
            return 4
        elif 48 <= ipe_score <= 50:
            return 5
        elif 51 <= ipe_score <= 52:
            return 6
        elif 53 <= ipe_score <= 55:
            return 7
        elif 56 <= ipe_score <= 57:
            return 8
        elif 58 <= ipe_score <= 59:
            return 9
        elif 60 <= ipe_score <= 61:
            return 10
        elif 62 <= ipe_score <= 65:
            return 11
        elif 66 <= ipe_score <= 73:
            return 12
        else:
            return "N/A"
    return ""

###############################
# 6) Breadth map for 1->3     #
###############################
BREADTH_POINTS = {
    1.0: 0,
    1.5: 5,
    2.0: 10,
    2.5: 15,
    3.0: 20
}

###############################
# 7) Evaluate Job             #
###############################
def evaluate_job(job_desc, size, teams, breadth):
    prompt = f"""
    You are an HR expert specializing in IPE job evaluation.
    Please be conservative in your ratings. 
    Only assign high values (e.g., 4 or 5) for a sub-area if the job description strongly indicates 
    senior-level responsibilities and significant scope of influence. 
    Otherwise, err on the side of assigning mid- or low-range values.
    Evaluate the job description below and return JSON with numeric values in these ranges:
    impact: 1-5,
    contribution: 1-5,
    communication: 1-5,
    frame: 1-4,
    innovation: 1-6,
    complexity: 1-4,
    knowledge: 1-8.
    Half steps allowed. Nothing outside those ranges.
    Return ONLY JSON, no extra text:
    {{
      "impact": {{"value": X.X, "justification": "text"}},
      "contribution": {{"value": X.X, "justification": "text"}},
      "communication": {{"value": X.X, "justification": "text"}},
      "frame": {{"value": X.X, "justification": "text"}},
      "innovation": {{"value\": X.X, \"justification\": \"text\"}},
      "complexity": {{"value\": X.X, \"justification\": \"text\"}},
      "knowledge": {{"value\": X.X, \"justification\": \"text\"}}
    }}

    Job Description:
    {job_desc}
    """

    ai_response = query_gemini_api(prompt)
    if not ai_response or "API Error!" in ai_response:
        return "API Error", ai_response

    try:
        parsed = ai_response.strip().replace("```json", "").replace("```", "")
        data = json.loads(parsed)
    except Exception as e:
        return "Error", f"Could not parse AI JSON: {e}. Response was: {ai_response}"

    def get_clamped(dct, key, minv, maxv):
        val = dct.get(key, {}).get("value", 0)
        return clamp(val, minv, maxv)

    # 1) AI sub-area values
    impact_val   = get_clamped(data, "impact",        1, 5)
    contrib_val  = get_clamped(data, "contribution",  1, 5)
    comm_val     = get_clamped(data, "communication", 1, 5)
    frame_val    = get_clamped(data, "frame",         1, 4)
    innov_val    = get_clamped(data, "innovation",    1, 6)
    comp_val     = get_clamped(data, "complexity",    1, 4)
    know_val     = get_clamped(data, "knowledge",     1, 8)

    # 2) Impact => 2-step
    inter_impact = lookup_table(contrib_val, impact_val, impact_contribution_table, label="Impact-Contrib")
    final_impact = lookup_table(size, inter_impact, impact_size_table, label="Impact-Size")

    # 3) Communication => (communication, frame)
    comm_score   = lookup_table(frame_val, comm_val, communication_table, label="Communication")

    # 4) Innovation => (innovation, complexity)
    innov_score  = lookup_table(comp_val, innov_val, innovation_table, label="Innovation")

    # 5) Knowledge => knowledge_table(knowledge, teams)
    knowledgePoints = lookup_table(teams, know_val, knowledge_table, label="Knowledge Table")
    if knowledgePoints is None:
        knowledgePoints = 0

    # Then add user-input breadth => BREADTH_POINTS
    # 'breadth' is already a float from our dictionary, so no need to parse
    breadth_score = BREADTH_POINTS.get(breadth, 0)
    final_know = knowledgePoints + breadth_score

    # 6) Sum sub-areas
    fi = 0 if final_impact  is None else final_impact
    cs = 0 if comm_score    is None else comm_score
    iscore = 0 if innov_score is None else innov_score
    fk = final_know

    total_points = fi + cs + iscore + fk

    # 7) IPE + job level
    ipe_score = calculate_ipe_score(total_points, fi, cs, iscore, teams)
    job_level = map_job_level(ipe_score)

    # 8) Justifications
    justifications = {}
    for sub_area in ["impact","contribution","communication","frame","innovation","complexity","knowledge"]:
        sub = data.get(sub_area, {})
        justifications[sub_area] = sub.get("justification", "")

    # 9) Construct multi-section details

    details_sections = []

    # (A) AI Raw Scores
    raw_scores_md = ["### AI Raw Scores"]
    for sub_area in ["impact","contribution","communication","frame","innovation","complexity","knowledge"]:
        raw_val = data.get(sub_area, {}).get("value", "?")
        raw_j  = justifications[sub_area]
        raw_scores_md.append(f"**{sub_area.capitalize()}**: {raw_val}\nJustification: {raw_j}\n")
    details_sections.append("\n".join(raw_scores_md))

    # (B) Table Lookup Results
    table_md = ["### Table Lookup Results"]
    table_md.append(f"- Impact (intermediate): {inter_impact}, final: {fi}")
    table_md.append(f"- Communication: {cs}")
    table_md.append(f"- Innovation: {iscore}")
    table_md.append(f"- Knowledge (Points from knowledge_table): {knowledgePoints}")
    table_md.append(f"- Breadth Points: {breadth_score}")
    table_md.append(f"- Final Knowledge: {fk}")
    details_sections.append("\n".join(table_md))

    # (C) Final Calculation
    calc_md = ["### Final Calculation"]
    calc_md.append(f"Total Points: {total_points}")
    calc_md.append(f"IPE Score: {ipe_score}")
    calc_md.append(f"Job Level: {job_level}")
    details_sections.append("\n".join(calc_md))

    details = "\n\n".join(details_sections)

    return ipe_score, details

###############################
# 8) Main Streamlit UI
###############################
def main():
    st.title("📊 AI-Powered Job Evaluation Chatbot")
    st.write("Upload a job description and get an IPE Level & Job Architecture suggestion.")

    job_description = st.text_area("📌 Paste Job Description Here:")
    # Size can be half-steps
    size = st.slider("📏 Enter Size Score (1-20, half steps)", 1.0, 20.0, step=0.5)

    # We'll keep these textual descriptions for teams
    teams = st.radio(
        "👥 Select Team Responsibility",
        [
            "1 - Individual Contributor",
            "2 - Manager over Employees",
            "3 - Manager over Managers"
        ]
    )
    
    # Make a dictionary for Breadth textual -> numeric
    breadth_options = {
        "1 - Domestic Role": 1.0,
        # "1.5 - Some Half Step": 1.5,
        "2 - Regional Role": 2.0,
        # "2.5 - Some Half Step": 2.5,
        "3 - Global Role": 3.0
    }

    breadth_choice = st.radio(
        "🌍 Select Breadth of Role",
        list(breadth_options.keys())  # show textual options
    )

    if st.button("🚀 Evaluate Job"):
        if job_description:
            tval = float(teams[0])  # parse e.g. '1' => 1.0
            bval = breadth_options[breadth_choice]  # e.g. "3 - Global Role" => 3.0
            ipe_score, details = evaluate_job(job_description, size, tval, bval)
            st.subheader("🏆 Job Evaluation Result:")
            st.markdown(details)
        else:
            st.warning("Warning: Please enter a job description.")

if __name__ == "__main__":
    main()
