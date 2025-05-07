import streamlit as st
import pandas as pd
import requests
import os
import json

###############################
# 1) Streamlit Config & Setup #
###############################

st.set_page_config(page_title="Job Evaluation AI", layout="wide")

# -- Numeric tables in one doc:
SHEET_ID_NUMERIC = "1zziZhOUA9Bv3HZSROUdgqA81vWJQFUB4rL1zRohb0Wc"
SHEET_URL_NUMERIC = f"https://docs.google.com/spreadsheets/d/{SHEET_ID_NUMERIC}/gviz/tq?tqx=out:csv&sheet="

# -- Definitions in another doc:
SHEET_ID_DEFINITIONS = "1ZGJz_F7iDvFXCE_bpdNRpHU7r8Pd3YMAqPRzfxpdlQs"
SHEET_URL_DEFINITIONS = f"https://docs.google.com/spreadsheets/d/{SHEET_ID_DEFINITIONS}/gviz/tq?tqx=out:csv&sheet="

# --- For numeric tables
SHEET_NAMES_NUMERIC = {
    "impact_contribution_table": "impact_contribution_table",
    "impact_size_table":         "impact_size_table",
    "communication_table":       "communication_table",
    "innovation_table":          "innovation_table",
    "knowledge_table":           "knowledge_table",
}

# --- For definition tables
SHEET_NAMES_DEFINITIONS = {
    "impact_definitions":        "Impact_definitions",
    "communication_definitions": "Communication_definitions",
    "innovation_definitions":    "Innovation_definitions",
    "knowledge_definitions":     "Knowledge_definitions",
}

############################
# 2) Fetch Tables Functions #
############################

def fetch_numeric_table(sheet_key):
    """
    For the numeric lookup tables (impact_contribution_table, etc.).
    """
    try:
        sheet_name = SHEET_NAMES_NUMERIC.get(sheet_key, "")
        if not sheet_name:
            return None
        url = SHEET_URL_NUMERIC + sheet_name

        df = pd.read_csv(
            url,
            index_col=0,
            dtype=str,
            sep=",",
            quotechar='"',
            header=0
        )
        # Convert row/column labels to float
        df.columns = [float(str(c)) for c in df.columns]
        df.index   = [float(str(i)) for i in df.index]

        # Convert the entire DataFrame to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        return df.to_dict()
    except Exception as e:
        st.error(f"Error fetching numeric table '{sheet_key}': {e}")
        return None

def fetch_text_table(sheet_key):
    """
    For the definition sheets, storing text intersections.
    (No numeric conversion of the cell contents.)
    """
    try:
        sheet_name = SHEET_NAMES_DEFINITIONS.get(sheet_key, "")
        if not sheet_name:
            return None
        url = SHEET_URL_DEFINITIONS + sheet_name

        df = pd.read_csv(
            url,
            index_col=0,
            sep=",",
            quotechar='"',
            header=0,
            dtype=str  # keep cell contents as text
        )
        # Convert row/column labels to float if possible
        try:
            df.columns = [float(str(c)) for c in df.columns]
        except:
            pass
        try:
            df.index = [float(str(i)) for i in df.index]
        except:
            pass

        return df.to_dict()
    except Exception as e:
        st.error(f"Error fetching definition table '{sheet_key}': {e}")
        return None

#######################
# 3) Load Data Sources #
#######################
# Numeric tables
impact_contribution_table = fetch_numeric_table("impact_contribution_table")
impact_size_table         = fetch_numeric_table("impact_size_table")
communication_table       = fetch_numeric_table("communication_table")
innovation_table          = fetch_numeric_table("innovation_table")
knowledge_table           = fetch_numeric_table("knowledge_table")

# Definition tables
impact_definitions_table        = fetch_text_table("impact_definitions")
communication_definitions_table = fetch_text_table("communication_definitions")
innovation_definitions_table    = fetch_text_table("innovation_definitions")
knowledge_definitions_table     = fetch_text_table("knowledge_definitions")

###############################
# 4) Gemini API Query Function #
###############################
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
def query_gemini_api(prompt):
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    params = {"key": api_key}
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
# 5) Utility: clamp & lookup  #
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
# 6) IPE Score & Job Level    #
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
# 7) Breadth map for 1->3     #
###############################
BREADTH_POINTS = {
    1.0: 0,
    1.5: 5,
    2.0: 10,
    2.5: 15,
    3.0: 20
}

###################################
# 8) Build Definitions Text Block  #
###################################
def build_definitions_prompt():
    """
    Turn your definition tables into text so the LLM can see how each integer rating is officially described.
    """
    lines = []

    # Impact (row=impact, col=contribution)
    lines.append("**IMPACT DEFINITIONS (by Impact x Contribution)**")
    if impact_definitions_table:
        for imp_val, row_dict in impact_definitions_table.items():
            for contrib_val, text in row_dict.items():
                if text:
                    lines.append(f"Impact={imp_val}, Contribution={contrib_val} => {text}")

    # Communication (row=communication, col=frame)
    lines.append("\n**COMMUNICATION DEFINITIONS (by Communication x Frame)**")
    if communication_definitions_table:
        for comm_val, row_dict in communication_definitions_table.items():
            for frame_val, text in row_dict.items():
                if text:
                    lines.append(f"Communication={comm_val}, Frame={frame_val} => {text}")

    # Innovation (row=innovation, col=complexity)
    lines.append("\n**INNOVATION DEFINITIONS (by Innovation x Complexity)**")
    if innovation_definitions_table:
        for innov_val, row_dict in innovation_definitions_table.items():
            for comp_val, text in row_dict.items():
                if text:
                    lines.append(f"Innovation={innov_val}, Complexity={comp_val} => {text}")

    # Knowledge (single dimension)
    lines.append("\n**KNOWLEDGE DEFINITIONS (single dimension)**")
    if knowledge_definitions_table:
        for know_val, row_dict in knowledge_definitions_table.items():
            if isinstance(row_dict, dict):
                for col_key, text in row_dict.items():
                    if text:
                        lines.append(f"Knowledge={know_val} => {text}")

    return "\n".join(lines)

###############################
# 9) Evaluate Job             #
###############################
def evaluate_job(job_desc, size, teams, breadth):
    # Step A: Build the big chunk of text with definitions
    definitions_text = build_definitions_prompt()

    # Step B: Build the prompt that instructs the LLM to use those definitions
    #         and clarifies half-step usage for non-Impact areas.
    prompt = f"""
    You are an HR expert specializing in IPE job evaluation.

    Below are official definitions for each integer rating. 
    For Impact, you must pick an integer (1-5). 
    For all other areas (contribution, communication, frame, innovation, complexity, knowledge),
    you can pick half steps (like 2.5) if the job partially matches integer N and partially matches N+1.

    (Example: If the job is too advanced for 'Frame=2' but not quite as advanced as 'Frame=3', you can pick Frame=2.5.)

    Use these definitions to guide your choices, but if none fits perfectly, you may use .5 for any dimension 
    except Impact. 
    Impact must stay integer.

    {definitions_text}

    --- JOB DESCRIPTION ---
    {job_desc}

    Please return ONLY JSON with numeric values:
      "impact":       1-5 (integer only),
      "contribution": 1-5 (half steps allowed),
      "communication":1-5 (half steps allowed),
      "frame":        1-4 (half steps allowed),
      "innovation":   1-6 (half steps allowed),
      "complexity":   1-4 (half steps allowed),
      "knowledge":    1-8 (half steps allowed).

    Return them in JSON format:
    {{
      "impact": {{"value": X.X, "justification": "text"}},
      "contribution": {{"value": X.X, "justification": "text"}},
      "communication": {{"value": X.X, "justification": "text"}},
      "frame": {{"value": X.X, "justification": "text"}},
      "innovation": {{"value\": X.X, \"justification\": \"text\"}},
      "complexity": {{"value\": X.X, \"justification\": \"text\"}},
      "knowledge": {{"value\": X.X, \"justification\": \"text\"}}
    }}

    IMPORTANT:
    - Return ONLY JSON, no extra text or explanation.
    - If uncertain or in-between two integers for non-Impact areas, pick a .5 rating.
    - Impact must be integer only.
    """

    # Step C: Call Gemini with the new prompt
    ai_response = query_gemini_api(prompt)
    if not ai_response or "API Error!" in ai_response:
        return "API Error", ai_response

    # Step D: Parse AI JSON
    try:
        parsed = ai_response.strip().replace("```json", "").replace("```", "")
        data = json.loads(parsed)
    except Exception as e:
        return "Error", f"Could not parse AI JSON: {e}. Response was: {ai_response}"

    # 1) AI sub-area values, with clamping. Force Impact to integer.
    #    We'll do Impact first, then overwrite the dictionary so it's shown as integer
    raw_impact_val = data.get("impact", {}).get("value", 0)
    impact_val = clamp(raw_impact_val, 1, 5)
    impact_val = round(impact_val)  # force integer
    data["impact"]["value"] = float(impact_val)

    # The rest can have half-steps, so just clamp
    contrib_val = clamp(data.get("contribution", {}).get("value", 0), 1, 5)
    comm_val    = clamp(data.get("communication", {}).get("value", 0), 1, 5)
    frame_val   = clamp(data.get("frame", {}).get("value", 0), 1, 4)
    innov_val   = clamp(data.get("innovation", {}).get("value", 0), 1, 6)
    comp_val    = clamp(data.get("complexity", {}).get("value", 0), 1, 4)
    know_val    = clamp(data.get("knowledge", {}).get("value", 0), 1, 8)

    # 2) Numeric lookups for final scoring
    inter_impact = lookup_table(contrib_val, impact_val, impact_contribution_table, label="Impact-Contrib")
    final_impact = lookup_table(size, inter_impact, impact_size_table, label="Impact-Size")

    comm_score   = lookup_table(frame_val, comm_val, communication_table, label="Communication")
    innov_score  = lookup_table(comp_val, innov_val, innovation_table, label="Innovation")

    knowledgePoints = lookup_table(teams, know_val, knowledge_table, label="Knowledge Table")
    if knowledgePoints is None:
        knowledgePoints = 0
    breadth_score = BREADTH_POINTS.get(breadth, 0)
    final_know = knowledgePoints + breadth_score

    fi = 0 if final_impact  is None else final_impact
    cs = 0 if comm_score    is None else comm_score
    iscore = 0 if innov_score is None else innov_score
    fk = final_know

    total_points = fi + cs + iscore + fk
    ipe_score = calculate_ipe_score(total_points, fi, cs, iscore, teams)
    job_level = map_job_level(ipe_score)

    # 3) Build a final summary
    justifications = {}
    for sub_area in ["impact","contribution","communication","frame","innovation","complexity","knowledge"]:
        sub = data.get(sub_area, {})
        justifications[sub_area] = sub.get("justification", "")

    details_sections = []

    # (A) AI Raw Scores
    raw_scores_md = ["### AI Raw Scores (with Impact forced integer)"]
    for sub_area in ["impact","contribution","communication","frame","innovation","complexity","knowledge"]:
        raw_val = data.get(sub_area, {}).get("value", "?")
        raw_j  = justifications[sub_area]
        raw_scores_md.append(f"**{sub_area.capitalize()}**: {raw_val}\nJustification: {raw_j}\n")
    details_sections.append("\n".join(raw_scores_md))

    # (B) Numeric Table Results
    table_md = ["### Numeric Table Lookup Results"]
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
# 10) Main Streamlit UI
###############################
def main():
    st.title("📊 AI-Powered Job Evaluation Chatbot")
    st.write("Upload a job description and get an IPE Level & Job Architecture suggestion.")

    job_description = st.text_area("📌 Paste Job Description Here:")
    size = st.slider("📏 Enter Size Score (1-20, half steps)", 1.0, 20.0, step=0.5)

    teams = st.radio(
        "👥 Select Team Responsibility",
        [
            "1 - Individual Contributor",
            "2 - Manager over Employees",
            "3 - Manager over Managers"
        ]
    )

    breadth_options = {
        "1 - Domestic Role": 1.0,
        "2 - Regional Role": 2.0,
        "3 - Global Role": 3.0
    }
    breadth_choice = st.radio(
        "🌍 Select Breadth of Role",
        list(breadth_options.keys())
    )

    if st.button("🚀 Evaluate Job"):
        if job_description:
            tval = float(teams[0])  
            bval = breadth_options[breadth_choice]
            ipe_score, details = evaluate_job(job_description, size, tval, bval)
            st.subheader("🏆 Job Evaluation Result:")
            st.markdown(details)
        else:
            st.warning("Warning: Please enter a job description.")

if __name__ == "__main__":
    main()
