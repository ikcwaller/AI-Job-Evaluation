import streamlit as st
import pandas as pd
import requests
import os
import json

###############################
# Streamlit Config & Setup   #
###############################
# Title and layout
st.set_page_config(page_title="Job Description Generator & IPE Evaluator", layout="wide")
# Version caption
VERSION = "v1.4 – Updated August 2025"

###############################
# Google Sheets Configuration#
###############################
# IDs and base URLs for numeric and definition tables
SHEET_ID_NUMERIC      = "1zziZhOUA9Bv3HZSROUdgqA81vWJQFUB4rL1zRohb0Wc"
SHEET_URL_NUMERIC     = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID_NUMERIC}"
    "/gviz/tq?tqx=out:csv&sheet="
)
SHEET_ID_DEFINITIONS  = "1ZGJz_F7iDvFXCE_bpdNRpHU7r8Pd3YMAqPRzfxpdlQs"
SHEET_URL_DEFINITIONS = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID_DEFINITIONS}"
    "/gviz/tq?tqx=out:csv&sheet="
)
# Sheet name mappings
SHEET_NAMES_NUMERIC = {
    "impact_contribution_table":  "impact_contribution_table",
    "impact_size_table":          "impact_size_table",
    "communication_table":        "communication_table",
    "innovation_table":           "innovation_table",
    "knowledge_table":            "knowledge_table",
}
SHEET_NAMES_DEFINITIONS = {
    "impact_definitions":        "Impact_definitions",
    "communication_definitions": "Communication_definitions",
    "innovation_definitions":    "Innovation_definitions",
    "knowledge_definitions":     "Knowledge_definitions",
}

###############################
# Data Fetch Functions       #
###############################
@st.cache_data(show_spinner=False)
def fetch_numeric_table(key):
    """
    Load numeric lookup table from Google Sheets and return as dict.
    """
    name = SHEET_NAMES_NUMERIC.get(key, "")
    if not name:
        return {}
    url = SHEET_URL_NUMERIC + name
    df = pd.read_csv(url, index_col=0, dtype=str)
    # convert labels to floats
    df.columns = [float(c) for c in df.columns]
    df.index   = [float(i) for i in df.index]
    # convert values to numeric
    return df.apply(pd.to_numeric, errors="coerce").to_dict()

@st.cache_data(show_spinner=False)
def fetch_text_table(key):
    """
    Load definition table from Google Sheets and return as dict.
    """
    name = SHEET_NAMES_DEFINITIONS.get(key, "")
    if not name:
        return {}
    url = SHEET_URL_DEFINITIONS + name
    df = pd.read_csv(url, index_col=0, dtype=str)
    # convert labels to floats if possible
    try: df.columns = [float(c) for c in df.columns]
    except: pass
    try: df.index = [float(i) for i in df.index]
    except: pass
    return df.to_dict()

# Load all tables on startup
impact_contribution_table  = fetch_numeric_table("impact_contribution_table")
impact_size_table          = fetch_numeric_table("impact_size_table")
communication_table        = fetch_numeric_table("communication_table")
innovation_table           = fetch_numeric_table("innovation_table")
knowledge_table            = fetch_numeric_table("knowledge_table")
impact_definitions_table        = fetch_text_table("impact_definitions")
communication_definitions_table = fetch_text_table("communication_definitions")
innovation_definitions_table    = fetch_text_table("innovation_definitions")
knowledge_definitions_table     = fetch_text_table("knowledge_definitions")

###############################
# Gemini API Call            #
###############################
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/" \
    "v1/models/gemini-2.0-flash:generateContent"
)

def query_gemini_api(prompt: str) -> str:
    """
    Send prompt to Gemini API and return the text response.
    """
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    resp = requests.post(
        GEMINI_API_URL,
        headers=headers,
        params={"key": api_key},
        json=payload
    )
    if resp.status_code == 200:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    return ""

###############################
# Utility Functions          #
###############################

def clamp(val, lo, hi):
    """Clamp numeric val between lo and hi."""
    try:
        x = float(val)
    except:
        return lo
    return max(lo, min(hi, x))

def lookup_table(a, b, table):
    """Lookup table[a][b] or return 0 if missing."""
    try:
        return table[float(a)][float(b)]
    except:
        return 0

def calculate_ipe_score(total, imp, comm, innov, teams):
    """
    Final IPE = int((total - 26)/25 + 40) if total>26 and all dims valid.
    """
    if total > 26 and all(x != "" for x in [imp, comm, innov, teams]):
        return int((total - 26) / 25 + 40)
    return ""

def map_job_level(score):
    """Map IPE score to job architecture level."""
    if isinstance(score, int):
        ranges = [
            (40,41,1),(42,43,2),(44,45,3),(46,47,4),
            (48,50,5),(51,52,6),(53,55,7),(56,57,8),
            (58,59,9),(60,61,10),(62,65,11),(66,73,12)
        ]
        for lo, hi, lvl in ranges:
            if lo <= score <= hi:
                return lvl
    return "N/A"

# Seniority→IPE mapping used in generation prompt
SENIORITY_IPE_MAP = {
    "Junior":                  (41, 47),
    "Experienced/Supervisor":  (48, 52),
    "Senior/Manager":          (53, 55),
    "Expert/Sr Manager":       (56, 57),
    "Renowned Expert/Director":(58, 61),
    "Executive":               (60, 73),
}

# Breadth conversion tables
BREADTH_VALUE_MAP = {
    "Domestic role": 1.0,
    "Regional role": 2.0,
    "Global role":   3.0,
}
BREADTH_POINTS = {
    1.0:  0,
    1.5:  5,
    2.0: 10,
    2.5: 15,
    3.0: 20,
}

###############################
# Prompt Builders             #
###############################

def build_definitions_prompt() -> str:
    """
    Convert definition tables into markdown text for the LLM.
    """
    lines = ["**IMPACT DEFINITIONS (Impact x Contribution)**"]
    for i, row in impact_definitions_table.items():
        for c, txt in row.items():
            if txt:
                lines.append(f"Impact={i}, Contribution={c} => {txt}")
    lines.append("\n**COMMUNICATION DEFINITIONS (Communication x Frame)**")
    for i, row in communication_definitions_table.items():
        for f, txt in row.items():
            if txt:
                lines.append(f"Communication={i}, Frame={f} => {txt}")
    lines.append("\n**INNOVATION DEFINITIONS (Innovation x Complexity)**")
    for i, row in innovation_definitions_table.items():
        for comp, txt in row.items():
            if txt:
                lines.append(f"Innovation={i}, Complexity={comp} => {txt}")
    lines.append("\n**KNOWLEDGE DEFINITIONS (single dimension)**")
    for k, row in knowledge_definitions_table.items():
        for _, txt in row.items():
            if txt:
                lines.append(f"Knowledge={k} => {txt}")
    return "\n".join(lines)


def build_generation_prompt(
    title, purpose, geo, report, people, fin, decision,
    stake, delivs, background, seniority
) -> str:
    """
    Create job description prompt including formatting rules,
    seniority tone and recommended IPE range.
    """
    min_ipe, max_ipe = SENIORITY_IPE_MAP.get(seniority, (None, None))
    fmt = (
        "## Formatting Rules\n"
        "1. Objectives: plain paragraph, 3–6 sentences.\n"
        "2. Summary of Responsibilities: 5–8 bullets.\n"
        "3. Scope of Decision Making: plain paragraph.\n"
        "4. Experience and Qualifications: bullets.\n"
        "5. Skills and Capabilities: bullets.\n"
    )
    tone = f"Use language appropriate for a {seniority} level role—adjust verbs accordingly."
    range_note = (
        f"Recommended IPE score range for this seniority: {min_ipe}–{max_ipe}."
        if min_ipe else ""
    )
    # format deliverables into bullets
    ds = "\n".join(f"- {d.strip()}" for d in delivs.splitlines() if d.strip())

    return f"""{fmt}

{tone}
{range_note}

You are an HR expert. Write a job description (omit Position Information):

• Job Title: {title}
• Purpose of the Role: {purpose}
• Geographic Scope: {geo}
• Reports To: {report}
• People Responsibility: {people}
• Financial Responsibility: {fin}
• Decision-Making Authority: {decision}
• Main Stakeholders: {stake}
• Top Deliverables:
{ds}
• Required Background: {background}

Return in this exact structure:

---
Objectives
<…>

Summary of Responsibilities
- …

Scope of Decision Making
<…>

Experience and Qualifications
- …

Skills and Capabilities
- …
---"""

###############################
# Evaluate Job               #
###############################

def evaluate_job(job_desc, size, teams, breadth_str):
    """
    Evaluate IPE by sending job_desc to LLM, parsing JSON,
    then performing numeric lookups and breadth addition.
    """
    defs_text = build_definitions_prompt()
    prompt = f"""
You are an HR expert specializing in IPE job evaluation.

Below are official definitions for each integer rating; half steps allowed except Impact.

{defs_text}

--- JOB DESCRIPTION ---
{job_desc}

Return ONLY JSON with fields 
"impact","contribution","communication",
"frame","innovation","complexity","knowledge"
as {{ "value":X, "justification":"..." }}
"""
    ai_out = query_gemini_api(prompt)
    # extract JSON block
    raw = ai_out.strip()
    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1:
        return None, f"LLM returned non-JSON output:\n{ai_out}"
    blob = raw[start:end+1]
    try:
        data = json.loads(blob)
    except Exception as e:
        return None, f"Could not parse JSON: {e}\nRaw:\n{ai_out}"

    # clamp & round values
    vals, justs = {}, {}
    dims = [
        ("impact",1,5),("contribution",1,5),("communication",1,5),
        ("frame",1,4),("innovation",1,6),("complexity",1,4),("knowledge",1,8)
    ]
    for key, lo, hi in dims:
        ent = data.get(key, {})
        if isinstance(ent, dict):
            raw_v = ent.get("value", 0)
            jtxt  = ent.get("justification", "No justification provided.")
        else:
            raw_v = ent
            jtxt  = "No justification provided."
        if key == "impact":
            vals[key] = int(round(clamp(raw_v, lo, hi)))
        else:
            vals[key] = clamp(raw_v, lo, hi)
        justs[key] = jtxt

    # numeric lookups
    inter_imp  = lookup_table(vals["contribution"], vals["impact"], impact_contribution_table)
    final_imp  = lookup_table(size, inter_imp, impact_size_table)
    comm_s     = lookup_table(vals["frame"], vals["communication"], communication_table)
    innov_s    = lookup_table(vals["complexity"], vals["innovation"], innovation_table)
    base_kn    = lookup_table(teams, vals["knowledge"], knowledge_table)
    # breadth
    bval       = BREADTH_VALUE_MAP.get(breadth_str, 1.0)
    bpts       = BREADTH_POINTS.get(bval, 0)
    know_s     = base_kn + bpts
    # total & final
    total_pts  = final_imp + comm_s + innov_s + know_s
    ipe_score  = calculate_ipe_score(total_pts, final_imp, comm_s, innov_s, teams)
    job_level  = map_job_level(ipe_score)

    # build markdown sections
    raw_md = ["### AI Raw Scores (Impact forced integer)"]
    for k in ["impact","contribution","communication","frame","innovation","complexity","knowledge"]:
        raw_md.append(f"#### {k.capitalize()}")
        raw_md.append(f"- Score: {vals[k]}")
        raw_md.append(f"- Justification: {justs[k]}")
        raw_md.append("")
    table_md = [
        "### Numeric Table Lookup Results",
        f"- Impact (intermediate): {inter_imp}, final: {final_imp}",
        f"- Communication: {comm_s}",
        f"- Innovation: {innov_s}",
        f"- Knowledge: {know_s}"
    ]
    calc_md = [
        "### Final Calculation",
        f"Total Points: {total_pts}",
        f"IPE Score: {ipe_score}",
        f"Job Level: {job_level}"
    ]
    details = "\n\n".join([
        "\n".join(raw_md),
        "\n".join(table_md),
        "\n".join(calc_md)
    ])
    return ipe_score, details

###############################
# Main Streamlit UI          #
###############################
def main():
    st.title("📋 Job Description Generator & IPE Evaluator")
    mode = st.radio("Mode:", ["Create & Evaluate", "Evaluate Existing"])

    if mode == "Create & Evaluate":
        st.markdown("**Step 1: Enter Role Details**")
        jt = st.text_input("Job Title:")
        pu = st.text_area("Purpose of the Role:")
        gs = st.selectbox("Geographic Scope:", ["Global","Multi-country","National","Regional"])
        rp = st.text_input("Reports To:")
        pr = st.text_input("People Responsibility:")
        fr = st.text_area("Financial Responsibility (budget/P&L):")
        de = st.text_area("Decision-Making Authority:")
        stak = st.text_area("Main Stakeholders:")
        td = st.text_area("Top Deliverables (one per line):")
        bg = st.text_area("Required Background / Qualifications:")
        sr = st.selectbox("Estimated Seniority Level:", [
            "Junior","Experienced/Supervisor","Senior/Manager",
            "Expert/Sr Manager","Renowned Expert/Director","Executive"
        ])

        # generate job desc
        if "jd" not in st.session_state:
            st.session_state.jd = ""
        if st.button("Generate Job Description"):
            missing = [lbl for lbl,val in [
                ("Job Title",jt),("Purpose",pu),("Reports To",rp),
                ("Top Deliverables",td),("Background",bg)
            ] if not val.strip()]
            if missing:
                st.error(f"Please fill in: {', '.join(missing)}")
            else:
                prompt = build_generation_prompt(jt,pu,gs,rp,pr,fr,de,stak,td,bg,sr)
                with st.spinner("Generating job description..."):
                    st.session_state.jd = query_gemini_api(prompt)

        # display and evaluate
        if st.session_state.jd:
            st.subheader("🔧 Generated Job Description (No Position Info)")
            st.text_area("Job Description", st.session_state.jd, height=300)
            st.download_button("Download as Text", st.session_state.jd,
                               file_name="job_description.txt")
            st.markdown("---")
            st.markdown("**Step 2: Evaluate IPE**")
            sz = st.slider("Size Score (1–20)", 1.0,20.0,10.0,step=0.5)
            tm_str = st.selectbox("Team Responsibility:", [
                "1 - Individual Contributor","2 - Manager over Employees","3 - Manager over Managers"
            ])
            tm = float(tm_str[0])
            br = st.selectbox("Breadth of Role:", list(BREADTH_VALUE_MAP.keys()))
            if st.button("Evaluate IPE"):
                with st.spinner("Evaluating IPE level..."):
                    score, details = evaluate_job(st.session_state.jd, sz, tm, br)
                if score is None:
                    st.error(details)
                else:
                    st.subheader(f"🏆 IPE Evaluation Result (Score: {score})")
                    st.markdown(details)
        st.caption(VERSION)
    else:
        st.header("🔍 Evaluate an Existing Job Description")
        ex  = st.text_area("Paste Job Description Here:", height=300)
        sz_ex = st.slider("Size Score (1–20)",1.0,20.0,10.0,step=0.5,key="sz_ex")
        tm_str_ex = st.selectbox("Team Responsibility:", [
            "1 - Individual Contributor","2 - Manager over Employees","3 - Manager over Managers"
        ], key="tm_ex")
        tm_ex = float(tm_str_ex[0])
        br_ex = st.selectbox("Breadth of Role:", list(BREADTH_VALUE_MAP.keys()), key="br_ex")
        if st.button("Evaluate IPE", key="eval_existing"):
            if not ex.strip(): st.error("Please paste a job description first.")
            else:
                with st.spinner("Evaluating IPE level..."):
                    score, details = evaluate_job(ex, sz_ex, tm_ex, br_ex)
                if score is None:
                    st.error(details)
                else:
                    st.subheader(f"🏆 IPE Evaluation Result (Score: {score})")
                    st.markdown(details)
        st.caption(VERSION)

if __name__ == "__main__":
    main()
