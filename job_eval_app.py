"""
IPE Job Description Evaluator - Complete System
================================================

Two modes:
1. Evaluate Only - Evaluate existing job descriptions
2. Create & Evaluate - Generate JD from questionnaire and evaluate

Built for EU Pay Transparency Directive compliance.
"""

import streamlit as st
import pandas as pd
import requests
import os
import json
import re
from typing import Dict, Tuple, Optional, List

###############################
# Configuration
###############################
st.set_page_config(
    page_title="IPE Evaluator - Complete System",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLAUDE_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

###############################
# Official Mercer IPE Framework
###############################

MERCER_IPE_FRAMEWORK = """
═══════════════════════════════════════════════════════════════════════════
OFFICIAL MERCER IPE FRAMEWORK
═══════════════════════════════════════════════════════════════════════════

IMPACT DIMENSION (How broadly does the role affect the organization?)

1. DELIVERY
   Area: Job Area | Type: Within specific standards and guidelines
   Deliver own output by following defined procedures/processes under close supervision.

2. OPERATION  
   Area: Job Area(s) | Type: Within operational targets or service standards
   Work to achieve objectives with short-term, operational focus and limited impact on others.

3. TACTICAL
   Area: Business Function | Type: New products/processes/standards based on organizational strategy
   Develop new products, processes, standards or operational plans in support of business strategies.
   KEY: Impact at FUNCTION level (16-20%), not just within own area.

4. STRATEGIC
   Area: Organization level | Type: Longer-term plans based on organization vision
   Directly influences development of corporate business unit or organization's business strategies.

5. VISIONARY
   Area: Corporate level | Type: Vision, mission & value
   Lead an organization within a corporation; freedom to define vision and direction.

───────────────────────────────────────────────────────────────────────────

CONTRIBUTION DIMENSION (How much does the role determine outcomes?)

1. LIMITED - Hard to identify/discern contribution
2. SOME - Easily discernible, usually leads indirectly to results
3. DIRECT - Directly and clearly influences course of action determining results
4. SIGNIFICANT - Quite marked contribution with frontline/primary authority
5. MAJOR - Predominant authority in determining key results

NOTE: Contribution assessed in CONTEXT of Impact level.

───────────────────────────────────────────────────────────────────────────

COMMUNICATION DIMENSION (What level of persuasion is required?)

1. CONVEY - Obtain and provide information
2. ADAPT AND EXCHANGE - Explain facts/policies through flexibility and compromise
3. INFLUENCE - Convince others where interest/skepticism exists; persuasion required
4. NEGOTIATE - Convince others to accept proposals where cooperation varies
5. NEGOTIATE LONG-TERM - Strategic agreements with differing viewpoints

───────────────────────────────────────────────────────────────────────────

FRAME DIMENSION (What is the communication context?)

1. INTERNAL SHARED - Common desire to reach solution within corporation
   Examples: Finance, Engineering, Logistics working toward shared goals

2. EXTERNAL SHARED - Common desire to reach solution outside corporation  
   Examples: Vendor partnerships, dealer relationships with aligned interests

3. INTERNAL DIVERGENT - Conflicting objectives within corporation
   Examples: HR negotiating between departments, cross-functional roles with competing priorities

4. EXTERNAL DIVERGENT - Conflicting objectives outside corporation
   Examples: Adversarial negotiations, legal disputes

───────────────────────────────────────────────────────────────────────────

INNOVATION DIMENSION (What degree of originality is required?)

1. FOLLOW - Follow set procedures in repeated tasks
2. CHECK - Check and correct problems not immediately evident
3. MODIFY - Identify problems and update/modify methods without defined procedures
4. IMPROVE - Significantly improve, change or adapt entire existing processes/systems
5. CREATE/CONCEPTUALIZE - Create truly new methods across job areas/functions
6. SCIENTIFIC/TECHNICAL BREAKTHROUGH - Major new advances across functions

───────────────────────────────────────────────────────────────────────────

COMPLEXITY DIMENSION (How complex are the problems?)

1. DEFINED - Single job area/discipline; well-defined scope
2. DIFFICULT - Vaguely defined; requires understanding other disciplines
3. COMPLEX - Broad solutions requiring TWO of three dimensions (Operational, Financial, Human)
4. MULTI-DIMENSIONAL - End-to-end solutions with ALL THREE dimensions

───────────────────────────────────────────────────────────────────────────

KNOWLEDGE DIMENSION (What expertise is required?)

1. LIMITED - Primary school; basic knowledge
2. BASIC - Specialized school; foundational expertise
3. BROAD - Specialized degree; working professional knowledge
4. EXPERTISE - University degree; experienced professional with deep expertise
5. PROFESSIONAL STANDARD - Deep/cross-disciplinary mastery
6. ORG. GENERALIST/FUNCTIONAL SPECIALIST - Significant impact across functions
7. BROAD PRACTICAL EXPERIENCE - Enterprise-level expertise
8. BROAD AND DEEP - Enterprise leadership expertise

NOTE: Consider role scope and complexity, not just tenure. Years are guidelines only.

───────────────────────────────────────────────────────────────────────────

TEAMS DIMENSION (What leadership responsibility?)

1. TEAM MEMBER - Individual contributor, no formal management
1.5. HYBRID - Project leadership without formal direct reports
2. TEAM LEADER - Leads small team (3-8 people), coaches, schedules work
2.5. HYBRID MANAGER - Partial people management + significant IC work
3. TEAMS MANAGER - Directs multiple teams, determines structure and strategy

───────────────────────────────────────────────────────────────────────────

BREADTH DIMENSION (Geographic scope)

1. DOMESTIC (0 pts) | 1.5. SUB-REGION (5 pts) | 2. REGIONAL (10 pts)
2.5. MULTIPLE REGIONS (15 pts) | 3. GLOBAL (20 pts)
"""

EVALUATION_PRINCIPLES = """
═══════════════════════════════════════════════════════════════════════════
UNIVERSAL IPE EVALUATION PRINCIPLES
═══════════════════════════════════════════════════════════════════════════

PRINCIPLE 1: IMPACT - Sphere of Influence vs. Operational Excellence

IMPACT 2 (OPERATION) - Executing within your area:
✓ Improves processes WITHIN own area of responsibility
✓ Achieves operational targets for their team/department
✓ Senior/lead titles that execute within functional boundaries

IMPACT 3 (TACTICAL) - Defining direction for the function:
✓ Creates standards/processes that THE ENTIRE FUNCTION follows
✓ Function-wide impact (16-20% of organization)
✓ Sets direction for how others work

CRITICAL: Sales roles with dealers = Frame 2 (External Shared), not Frame 4
CRITICAL: Do NOT mention years of experience when evaluating Knowledge
"""

REASONING_FRAMEWORK = """
For each dimension, explain your reasoning clearly using the official definitions.
"""

###############################
# Lookup Tables
###############################

IMPACT_CONTRIBUTION_TABLE = {
    1: {1:1, 2:2, 3:3, 4:4, 5:5},
    2: {1:3, 2:5, 3:6, 4:7, 5:8},
    3: {1:6, 2:7, 3:9, 4:10, 5:11},
    4: {1:9, 2:11, 3:12, 4:13, 5:14},
    5: {1:12, 2:13, 3:14, 4:15, 5:16},
}

IMPACT_SIZE_TABLE = {
    1: {1:5,1.5:5,2:5,2.5:5,3:5,3.5:5,4:5,4.5:5,5:5,5.5:5,6:5,6.5:5,7:5,7.5:5,8:5,8.5:5,9:5,9.5:5,10:5,10.5:5,11:5,11.5:5,12:5,12.5:5,13:5},
    1.5: {1:10,1.5:10,2:10,2.5:10,3:10,3.5:10,4:10,4.5:10,5:10,5.5:10,6:10,6.5:10,7:10,7.5:10,8:10,8.5:10,9:10,9.5:10,10:10,10.5:10,11:10,11.5:10,12:10,12.5:10,13:10},
    2: {1:15,1.5:15,2:15,2.5:15,3:15,3.5:15,4:15,4.5:15,5:15,5.5:15,6:15,6.5:15,7:15,7.5:15,8:15,8.5:15,9:15,9.5:15,10:15,10.5:15,11:15,11.5:15,12:15,12.5:15,13:15},
    2.5: {1:20,1.5:20,2:20,2.5:20,3:20,3.5:20,4:20,4.5:20,5:20,5.5:20,6:20,6.5:20,7:20,7.5:20,8:20,8.5:20,9:20,9.5:20,10:20,10.5:20,11:20,11.5:20,12:20,12.5:20,13:20},
    3: {1:25,1.5:25,2:25,2.5:25,3:25,3.5:25,4:25,4.5:25,5:25,5.5:25,6:25,6.5:25,7:25,7.5:25,8:25,8.5:25,9:25,9.5:25,10:25,10.5:25,11:25,11.5:25,12:25,12.5:25,13:25},
    3.5: {1:31,1.5:32,2:32,2.5:33,3:33,3.5:34,4:34,4.5:35,5:35,5.5:36,6:36,6.5:37,7:37,7.5:38,8:38,8.5:39,9:39,9.5:40,10:40,10.5:41,11:41,11.5:42,12:42,12.5:43,13:43},
    4: {1:37,1.5:38,2:39,2.5:40,3:41,3.5:42,4:43,4.5:44,5:45,5.5:46,6:47,6.5:48,7:49,7.5:50,8:51,8.5:52,9:53,9.5:54,10:55,10.5:56,11:57,11.5:58,12:59,12.5:60,13:61},
    4.5: {1:41,1.5:43,2:44,2.5:46,3:47,3.5:49,4:50,4.5:52,5:53,5.5:55,6:56,6.5:58,7:59,7.5:61,8:62,8.5:64,9:65,9.5:67,10:68,10.5:70,11:71,11.5:73,12:74,12.5:76,13:77},
    5: {1:44,1.5:46,2:48,2.5:50,3:52,3.5:54,4:56,4.5:58,5:60,5.5:62,6:64,6.5:66,7:68,7.5:70,8:72,8.5:74,9:76,9.5:78,10:80,10.5:82,11:84,11.5:86,12:88,12.5:90,13:92},
    5.5: {1:50,1.5:53,2:55,2.5:58,3:60,3.5:63,4:65,4.5:68,5:70,5.5:73,6:75,6.5:78,7:80,7.5:83,8:85,8.5:88,9:90,9.5:93,10:95,10.5:98,11:100,11.5:103,12:105,12.5:108,13:110},
    6: {1:56,1.5:59,2:62,2.5:65,3:68,3.5:71,4:74,4.5:77,5:80,5.5:83,6:86,6.5:89,7:92,7.5:95,8:98,8.5:101,9:104,9.5:107,10:110,10.5:113,11:116,11.5:119,12:122,12.5:125,13:128},
    6.5: {1:60,1.5:64,2:67,2.5:71,3:74,3.5:78,4:81,4.5:85,5:88,5.5:92,6:95,6.5:99,7:102,7.5:106,8:109,8.5:113,9:116,9.5:120,10:123,10.5:127,11:130,11.5:134,12:137,12.5:141,13:144},
    7: {1:63,1.5:67,2:71,2.5:75,3:79,3.5:83,4:87,4.5:91,5:95,5.5:99,6:103,6.5:107,7:111,7.5:115,8:119,8.5:123,9:127,9.5:131,10:135,10.5:139,11:143,11.5:147,12:151,12.5:155,13:159},
    7.5: {1:72,1.5:76,2:80,2.5:85,3:89,3.5:93,4:97,4.5:102,5:106,5.5:110,6:114,6.5:119,7:123,7.5:127,8:131,8.5:136,9:140,9.5:144,10:148,10.5:153,11:157,11.5:161,12:165,12.5:170,13:174},
    8: {1:80,1.5:85,2:89,2.5:94,3:98,3.5:103,4:107,4.5:112,5:116,5.5:121,6:125,6.5:130,7:134,7.5:139,8:143,8.5:148,9:152,9.5:157,10:161,10.5:166,11:170,11.5:175,12:179,12.5:184,13:188},
    8.5: {1:84,1.5:89,2:94,2.5:99,3:104,3.5:109,4:114,4.5:119,5:124,5.5:129,6:134,6.5:139,7:144,7.5:149,8:154,8.5:159,9:164,9.5:169,10:174,10.5:179,11:184,11.5:189,12:194,12.5:200,13:206},
    9: {1:87,1.5:93,2:98,2.5:104,3:109,3.5:115,4:120,4.5:126,5:131,5.5:137,6:142,6.5:148,7:153,7.5:159,8:164,8.5:170,9:175,9.5:181,10:186,10.5:192,11:197,11.5:203,12:208,12.5:216,13:224},
    9.5: {1:96,1.5:102,2:107,2.5:113,3:119,3.5:125,4:130,4.5:136,5:142,5.5:148,6:153,6.5:159,7:165,7.5:171,8:176,8.5:182,9:188,9.5:194,10:199,10.5:205,11:211,11.5:218,12:225,12.5:233,13:241},
    10: {1:104,1.5:110,2:116,2.5:122,3:128,3.5:134,4:140,4.5:146,5:152,5.5:158,6:164,6.5:170,7:176,7.5:182,8:188,8.5:194,9:200,9.5:206,10:212,10.5:218,11:224,11.5:233,12:241,12.5:250,13:258},
    10.5: {1:108,1.5:115,2:121,2.5:128,3:134,3.5:141,4:147,4.5:154,5:160,5.5:167,6:173,6.5:180,7:186,7.5:193,8:199,8.5:206,9:212,9.5:219,10:225,10.5:233,11:240,11.5:249,12:258,12.5:267,13:276},
    11: {1:111,1.5:118,2:125,2.5:132,3:139,3.5:146,4:153,4.5:160,5:167,5.5:174,6:181,6.5:188,7:195,7.5:202,8:209,8.5:216,9:223,9.5:230,10:237,10.5:247,11:256,11.5:266,12:275,12.5:285,13:294},
    11.5: {1:120,1.5:128,2:135,2.5:143,3:150,3.5:158,4:165,4.5:173,5:180,5.5:188,6:195,6.5:203,7:210,7.5:218,8:225,8.5:233,9:240,9.5:249,10:257,10.5:267,11:277,11.5:287,12:297,12.5:307,13:317},
    12: {1:128,1.5:136,2:144,2.5:152,3:160,3.5:168,4:176,4.5:184,5:192,5.5:200,6:208,6.5:216,7:224,7.5:232,8:240,8.5:248,9:256,9.5:267,10:277,10.5:288,11:298,11.5:309,12:319,12.5:330,13:340},
    12.5: {1:132,1.5:141,2:149,2.5:158,3:166,3.5:175,4:183,4.5:192,5:200,5.5:209,6:217,6.5:226,7:234,7.5:243,8:251,8.5:261,9:270,9.5:281,10:292,10.5:303,11:314,11.5:325,12:336,12.5:347,13:358},
    13: {1:135,1.5:144,2:153,2.5:162,3:171,3.5:180,4:189,4.5:198,5:207,5.5:216,6:225,6.5:234,7:243,7.5:252,8:261,8.5:273,9:284,9.5:296,10:307,10.5:319,11:330,11.5:342,12:353,12.5:365,13:376},
    13.5: {1:141,1.5:151,2:160,2.5:170,3:179,3.5:189,4:198,4.5:208,5:217,5.5:227,6:236,6.5:246,7:255,7.5:266,8:277,8.5:289,9:301,9.5:313,10:325,10.5:337,11:349,11.5:361,12:373,12.5:385,13:397},
    14: {1:147,1.5:157,2:167,2.5:177,3:187,3.5:197,4:207,4.5:217,5:227,5.5:237,6:247,6.5:257,7:267,7.5:280,8:292,8.5:305,9:317,9.5:330,10:342,10.5:355,11:367,11.5:380,12:392,12.5:405,13:417},
    14.5: {1:151,1.5:162,2:172,2.5:183,3:193,3.5:204,4:214,4.5:225,5:235,5.5:246,6:256,6.5:268,7:280,7.5:293,8:306,8.5:319,9:332,9.5:345,10:358,10.5:370,11:381,11.5:394,12:407,12.5:420,13:433},
    15: {1:155,1.5:166,2:177,2.5:188,3:199,3.5:210,4:221,4.5:232,5:243,5.5:254,6:265,6.5:279,7:292,7.5:306,8:319,8.5:333,9:346,9.5:360,10:373,10.5:384,11:395,11.5:409,12:422,12.5:436,13:449},
    15.5: {1:162,1.5:174,2:185,2.5:197,3:208,3.5:220,4:231,4.5:243,5:254,5.5:267,6:279,6.5:293,7:307,7.5:321,8:335,8.5:349,9:363,9.5:377,10:391,10.5:404,11:417,11.5:431,12:445,12.5:458,13:470},
    16: {1:168,1.5:180,2:192,2.5:204,3:216,3.5:228,4:240,4.5:252,5:264,5.5:279,6:293,6.5:308,7:322,7.5:337,8:351,8.5:366,9:380,9.5:395,10:409,10.5:424,11:438,11.5:453,12:467,12.5:479,13:491},
    16.5: {1:172,1.5:185,2:197,2.5:210,3:222,3.5:235,4:247,4.5:261,5:275,5.5:290,6:305,6.5:320,7:335,7.5:350,8:365,8.5:380,9:395,9.5:410,10:425,10.5:440,11:455,11.5:469,12:482,12.5:495,13:507},
    17: {1:176,1.5:189,2:202,2.5:215,3:228,3.5:241,4:254,4.5:270,5:285,5.5:301,6:316,6.5:332,7:347,7.5:363,8:378,8.5:394,9:409,9.5:425,10:440,10.5:456,11:471,11.5:484,12:497,12.5:510,13:523},
}

COMMUNICATION_FRAME_TABLE = {
    1: {1:10,1.5:18,2:25,2.5:28,3:30,3.5:38,4:45},
    1.5: {1:18,1.5:25,2:33,2.5:35,3:38,3.5:45,4:53},
    2: {1:25,1.5:33,2:40,2.5:43,3:45,3.5:53,4:60},
    2.5: {1:33,1.5:40,2:48,2.5:50,3:53,3.5:60,4:68},
    3: {1:40,1.5:48,2:55,2.5:58,3:60,3.5:68,4:75},
    3.5: {1:48,1.5:56,2:65,2.5:68,3:70,3.5:79,4:88},
    4: {1:55,1.5:65,2:75,2.5:78,3:80,3.5:90,4:100},
    4.5: {1:63,1.5:73,2:83,2.5:85,3:88,3.5:98,4:108},
    5: {1:70,1.5:80,2:90,2.5:93,3:95,3.5:105,4:115},
}

INNOVATION_COMPLEXITY_TABLE = {
    1: {1:10,1.5:13,2:15,2.5:18,3:20,3.5:23,4:25},
    1.5: {1:18,1.5:20,2:23,2.5:25,3:28,3.5:30,4:33},
    2: {1:25,1.5:28,2:30,2.5:33,3:35,3.5:38,4:40},
    2.5: {1:33,1.5:35,2:38,2.5:40,3:43,3.5:45,4:48},
    3: {1:40,1.5:43,2:45,2.5:48,3:50,3.5:53,4:55},
    3.5: {1:53,1.5:55,2:58,2.5:60,3:63,3.5:65,4:68},
    4: {1:65,1.5:68,2:70,2.5:73,3:75,3.5:78,4:80},
    4.5: {1:78,1.5:80,2:83,2.5:85,3:88,3.5:90,4:93},
    5: {1:90,1.5:93,2:95,2.5:98,3:100,3.5:103,4:105},
    5.5: {1:103,1.5:105,2:108,2.5:110,3:113,3.5:115,4:118},
    6: {1:115,1.5:118,2:120,2.5:123,3:125,3.5:128,4:130},
}

KNOWLEDGE_TEAMS_TABLE = {
    1: {1:15,1.5:33,2:50,2.5:63,3:75},
    1.5: {1:23,1.5:40,2:58,2.5:70,3:83},
    2: {1:30,1.5:48,2:65,2.5:78,3:90},
    2.5: {1:45,1.5:63,2:80,2.5:93,3:105},
    3: {1:60,1.5:78,2:95,2.5:108,3:120},
    3.5: {1:75,1.5:93,2:110,2.5:123,3:135},
    4: {1:90,1.5:108,2:125,2.5:138,3:150},
    4.5: {1:102,1.5:119,2:137,2.5:149,3:162},
    5: {1:113,1.5:131,2:148,2.5:161,3:173},
    5.5: {1:124,1.5:142,2:159,2.5:172,3:184},
    6: {1:135,1.5:153,2:170,2.5:183,3:195},
    6.5: {1:147,1.5:164,2:182,2.5:194,3:207},
    7: {1:158,1.5:176,2:193,2.5:206,3:218},
    7.5: {1:169,1.5:187,2:204,2.5:217,3:229},
    8: {1:180,1.5:198,2:215,2.5:228,3:240},
}

BREADTH_TABLE = {1: 0, 1.5: 5, 2: 10, 2.5: 15, 3: 20}

###############################
# API Helper Functions
###############################

def query_claude(prompt: str, system_prompt: str = "", temperature: float = 0.2, max_tokens: int = 4000) -> str:
    """Query Claude API and return text response."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not configured")
    
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data,
        timeout=120
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]

def query_claude_json(prompt: str, system_prompt: str = "", temperature: float = 0.2) -> dict:
    """Query Claude API and parse JSON response."""
    response_text = query_claude(prompt, system_prompt, temperature)
    
    # Extract JSON from response
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    else:
        return json.loads(response_text)

###############################
# Calculation Functions
###############################

def calculate_ipe_score(ratings: Dict, size: float) -> Tuple[Optional[int], Optional[Dict]]:
    """Calculate IPE score from dimension ratings."""
    try:
        # Impact × Contribution → Intermediate
        ic_intermediate = IMPACT_CONTRIBUTION_TABLE[int(ratings["impact"])][int(ratings["contribution"])]
        
        # Intermediate × Size → Impact/Contribution Points
        ic_points = IMPACT_SIZE_TABLE[ic_intermediate][size]
        
        # Communication × Frame → Points
        cf_points = COMMUNICATION_FRAME_TABLE[ratings["communication"]][ratings["frame"]]
        
        # Innovation × Complexity → Points
        innov_complex_points = INNOVATION_COMPLEXITY_TABLE[ratings["innovation"]][ratings["complexity"]]
        
        # Knowledge × Teams → Points
        kt_points = KNOWLEDGE_TEAMS_TABLE[ratings["knowledge"]][ratings["teams"]]
        
        # Breadth → Points
        breadth_points = BREADTH_TABLE[ratings["breadth"]]
        
        # Total Points (NOT including size)
        total_points = ic_points + cf_points + innov_complex_points + kt_points + breadth_points
        
        # Calculate IPE level using Excel formula: INT((total_points - 26) / 25 + 40)
        if total_points > 26:
            ipe_level = int((total_points - 26) / 25 + 40)
        else:
            ipe_level = 40
        
        # Calculate Job Level from IPE Level
        job_level_mapping = [
            (40, 41, 1), (42, 43, 2), (44, 45, 3), (46, 47, 4),
            (48, 50, 5), (51, 52, 6), (53, 55, 7), (56, 57, 8),
            (58, 59, 9), (60, 61, 10), (62, 65, 11), (66, 73, 12)
        ]
        
        job_level = 1
        for min_ipe, max_ipe, level in job_level_mapping:
            if min_ipe <= ipe_level <= max_ipe:
                job_level = level
                break
        
        breakdown = {
            "ic_points": ic_points,
            "cf_points": cf_points,
            "innov_complex_points": innov_complex_points,
            "kt_points": kt_points,
            "breadth_points": breadth_points,
            "total_points": total_points,
            "ipe_level": ipe_level,
            "job_level": job_level
        }
        
        return total_points, breakdown
        
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None, None

def evaluate_dimensions(title, purpose, deliverables, decision, people, financial, stakeholders, background, org_context):
    """Evaluate dimensions from structured inputs."""
    system_prompt = f"""You are an expert in Mercer IPE job evaluation.

{MERCER_IPE_FRAMEWORK}

{EVALUATION_PRINCIPLES}

{REASONING_FRAMEWORK}

CRITICAL: Do NOT mention years of experience when evaluating Knowledge dimension.

Return ONLY valid JSON."""
    
    prompt = f"""
Evaluate this role across IPE dimensions.

Job Title: {title}
Purpose: {purpose}
Deliverables: {deliverables}
Decision Authority: {decision}
People Responsibility: {people}
Financial Responsibility: {financial}
Stakeholders: {stakeholders}
Background: {background}
{f"Context: {org_context}" if org_context else ""}

Return JSON:
{{
  "impact": {{"value": X, "reasoning": "..."}},
  "contribution": {{"value": X, "reasoning": "..."}},
  "communication": {{"value": X, "reasoning": "..."}},
  "frame": {{"value": X, "reasoning": "..."}},
  "innovation": {{"value": X, "reasoning": "..."}},
  "complexity": {{"value": X, "reasoning": "..."}},
  "knowledge": {{"value": X, "reasoning": "..."}},
  "breadth": {{"value": X, "reasoning": "..."}},
  "principles_applied": ["list of principles"]
}}
"""
    
    result = query_claude_json(prompt, system_prompt)
    
    ratings = {}
    justifications = {}
    principles = result.get("principles_applied", [])
    
    for dim in ["impact", "contribution", "communication", "frame", "innovation", "complexity", "knowledge", "breadth"]:
        if dim in result:
            ratings[dim] = result[dim].get("value", 0)
            justifications[dim] = result[dim].get("reasoning", "")
    
    ratings["teams"] = people
    justifications["teams"] = f"User-provided: {people}"
    
    return ratings, justifications, principles

def display_evaluation_results(ipe_score, breakdown, ratings, justifications, principles, job_title):
    """Display evaluation results."""
    st.success(f"✅ IPE Evaluation Complete for: **{job_title}**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Points", breakdown["total_points"])
    with col2:
        st.metric("IPE Level", breakdown["ipe_level"])
    with col3:
        st.metric("Job Level", breakdown["job_level"])
    
    st.markdown("### Point Breakdown")
    breakdown_df = pd.DataFrame([
        {"Component": "Impact × Contribution × Size", "Points": breakdown["ic_points"]},
        {"Component": "Communication × Frame", "Points": breakdown["cf_points"]},
        {"Component": "Innovation × Complexity", "Points": breakdown["innov_complex_points"]},
        {"Component": "Knowledge × Teams", "Points": breakdown["kt_points"]},
        {"Component": "Breadth", "Points": breakdown["breadth_points"]},
    ])
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    
    st.markdown("### Dimension Ratings")
    dims_df = pd.DataFrame([
        {"Dimension": dim.capitalize(), "Rating": ratings[dim]}
        for dim in ["impact", "contribution", "communication", "frame", "innovation", "complexity", "knowledge", "teams", "breadth"]
    ])
    st.dataframe(dims_df, use_container_width=True, hide_index=True)
    
    with st.expander("📋 View Detailed Reasoning"):
        for dim, reasoning in justifications.items():
            st.markdown(f"**{dim.upper()}:** {reasoning}")

###############################
# JD Generation Functions
###############################

def interpret_questionnaire(answers: Dict) -> Dict:
    """Interpret questionnaire answers and determine IPE ratings."""
    
    # Extract only structured multiple-choice answers for rating determination
    # Exclude free-text fields that may contain inflated or inaccurate language
    structured_answers = {
        k: v for k, v in answers.items() 
        if k not in ['role_description', 'role_title', 'key_responsibilities', 
                     'education_requirements', 'key_skills']
    }
    
    interpretation_prompt = f"""
You are an expert in Mercer IPE methodology. Based on these questionnaire answers, determine the appropriate IPE dimension ratings.

{MERCER_IPE_FRAMEWORK}

QUESTIONNAIRE ANSWERS:
{json.dumps(structured_answers, indent=2)}

CRITICAL INSTRUCTIONS:
- Analyze answers carefully
- Map to IPE dimensions using official framework
- Be conservative (avoid inflation)
- Impact: Must use INTEGERS ONLY (1, 2, 3, 4, or 5)
- ALL other dimensions: Can and should use HALF-STEPS (e.g., 2.5, 3.5) when role falls between levels
- Knowledge: Focus on expertise depth/breadth required, NOT years of experience
- Show clear reasoning

Return JSON:
{{
  "impact": {{"value": X, "reasoning": "..."}},
  "contribution": {{"value": X, "reasoning": "..."}},
  "communication": {{"value": X, "reasoning": "..."}},
  "frame": {{"value": X, "reasoning": "..."}},
  "innovation": {{"value": X, "reasoning": "..."}},
  "complexity": {{"value": X, "reasoning": "..."}},
  "knowledge": {{"value": X, "reasoning": "..."}}
}}
"""
    
    result = query_claude_json(
        interpretation_prompt,
        system_prompt="You are a Mercer IPE expert. Map answers to IPE dimensions accurately without mentioning years.",
        temperature=0.2
    )
    
    dimensions = {}
    reasoning_text = ""
    
    for dim in ["impact", "contribution", "communication", "frame", "innovation", "complexity", "knowledge"]:
        if dim in result:
            dimensions[dim] = result[dim].get("value", 0)
            reasoning_text += f"\n**{dim.upper()}:** {result[dim].get('reasoning', '')}\n"
    
    return {
        'dimensions': dimensions,
        'reasoning': reasoning_text,
        'success': True
    }

def generate_job_description(answers: Dict, dimensions: Dict) -> str:
    """Generate professional JD from questionnaire."""
    
    # Build IPE context for each dimension target
    ipe_context = f"""
OFFICIAL MERCER IPE FRAMEWORK CONTEXT:

{MERCER_IPE_FRAMEWORK}

TARGET IPE LEVELS FOR THIS ROLE:
- Impact: {dimensions.get('impact', 'N/A')}
- Contribution: {dimensions.get('contribution', 'N/A')}
- Communication: {dimensions.get('communication', 'N/A')}
- Frame: {dimensions.get('frame', 'N/A')}
- Innovation: {dimensions.get('innovation', 'N/A')}
- Complexity: {dimensions.get('complexity', 'N/A')}
- Knowledge: {dimensions.get('knowledge', 'N/A')}

CRITICAL: The job description must authentically reflect these IPE levels. Use language and describe responsibilities that clearly demonstrate why each dimension is rated at its target level.

CRITICAL: The target IPE levels are authoritative. If the user's role descriptions sound more senior or junior than the targets, rewrite the content to accurately match the target levels.
"""
    
    generation_prompt = f"""
Generate a professional job description that accurately reflects the IPE dimension targets.

{ipe_context}

QUESTIONNAIRE ANSWERS:
{json.dumps(answers, indent=2)}

STRUCTURE (use proper markdown formatting):

**Objectives**

Write 3-6 sentences in flowing paragraph format describing the purpose and primary objective of the role. The language must reflect Impact level {dimensions.get('impact')} and Contribution level {dimensions.get('contribution')}.

**Summary of Responsibilities**

- First core responsibility based on questionnaire
- Second core responsibility based on questionnaire
- Third core responsibility based on questionnaire
- Fourth core responsibility based on questionnaire
- Fifth core responsibility based on questionnaire

(5-8 bullet points total, each starting with "- " on its own line)

**Scope of Decision Making**

Write a flowing paragraph describing decision-making authority that clearly demonstrates Impact level {dimensions.get('impact')}.

**Experience and Qualifications**

- First qualification based on education/knowledge
- Second qualification based on education/knowledge
- Third qualification based on education/knowledge

**Skills and Capabilities**

- First skill based on communication/innovation/complexity
- Second skill based on communication/innovation/complexity
- Third skill based on communication/innovation/complexity
- Fourth skill based on communication/innovation/complexity

FORMATTING RULES:
- Headers: Use **Header** format (bold with double asterisks)
- Bullets: Each bullet must start with "- " and be on its own line
- Paragraphs: Flowing text without bullets
- Generate specific, role-appropriate content
- Language intensity MUST match the target IPE levels

Generate the complete, properly formatted job description now:
"""
    
    response = query_claude(
        generation_prompt,
        system_prompt=f"You are an expert HR professional and Mercer IPE specialist. Write job descriptions that authentically reflect IPE levels.\n\n{EVALUATION_PRINCIPLES}",
        temperature=0.3,
        max_tokens=4000
    )
    
    return response

def get_ipe_level_definition(dimension: str, level: float) -> str:
    """Extract the official IPE definition for a specific dimension and level."""
    level_int = int(level)
    
    # Map dimensions to their section markers in MERCER_IPE_FRAMEWORK
    dimension_map = {
        'impact': 'IMPACT DIMENSION',
        'contribution': 'CONTRIBUTION DIMENSION',
        'communication': 'COMMUNICATION DIMENSION',
        'frame': 'FRAME DIMENSION',
        'innovation': 'INNOVATION DIMENSION',
        'complexity': 'COMPLEXITY DIMENSION',
        'knowledge': 'KNOWLEDGE DIMENSION'
    }
    
    if dimension not in dimension_map:
        return f"Level {level}"
    
    # Extract the section for this dimension
    framework_text = MERCER_IPE_FRAMEWORK
    section_start = framework_text.find(dimension_map[dimension])
    if section_start == -1:
        return f"Level {level}"
    
    # Find the next section separator
    next_section = framework_text.find("───────────", section_start + 1)
    if next_section == -1:
        section = framework_text[section_start:]
    else:
        section = framework_text[section_start:next_section]
    
    # Extract the specific level definition
    lines = section.split('\n')
    level_def = []
    capturing = False
    
    for line in lines:
        # Look for the level number at the start of a line
        if line.strip().startswith(f"{level_int}."):
            capturing = True
            level_def.append(line.strip())
        elif capturing:
            # Continue capturing until we hit another number or separator
            if line.strip() and (line.strip()[0].isdigit() or '───' in line):
                break
            if line.strip():
                level_def.append(line.strip())
    
    if level_def:
        return ' '.join(level_def)
    else:
        return f"Level {level}"

def re_evaluate_jd(jd_text: str, teams: float, breadth: float) -> Dict:
    """Re-evaluate a generated JD to check alignment with targets."""
    
    system_prompt = f"""You are an expert in Mercer IPE evaluation.

{MERCER_IPE_FRAMEWORK}

{EVALUATION_PRINCIPLES}

CRITICAL RATING INSTRUCTIONS:
- Impact: Must use INTEGERS ONLY (1, 2, 3, 4, or 5)
- ALL other dimensions: Can and should use HALF-STEPS (e.g., 2.5, 3.5) when role falls between levels
- Knowledge: Focus on expertise depth/breadth required, NOT years of experience

Return ONLY valid JSON."""
    
    prompt = f"""
Re-evaluate this job description to check IPE dimension alignment.

JOB DESCRIPTION:
{jd_text}

Return JSON:
{{
  "impact": {{"value": X, "reasoning": "..."}},
  "contribution": {{"value": X, "reasoning": "..."}},
  "communication": {{"value": X, "reasoning": "..."}},
  "frame": {{"value": X, "reasoning": "..."}},
  "innovation": {{"value": X, "reasoning": "..."}},
  "complexity": {{"value": X, "reasoning": "..."}},
  "knowledge": {{"value": X, "reasoning": "..."}}
}}
"""
    
    result = query_claude_json(prompt, system_prompt, temperature=0.2)
    return result

###############################
# Main Application
###############################

def main():
    st.title("🎯 IPE Job Evaluator")
    st.markdown("**Universal Mercer IPE Framework Implementation**")
    
    # Mode selection
    mode = st.radio(
        "Select Mode:",
        ["Evaluate Only", "Create & Evaluate"],
        horizontal=True
    )
    
    st.markdown("---")
    
    ###############################
    # MODE 1: EVALUATE ONLY
    ###############################
    
    if mode == "Evaluate Only":
        st.header("📄 Evaluate Existing Job Description")
        
        jd_text = st.text_area("Job Description (paste complete JD) *", height=400)
        job_title = st.text_input("Job Title (optional)")
        
        st.markdown("### Organizational Context")
        col1, col2, col3 = st.columns(3)
        with col1:
            size = st.slider("Organization Size (1-13)", min_value=1.0, max_value=13.0, value=10.0, step=0.5)
        with col2:
            teams_input = st.selectbox("Teams", [1, 1.5, 2, 2.5, 3], index=0, format_func=lambda x: {
                1: "1 - Individual Contributor",
                1.5: "1.5 - Project Lead",
                2: "2 - Team Leader",
                2.5: "2.5 - Hybrid Manager",
                3: "3 - Teams Manager"
            }.get(x, str(x)))
        with col3:
            breadth_input = st.selectbox("Breadth", [1, 1.5, 2, 2.5, 3], index=0, format_func=lambda x: {
                1: "1 - Domestic",
                1.5: "1.5 - Sub-Region",
                2: "2 - Regional",
                2.5: "2.5 - Multiple Regions",
                3: "3 - Global"
            }.get(x, str(x)))
        
        if st.button("🔍 Evaluate IPE", use_container_width=True):
            if not jd_text.strip():
                st.error("Please paste a job description")
            else:
                with st.spinner("Evaluating..."):
                    try:
                        system_prompt = f"""You are an expert in Mercer IPE evaluation.

{MERCER_IPE_FRAMEWORK}

{EVALUATION_PRINCIPLES}

{REASONING_FRAMEWORK}

CRITICAL RATING INSTRUCTIONS:
- Impact: Must use INTEGERS ONLY (1, 2, 3, 4, or 5)
- ALL other dimensions (Contribution, Communication, Frame, Innovation, Complexity, Knowledge): Can and should use HALF-STEPS (e.g., 2.5, 3.5) when the role falls between levels
- Use half-steps when a JD shows characteristics of two adjacent levels
- Knowledge: Focus on expertise depth/breadth required, NOT years of experience

Return ONLY valid JSON."""
                        
                        prompt = f"""
Evaluate this job description.

{f"Job Title: {job_title}" if job_title else ""}

JOB DESCRIPTION:
{jd_text}

USER PROVIDED:
- Teams: {teams_input}
- Breadth: {breadth_input}

REMEMBER: Impact must be integer (1-5), all other dimensions can use half-steps (e.g., 2.5, 3.5).

Return JSON:
{{
  "impact": {{"value": X, "reasoning": "..."}},
  "contribution": {{"value": X, "reasoning": "..."}},
  "communication": {{"value": X, "reasoning": "..."}},
  "frame": {{"value": X, "reasoning": "..."}},
  "innovation": {{"value": X, "reasoning": "..."}},
  "complexity": {{"value": X, "reasoning": "..."}},
  "knowledge": {{"value": X, "reasoning": "..."}},
  "principles_applied": ["list"]
}}
"""
                        
                        result = query_claude_json(prompt, system_prompt, temperature=0.2)
                        
                        ratings = {}
                        justifications = {}
                        principles = result.get("principles_applied", [])
                        
                        for dim in ["impact", "contribution", "communication", "frame", "innovation", "complexity", "knowledge"]:
                            if dim in result:
                                ratings[dim] = result[dim].get("value", 0)
                                justifications[dim] = result[dim].get("reasoning", "")
                        
                        ratings["teams"] = teams_input
                        ratings["breadth"] = breadth_input
                        justifications["teams"] = f"User-provided: {teams_input}"
                        justifications["breadth"] = f"User-provided: {breadth_input}"
                        
                        ipe_score, breakdown = calculate_ipe_score(ratings, size)
                        
                        if ipe_score:
                            display_evaluation_results(ipe_score, breakdown, ratings, justifications, principles, job_title or "Job")
                        else:
                            st.error("Could not calculate IPE score")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.exception(e)
    
    ###############################
    # MODE 2: CREATE & EVALUATE
    ###############################
    
    else:
        st.header("✨ Create & Evaluate New Job Description")
        st.markdown("Answer the questions to generate a professional JD and automatic IPE evaluation.")
        
        # Initialize session state
        if 'jd_results' not in st.session_state:
            st.session_state.jd_results = None
        
        with st.form("jd_questionnaire"):
            st.subheader("Role Overview")
            
            role_description = st.text_area(
                "1. Briefly describe this role in your own words (2-4 sentences):",
                height=150,
                placeholder="What does this person do? What's the main purpose of the role?"
            )
            
            role_title = st.text_input(
                "2. What is the job title?",
                placeholder="e.g., Senior Accounting Controller, Sales Manager"
            )
            
            st.markdown("---")
            st.subheader("Scope & Organizational Reach")
            
            decision_scope = st.radio(
                "3. What is the scope and nature of decisions this role makes?",
                [
                    "Follows defined procedures and standards and delivers with minimal decision-making authority",
                    "Makes operational decisions within defined targets and guidelines and achieves short-term objectives within their area",
                    "Develops plans, processes, or standards that guide their function (1-3 years)",
                    "Influences organization-wide strategy and long-term direction (3-5 years)",
                    "Defines the organization's vision, mission, and strategic direction (5+ years)"
                ]
            )
            
            standards_authority = st.radio(
                "4. Does this role create standards/processes that others must follow?",
                [
                    "No - follows existing standards",
                    "Can suggest improvements, others decide",
                    "Yes - creates standards for own team/area",
                    "Yes - creates standards for entire function",
                    "Yes - creates standards across multiple functions"
                ]
            )
            
            decision_authority = st.radio(
                "5. What decisions can this role make independently?",
                [
                    "Day-to-day task decisions only",
                    "Operational decisions within their area",
                    "Significant decisions affecting their department",
                    "Strategic decisions for their function",
                    "Enterprise-level strategic decisions"
                ]
            )
            
            st.markdown("---")
            st.subheader("Responsibilities & Contribution")
            
            key_responsibilities = st.text_area(
                "6. What are the 3-5 most important responsibilities?",
                height=150,
                placeholder="List the core things this role delivers"
            )
            
            contribution_type = st.radio(
                "7. How does this role contribute?",
                [
                    "Follows instructions to complete tasks",
                    "Provides specialized support",
                    "Analyzes and makes recommendations",
                    "Improves and optimizes approaches",
                    "Designs new solutions and strategies"
                ]
            )
            
            st.markdown("---")
            st.subheader("Required Expertise")
            
            education_required = st.radio(
                "8. What education level is required?",
                [
                    "High school or equivalent",
                    "Vocational/specialized training",
                    "Bachelor's degree",
                    "Master's degree",
                    "PhD or equivalent"
                ]
            )
            
            knowledge_scope = st.radio(
                "9. What level of expertise does this role require?",
                [
                    "Basic foundational skills and procedures",
                    "Deep specialist in one specific area",
                    "Expert across one functional area",
                    "Deep in one area + knowledge of 1-2 related areas",
                    "Broad expertise across multiple areas in one function",
                    "Significant expertise across multiple functions"
                ]
            )
            
            organizational_knowledge = st.radio(
                "10. How much understanding of other parts of the organization?",
                [
                    "Minimal - focuses on own area",
                    "Some - basic understanding of related areas",
                    "Significant - regularly works across areas",
                    "Extensive - understands multiple functions deeply"
                ]
            )
            
            st.markdown("---")
            st.subheader("Problem Solving & Innovation")
            
            problem_structure = st.radio(
                "11. How well-defined are the problems this role needs to solve?",
                [
                    "Well-defined with clear procedures",
                    "Some ambiguity, generally clear",
                    "Often unclear - must figure out the problem",
                    "Highly ambiguous - define problem and solution"
                ]
            )
            
            solution_approach = st.radio(
                "12. When solving problems, this role:",
                [
                    "Follows established procedures",
                    "Adapts existing methods",
                    "Designs new approaches for their area",
                    "Creates frameworks others can use",
                    "Develops breakthrough innovations"
                ]
            )
            
            dimensions_considered = st.multiselect(
                "13. To solve problems, must consider: (select all)",
                [
                    "Operational processes and workflows",
                    "Financial implications and budgets",
                    "People, organizational structure, change management"
                ]
            )
            
            discipline_span = st.radio(
                "14. Does work stay within one discipline or span multiple?",
                [
                    "Stays within one discipline",
                    "Occasionally touches adjacent areas",
                    "Regularly works across multiple disciplines",
                    "Spans multiple functions enterprise-wide"
                ]
            )
            
            st.markdown("---")
            st.subheader("Communication & Stakeholders")
            
            communication_type = st.radio(
                "15. What describes typical communication?",
                [
                    "Share information and provide updates",
                    "Explain concepts and reach agreement",
                    "Persuade others to adopt new ideas",
                    "Negotiate agreements on proposals",
                    "Build strategic consensus on long-term direction"
                ]
            )
            
            resistance_frequency = st.radio(
                "16. How often must convince people who initially disagree?",
                [
                    "Rarely - mostly shares information",
                    "Sometimes - occasional persuasion",
                    "Often - regularly needs to persuade",
                    "Constantly - persuasion is core"
                ]
            )
            
            primary_stakeholders = st.multiselect(
                "17. Primary stakeholders? (select all)",
                [
                    "Their own team/work group",
                    "Other internal departments/teams",
                    "Senior management/executives",
                    "External partners/vendors (aligned interests)",
                    "External customers/clients",
                    "Parties with conflicting interests (legal, disputes)"
                ]
            )
            
            stakeholder_alignment = st.radio(
                "18. Stakeholders typically:",
                [
                    "Have aligned goals - same outcome",
                    "Have different priorities needing balancing",
                    "Have competing interests creating tension"
                ]
            )
            
            st.markdown("---")
            st.subheader("Leadership & Scope")
            
            teams = st.selectbox(
                "19. Team leadership responsibility:",
                [1.0, 1.5, 2.0, 2.5, 3.0],
                format_func=lambda x: {
                    1.0: "1 - Individual contributor",
                    1.5: "1.5 - Leads projects, no direct reports",
                    2.0: "2 - Manages a team",
                    2.5: "2.5 - Manages multiple teams/leaders",
                    3.0: "3 - Senior leadership over functions"
                }[x]
            )
            
            breadth = st.selectbox(
                "20. Geographic scope:",
                [1.0, 1.5, 2.0, 2.5, 3.0],
                format_func=lambda x: {
                    1.0: "1 - Local",
                    1.5: "1.5 - Sub-regional",
                    2.0: "2 - Regional",
                    2.5: "2.5 - Multi-regional",
                    3.0: "3 - Global"
                }[x]
            )
            
            size = st.slider(
                "21. Organization Size:",
                min_value=1.0,
                max_value=13.0,
                value=10.0,
                step=0.5,
                help="**CRITICAL:** Contact Total Rewards if you don't know the exact size. Incorrect size invalidates the evaluation."
            )
            
            st.markdown("---")
            st.subheader("Qualifications & Skills")
            
            education_requirements = st.text_area(
                "23. What education/certifications are required?",
                height=100,
                placeholder="e.g., Bachelor's in Accounting, CPA preferred"
            )
            
            key_skills = st.text_area(
                "24. Key skills and capabilities needed?",
                height=150,
                placeholder="List technical skills, soft skills, competencies"
            )
            
            st.markdown("---")
            
            submitted = st.form_submit_button("🚀 Generate & Evaluate", type="primary")
        
        # Process submission OUTSIDE form
        if submitted:
            if not role_title or not role_description:
                st.error("Please provide role title and description")
            else:
                answers = {
                    "role_description": role_description,
                    "role_title": role_title,
                    "decision_scope": decision_scope,
                    "standards_authority": standards_authority,
                    "decision_authority": decision_authority,
                    "key_responsibilities": key_responsibilities,
                    "contribution_type": contribution_type,
                    "education_required": education_required,
                    "knowledge_scope": knowledge_scope,
                    "organizational_knowledge": organizational_knowledge,
                    "problem_structure": problem_structure,
                    "solution_approach": solution_approach,
                    "dimensions_considered": dimensions_considered,
                    "discipline_span": discipline_span,
                    "communication_type": communication_type,
                    "resistance_frequency": resistance_frequency,
                    "primary_stakeholders": primary_stakeholders,
                    "stakeholder_alignment": stakeholder_alignment,
                    "teams": teams,
                    "breadth": breadth,
                    "size": size,
                    "education_requirements": education_requirements,
                    "key_skills": key_skills
                }
                
                try:
                    # Step 1: Interpret
                    with st.spinner("Step 1/3: Analyzing answers..."):
                        interpretation = interpret_questionnaire(answers)
                    
                    if not interpretation['success']:
                        st.error(f"Failed to interpret: {interpretation.get('error')}")
                        st.stop()
                    
                    dimensions = interpretation['dimensions']
                    st.success("✅ Step 1 complete: Questionnaire analyzed")
                    
                    # Step 2: Generate JD with IPE context
                    with st.spinner("Step 2/4: Generating job description with IPE alignment..."):
                        jd_text = generate_job_description(answers, dimensions)
                    
                    st.success("✅ Step 2 complete: Job description generated")
                    
                    # Step 3: Re-evaluate the generated JD
                    with st.spinner("Step 3/4: Re-evaluating generated JD for alignment..."):
                        reevaluation = re_evaluate_jd(jd_text, teams, breadth)
                        
                        # Check for mismatches
                        mismatches = []
                        tolerance = 0.5  # Allow 0.5 difference
                        
                        for dim in ['impact', 'contribution', 'communication', 'frame', 'innovation', 'complexity', 'knowledge']:
                            expected = dimensions[dim]
                            actual = reevaluation[dim].get('value', 0)
                            if abs(expected - actual) > tolerance:
                                mismatches.append(f"{dim}: expected {expected}, got {actual}")
                    
                    # Step 3b: Regenerate if needed (up to 2 attempts)
                    regeneration_count = 0
                    max_regenerations = 2
                    
                    while mismatches and regeneration_count < max_regenerations:
                        regeneration_count += 1
                        st.warning(f"⚠️ Alignment issues detected (attempt {regeneration_count}/{max_regenerations}): {', '.join(mismatches)}")
                        
                        with st.spinner(f"Step 3.{regeneration_count}/4: Regenerating JD with corrections..."):
                            # Build specific correction feedback using official IPE definitions
                            correction_context = "\n\n═══════════════════════════════════════════\n"
                            correction_context += "REGENERATION REQUIRED - ALIGNMENT ISSUES\n"
                            correction_context += "═══════════════════════════════════════════\n\n"
                            
                            for dim in ['impact', 'contribution', 'communication', 'frame', 'innovation', 'complexity', 'knowledge']:
                                expected = dimensions[dim]
                                actual = reevaluation[dim].get('value', 0)
                                
                                if abs(expected - actual) > tolerance:
                                    expected_def = get_ipe_level_definition(dim, expected)
                                    actual_def = get_ipe_level_definition(dim, actual)
                                    
                                    if actual > expected:
                                        correction_context += f"**{dim.upper()} is TOO HIGH** (got {actual}, need {expected}):\n"
                                        correction_context += f"❌ Current JD language suggests: {actual_def}\n"
                                        correction_context += f"✅ Rewrite to match: {expected_def}\n\n"
                                    else:
                                        correction_context += f"**{dim.upper()} is TOO LOW** (got {actual}, need {expected}):\n"
                                        correction_context += f"❌ Current JD language suggests: {actual_def}\n"
                                        correction_context += f"✅ Elevate to match: {expected_def}\n\n"
                            
                            correction_context += "Generate a NEW job description that accurately reflects the target levels above."
                            
                            answers['_correction_feedback'] = correction_context
                            
                            jd_text = generate_job_description(answers, dimensions)
                            reevaluation = re_evaluate_jd(jd_text, teams, breadth)
                            
                            # Recheck
                            mismatches = []
                            for dim in ['impact', 'contribution', 'communication', 'frame', 'innovation', 'complexity', 'knowledge']:
                                expected = dimensions[dim]
                                actual = reevaluation[dim].get('value', 0)
                                if abs(expected - actual) > tolerance:
                                    mismatches.append(f"{dim}: expected {expected}, got {actual}")
                    
                    if mismatches:
                        st.info(f"ℹ️ Minor alignment differences remain after {max_regenerations} attempts. Proceeding with current JD.")
                    else:
                        st.success("✅ Step 3 complete: JD alignment verified")
                    
                    # Step 4: Final evaluation with re-evaluated dimensions
                    with st.spinner("Step 4/4: Calculating final IPE score..."):
                        # Use the re-evaluated dimensions for final score
                        dimensions['teams'] = teams
                        dimensions['breadth'] = breadth
                        dimensions['size'] = size
                        
                        # Update dimensions with re-evaluated values
                        for dim in ['impact', 'contribution', 'communication', 'frame', 'innovation', 'complexity', 'knowledge']:
                            dimensions[dim] = reevaluation[dim].get('value', dimensions[dim])
                        
                        ratings = {
                            "impact": dimensions['impact'],
                            "contribution": dimensions['contribution'],
                            "communication": dimensions['communication'],
                            "frame": dimensions['frame'],
                            "innovation": dimensions['innovation'],
                            "complexity": dimensions['complexity'],
                            "knowledge": dimensions['knowledge'],
                            "teams": teams,
                            "breadth": breadth
                        }
                        
                        ipe_score, breakdown = calculate_ipe_score(ratings, size)
                    
                    st.success("✅ Step 4 complete: Evaluation finished")
                    
                    # Build reasoning text from re-evaluation
                    reasoning_text = ""
                    for dim in ['impact', 'contribution', 'communication', 'frame', 'innovation', 'complexity', 'knowledge']:
                        reasoning_text += f"\n**{dim.upper()}:** {reevaluation[dim].get('reasoning', 'N/A')}\n"
                    
                    # Store results
                    st.session_state.jd_results = {
                        'jd_text': jd_text,
                        'role_title': role_title,
                        'breakdown': breakdown,
                        'ratings': ratings,
                        'reasoning': reasoning_text
                    }
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)
        
        # Display results if they exist
        if st.session_state.jd_results:
            results = st.session_state.jd_results
            
            st.success("✅ Complete! Job description generated and evaluated.")
            
            st.markdown("---")
            
            # Show JD
            st.subheader("📄 Generated Job Description")
            st.markdown(results['jd_text'])
            
            st.markdown("")
            
            # Download button
            st.download_button(
                label="⬇️ Download Job Description",
                data=results['jd_text'],
                file_name=f"{results['role_title'].replace(' ', '_')}_JD.txt",
                mime="text/plain"
            )
            
            st.markdown("---")
            
            # Show results
            st.subheader("🎯 IPE Evaluation Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Points", results['breakdown']['total_points'])
            with col2:
                st.metric("IPE Level", results['breakdown']['ipe_level'])
            with col3:
                st.metric("Job Level", results['breakdown']['job_level'])
            
            # Dimensions
            st.markdown("**Dimension Ratings:**")
            dims_df = pd.DataFrame([
                {"Dimension": dim.capitalize(), "Rating": results['ratings'][dim]}
                for dim in ["impact", "contribution", "communication", "frame", "innovation", "complexity", "knowledge", "teams", "breadth"]
            ])
            st.dataframe(dims_df, use_container_width=True, hide_index=True)
            
            # Reasoning
            with st.expander("📋 View Dimension Reasoning"):
                st.markdown(results['reasoning'])
            
            st.markdown("---")
            st.info("💡 **Next Steps:** Review the JD and evaluation. Refine answers and regenerate if needed, or manually edit the JD and re-evaluate using 'Evaluate Only' mode.")

if __name__ == "__main__":
    main()
