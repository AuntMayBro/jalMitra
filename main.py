import os
import pandas as pd
import spacy
import json
import re
import ast
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from thefuzz import process, fuzz
from jalMitra.list import districts, blocks

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# --- FastAPI Application ---
app = FastAPI(
    title="Data Query API",
    description="An API to query district and block data and get structured responses.",
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Objects and Pre-loading ---
try:
    df = pd.read_excel('jalMitra/MP_21-25.xlsx')
    df.columns = df.columns.str.strip().str.lower()
    columns_to_process = ['state', 'district', 'block']
    for column in columns_to_process:
        if column in df.columns:
            df[column] = df[column].astype(str).str.lower()
except FileNotFoundError:
    print("Warning: 'MP_21-25.xlsx' not found. A script using a dataframe is unlikely to work.")
    df = pd.DataFrame()

all_locations = districts + blocks

nlp = spacy.load("en_core_web_sm")
if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = []
    patterns += [{"label": "DISTRICT", "pattern": d} for d in districts]
    patterns += [{"label": "BLOCK", "pattern": b} for b in blocks]
    ruler.add_patterns(patterns)


# --- Helper Functions ---
def correct_spellings(text, known_words, score_cutoff=80):
    corrected_text = []
    words = text.replace(",", " ").split()
    for word in words:
        if not isinstance(word, str):
            corrected_text.append(word)
            continue
        if word.lower() in known_words:
            corrected_text.append(word)
            continue
        string_known_words = [w for w in known_words if isinstance(w, str)]
        if not string_known_words:
            corrected_text.append(word)
            continue
        best_match, score = process.extractOne(word, string_known_words, scorer=fuzz.ratio)
        if score >= score_cutoff:
            corrected_text.append(best_match)
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

def extract_locations(text_input):
    text_corrected = correct_spellings(text_input.lower(), all_locations)
    doc = nlp(text_corrected)
    found_districts = []
    found_blocks = []
    for ent in doc.ents:
        entity_text = ent.text.lower()
        if ent.label_ == "DISTRICT":
            found_districts.append(entity_text)
        elif ent.label_ == "BLOCK":
            found_blocks.append(entity_text)
    return list(set(found_districts)), list(set(found_blocks))

# --- Agent Prompt ---
AGENT_PREFIX = """
You are a data analysis agent working with a pandas dataframe in Python named `pdf`.
Your task is to generate a single, valid JSON object based on the user's query.

CRITICAL INSTRUCTIONS:
- Your response MUST be a single, valid JSON object and nothing else.
- DO NOT add any comments, explanations, or any text outside of the JSON structure.
- Start your response immediately with {{ and end it with }}.

The JSON object must have exactly two keys: "answer" and "graphs".
- "answer": A clear, well-written paragraph summarizing the key findings, trends, and insights from the analysis.
- "graphs": A list of dictionaries, where each dictionary represents a graph.
  - Allowed graph types: "bar" or "line".
  - Each dictionary must include:
    - "xlabel": Label for the x-axis.
    - "ylabel": Label for the y-axis.
    - "graph_type": Both "bar" and "line".
    - "graph_data": A list of {{"x": <x_value>, "y1": <y_value>}} objects.

Example of a perfect, complete output:
{{
  "answer": "Between 2010 and 2020, rainfall patterns in Jabalpur showed a steady increase, while Indore remained relatively stable. Jabalpur consistently received more rainfall overall, which highlights its higher dependency on monsoon contributions compared to Indore.",
  "graphs": [
    {{
      "xlabel": "District",
      "ylabel": "Average Rainfall Total",
      "graph_type": "bar",
      "graph_data": [
        {{"x": "indore", "y1": 924.52}},
        {{"x": "jabalpur", "y1": 1184.87}}
      ]
    }},
    {{
      "xlabel": "Year",
      "ylabel": "Rainfall Total",
      "graph_type": "line",
      "graph_data": [
        {{"x": 2010, "y1": 850.12}},
        {{"x": 2015, "y1": 1100.56}},
        {{"x": 2020, "y1": 1200.75}}
      ]
    }}
  ]
}}
"""

# --- Combined Agent Function ---
def get_structured_response(text, pdf):
    # WARNING: Hardcoding API keys is a major security risk.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY is not set."

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=api_key)

    agent = create_pandas_dataframe_agent(
        llm,
        pdf,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        prefix=AGENT_PREFIX
    )
    try:
        response = agent.invoke(text)
        return response['output']
    except Exception as e:
        return f"An error occurred in the agent: {e}"

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    graphs: object

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    graphs: object

@app.post("/process-query/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    query = request.query

    districts_found, blocks_found = extract_locations(query)
    filtered_pdf = df
    if districts_found or blocks_found:
        filtered_pdf = df[df['district'].isin(districts_found) | df['block'].isin(blocks_found)]

    if filtered_pdf.empty:
        return {"answer": "No data found for the specified locations.", "graphs": []}

    response_str = get_structured_response(query, filtered_pdf)

    try:
        match = re.search(r"\{.*\}", response_str, re.DOTALL)
        if match:
            clean_str = match.group(0)
            response_data = json.loads(clean_str)
            answer = response_data.get("answer", "No answer text was generated.")
            graphs_field = response_data.get("graphs", [])
            if isinstance(graphs_field, str):
                graphs_json = json.loads(graphs_field)
            else:
                graphs_json = graphs_field
        else:
            raise ValueError("No valid dictionary-like string found in the response.")
    except (ValueError, json.JSONDecodeError, TypeError) as e:
        answer = "Failed to parse the structured response from the AI model."
        safe_response_str = str(response_str).replace("\"", "'")
        graphs_json = {"error": "Invalid format received", "details": str(e), "raw_output": safe_response_str}

    return {"answer": answer, "graphs": graphs_json}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Data Query API. Use the /docs endpoint to see the API documentation."}
