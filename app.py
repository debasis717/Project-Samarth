import streamlit as st
import pandas as pd
import requests  # Use 'requests' library
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import re

# --- THIS IS THE FIX: Removed the failing import ---
# from langchain_core.exceptions import OutputParsingError (REMOVED)

# --- 1. CONFIGURATION ---
# Load secrets from Streamlit's secret manager
try:
    YOUR_API_KEY = st.secrets["DATA_GOV_KEY"]
    YOUR_GEMINI_KEY = st.secrets["GEMINI_KEY"]
except KeyError:
    st.error("ERROR: API keys not found. Please add DATA_GOV_KEY and GEMINI_KEY to your Streamlit secrets.")
    st.stop()

AGRI_RESOURCE_ID = "35be999b-0208-4354-b557-f6ca9a5355de"
CLIMATE_RESOURCE_ID = "8e0bd482-4aba-4d99-9cb9-ff124f6f1c2f"

# --- 2. DATA SOURCING, CLEANING, AND MERGING ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_and_clean_data():
    
    # Add headers to pretend to be a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'Referer': 'https://data.gov.in/'
    }
    
    # --- 1. Fetch Agriculture Data ---
    agri_url = f"https://api.data.gov.in/resource/{AGRI_RESOURCE_ID}?api-key={YOUR_API_KEY}&format=json&limit=500000"
    try:
        response_agri = requests.get(agri_url, headers=headers)
        response_agri.raise_for_status()
        agri_data = response_agri.json()
    except Exception as e:
        st.error(f"Error loading Agriculture data. Check API key. Error: {e}")
        st.stop()

    if 'records' not in agri_data:
        st.error("Failed to load Agriculture Data. Server said:")
        st.json(agri_data)
        st.stop()

    # --- 2. Fetch Climate Data ---
    climate_url = f"https://api.data.gov.in/resource/{CLIMATE_RESOURCE_ID}?api-key={YOUR_API_KEY}&format=json&limit=500000"
    try:
        response_climate = requests.get(climate_url, headers=headers)
        response_climate.raise_for_status()
        climate_data = response_climate.json()
    except Exception as e:
        st.error(f"Error loading Climate data. Check API key. Error: {e}")
        st.stop()

    if 'records' not in climate_data:
        st.error("Failed to load Climate Data. Server said:")
        st.json(climate_data)
        st.stop()
    
    # --- 3. Load and Clean Agri Data ---
    agri_df = pd.DataFrame(agri_data['records'])
    
    # Clean all string columns to be Title Case and stripped of whitespace
    agri_df['state_name'] = agri_df['state_name'].str.strip().str.title()
    agri_df['district_name'] = agri_df['district_name'].str.strip().str.title()
    agri_df['crop'] = agri_df['crop'].str.strip().str.title()
    agri_df['season'] = agri_df['season'].str.strip().str.title()
    agri_df['state_name'] = agri_df['state_name'].str.replace(' & ', ' And ')

    # Convert numeric columns
    agri_df['production_'] = pd.to_numeric(agri_df['production_'], errors='coerce')
    agri_df['area_'] = pd.to_numeric(agri_df['area_'], errors='coerce')
    agri_df['crop_year'] = pd.to_numeric(agri_df['crop_year'], errors='coerce')
    
    agri_df.rename(columns={
        'state_name': 'State',
        'district_name': 'District',
        'crop_year': 'Year',
        'season': 'Season',
        'crop': 'Crop',
        'area_': 'Area',
        'production_': 'Production'
    }, inplace=True)
    
    agri_df = agri_df.dropna(subset=['Production', 'State', 'Year'])

    # --- 4. Load and Clean Climate Data ---
    climate_df = pd.DataFrame(climate_data['records'])
    
    # Clean to Title Case to match agri_df
    climate_df['subdivision'] = climate_df['subdivision'].str.strip().str.title()
    climate_df['subdivision'] = climate_df['subdivision'].str.replace(' & ', ' And ')
    
    climate_df['year'] = pd.to_numeric(climate_df['year'], errors='coerce')
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'annual']
    for col in month_cols:
        climate_df[col] = pd.to_numeric(climate_df[col], errors='coerce')
        
    climate_df.rename(columns={
        'subdivision': 'State',
        'year': 'Year',
        'annual': 'Annual_Rainfall'
    }, inplace=True)
    
    # Select only the columns we need before merging
    climate_df_simplified = climate_df[['State', 'Year', 'Annual_Rainfall']]
    
    # --- 5. THE CRITICAL FIX: Merge DataFrames ---
    st.write("Merging agriculture and climate data...")
    merged_df = pd.merge(
        agri_df,
        climate_df_simplified,
        on=['State', 'Year'],
        how='outer' # 'outer' keeps all rows from both tables
    )
    st.write("Data loaded and merged!")
    
    return merged_df

# --- Load data ---
df = load_and_clean_data()


# --- 3. THE "BRAIN" (LangChain Agent) ---
try:
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash", 
        google_api_key=YOUR_GEMINI_KEY, 
        temperature=0
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,  # Pass the single, merged dataframe
        verbose=True,
        # Update the instructions
        prefix="""You are an agent. You have one single dataframe named 'df'
        that contains all data.
        'State', 'District', 'Year', 'Season', 'Crop', 'Area', 'Production'
        are from the agriculture data.
        'Annual_Rainfall' is from the climate data.
        All string columns (State, District, Crop) are in Title Case.
        
        IMPORTANT: Some rows may only have agriculture data (Annual_Rainfall will be NaN).
        Some rows may only have climate data (Production, Crop, etc. will be NaN).
        This is normal. For example, 'West Uttar Pradesh' only has climate data.
        
        Always state the years for the data you are referencing.""",
        allow_dangerous_code=True,
        handle_parsing_errors=True  # Handle any formatting mistakes by the LLM
    )
except Exception as e:
    st.error(f"Failed to create AI Agent. Check your GEMINI_KEY.")
    st.error(e)
    st.stop()


# --- 4. THE INTERFACE (Streamlit) ---
st.title("Project Samarth: Agri-Climate Q&A 🇮N")

question = st.text_input("Ask a question about agriculture and climate:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Analyzing live government data..."):
            try:
                # Add the parsing config directly to the invoke call
                response = agent.invoke(question, config={"handle_parsing_errors": True}) 
                st.write(response['output'])
                
            except Exception as e:
                # --- THIS IS THE FIX ---
                # We will catch any error and check its text
                error_message = str(e)
                if "Could not parse LLM output: " in error_message:
                    # Extract the text after "Could not parse LLM output: "
                    final_answer = error_message.split("Could not parse LLM output: ", 1)[-1]
                    # Clean up any backticks
                    final_answer = final_answer.strip().strip('`')
                    st.markdown(final_answer) # Use st.markdown to render newlines
                
                # Check for the rate limit error specifically
                elif "ResourceExhausted" in error_message:
                    st.error("You have hit the free API rate limit. Please wait 1 minute and try again.")
                else:
                    # If we can't extract it, just show the error
                    st.error(f"An error occurred while getting the answer: {e}")
    else:
        st.warning("Please enter a question.")

