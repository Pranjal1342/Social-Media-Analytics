import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURATION - FOR LOCAL NEO4J DOCKER ---
# These credentials match the ones in the README's docker run command.
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "fF38IYJ5t11rEUYHSVuzS8aiKojPcGWNQXw0SKW4SG8"

# --- STREAMLIT APP UI ---
st.set_page_config(page_title="SIH Social Media Analyzer", layout="wide")
st.title("🔎 Modernized Social Media Analysis Prototype")

# Auto-refresh the dashboard every 5 seconds to show new results live
st_autorefresh(interval=5000, limit=None, key="dashboard_refresh")

# --- DATABASE CONNECTION AND QUERY ---
@st.cache_resource
def get_neo4j_driver():
    """
    Creates a Neo4j driver instance.
    Using @st.cache_resource prevents reconnecting on every page refresh.
    """
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.error(f"Neo4j Connection Error: {e}. Is the Neo4j Docker container running?")
        return None

def fetch_data(_driver):
    """
    Fetches the latest reports that have been identified as a claim from Neo4j
    and returns them as a Pandas DataFrame.
    """
    if _driver is None:
        return pd.DataFrame()
    
    with _driver.session() as session:
        # This Cypher query gets all posts that have a verdict containing the word 'CLAIM',
        # sorts them by the time they were created, and returns the top 50.
        query = """
        MATCH (p:Post)
        WHERE p.verdict CONTAINS 'CLAIM'
        RETURN 
            p.platform AS Platform, 
            p.author AS Author, 
            p.text AS Text, 
            p.verdict AS Verdict, 
            p.url as URL
        ORDER BY p.timestamp DESC
        LIMIT 50
        """
        result = session.run(query)
        # Convert the Neo4j result into a Pandas DataFrame for easy display
        return pd.DataFrame([dict(record) for record in result])

# --- MAIN DASHBOARD LOGIC ---
if driver := get_neo4j_driver():
    # Fetch the latest data from the database
    df = fetch_data(driver)
    
    st.header("Live Reports: Potential & Misleading Claims Detected")
    
    if df.empty:
        st.info("Waiting for the ingestor to process data and detect new claims...")
    elif 'Platform' in df.columns:
        # Reorder columns for a better presentation on the dashboard
        st.dataframe(
            df[['Platform', 'Verdict', 'Text', 'Author', 'URL']],
            use_container_width=True,
            hide_index=True
        )
    else:
        # Fallback in case the Platform column isn't returned
        st.dataframe(df, use_container_width=True, hide_index=True)

