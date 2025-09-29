import json
import requests
import time
import os

# --- CONFIGURATION ---
# URLs for the other local microservices.
# These must match the ports defined in the other app.py files.
ANALYZER_URL = "http://127.0.0.1:5001/analyze"
REASONER_URL = "http://127.0.0.1:5002/reason"

# Paths to the mock data files located in the 'data' subfolder
DATA_FILES = [
    os.path.join("data", "reddit_data.json"),
    os.path.join("data", "youtube_data.json"),
    os.path.join("data", "bluesky_data.json")
]

def send_to_analyzer(item: dict, session: requests.Session):
    """
    Sends data to the multimodal analyzer service.
    """
    analyzer_response = session.post(ANALYZER_URL, json=item, timeout=60)  # Increased timeout for AI models
    analyzer_response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    return analyzer_response.json()

def process_item(item: dict, session: requests.Session):
    """
    Sends a single data item through the analysis and reasoning pipeline.
    """
    item_id = item.get("id", "unknown")
    print(f"Sending item: {item_id}")

    try:
        # --- Step 1: Send data to the Multimodal Analyzer ---
        # The analyzer will run the CLIP model and return enriched data.
        enriched_data = send_to_analyzer(item, session)
        print("  -> Analyzer returned successfully.")

        # --- Step 2: Send enriched data to the Reasoning Agent ---
        # The reasoner will store the final results in the databases.
        reasoner_response = session.post(REASONER_URL, json=enriched_data, timeout=30)
        reasoner_response.raise_for_status()
        print("  -> Reasoner stored successfully.")

    except requests.exceptions.RequestException as e:
        print("\n--- FATAL ERROR ---")
        print(f"Could not connect to one of the services for item {item_id}.")
        print("Please ensure both multimodal_analyzer/app.py and reasoning_agent/app.py are running in separate terminals before starting this script.")
        print(f"Details: {e}")
        return False # Signal that we should stop processing
    except Exception as e:
        print("\n--- UNEXPECTED ERROR ---")
        print(f"An unexpected error occurred while processing item {item_id}: {e}")
        return False

    return True # Signal that processing was successful

def run():
    """
    Main function to read all data files and process them.
    """
    print("--- Starting Local Ingestor Runner ---")
    print(f"Analyzer URL: {ANALYZER_URL}")
    print(f"Reasoner URL: {REASONER_URL}")

    # Use a requests Session for performance (keeps the connection open)
    session = requests.Session()

    for file_path in DATA_FILES:
        print(f"\n--- Processing {file_path} ---")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                if not process_item(item, session):
                    # If processing fails (e.g., a service is down), stop the script.
                    print("Stopping ingestor due to processing error.")
                    return
                
                # Wait for a few seconds to simulate a real-time stream for the demo
                print("-----------------------------------------")
                time.sleep(5)

        except FileNotFoundError:
            print(f"Warning: Data file not found at {file_path}. Please ensure it is in a 'data' subfolder. Skipping.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. Please check for syntax errors. Skipping.")

    print("\n--- Finished processing all data files. ---")

if __name__ == "__main__":
    run()

