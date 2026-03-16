import os
import json
import ast  # Add this for safe string-to-dict conversion
from typing import Optional
import requests
from langchain_core.tools import tool

@tool
def latest_exchange_rates(base: str = "USD", symbols: Optional[str] = None) -> str:
    """Return latest exchange rates from CurrencyBeacon."""
    
    # --- NEW: Cleanup Logic ---
    # If the LLM sends "{'base': 'USD'}O", we extract just the 'USD'
    clean_base = base
    if "{" in base:
        try:
            # Try to parse the string if the model sent a dict-string
            # We strip any stray characters like 'O' from the end
            potential_dict = ast.literal_eval(base.strip().rstrip('O'))
            if isinstance(potential_dict, dict):
                clean_base = potential_dict.get("base", "USD")
                symbols = potential_dict.get("symbols", symbols)
        except:
            # Fallback: manually grab the first 3-letter uppercase code found
            import re
            match = re.search(r'[A-Z]{3}', base)
            clean_base = match.group(0) if match else "USD"

    api_key = os.environ.get("CURRENCY_BEACON_API_KEY")
    if not api_key:
        return "Error: API Key not found."

    # Use the cleaned parameters
    params = {"api_key": api_key, "base": clean_base}
    if symbols:
        # Ensure symbols is a clean string if it came from the dict-string
        params["symbols"] = str(symbols).strip().rstrip('O').replace("{", "").replace("}", "")

    url = "https://api.currencybeacon.com/v1/latest"
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        # If rates are empty, tell the agent why
        print(data)
        if not data.get("rates"):
            return f"No rates found for base '{clean_base}'. Ensure symbols are correct (e.g., 'INR')."
            
        rates = data.get("rates", {})
        return json.dumps({
            "base": clean_base,
            "rates": rates
        })

    except Exception as e:
        return f"Request failed: {e}"

# 2. Add the custom tool to the list
tools = [latest_exchange_rates]