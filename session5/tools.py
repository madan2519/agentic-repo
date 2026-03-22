from langchain_core.tools import tool

# 1. Setup Tools
@tool
def book_hotel(city: str) -> str:
    """Useful for booking hotel rooms in a specific city."""
    return f"Successfully booked a 5-star hotel in {city}."

@tool
def book_flight(destination: str) -> str:
    """Useful for finding or booking flights to a destination."""
    return f"Found a direct flight to {destination} for $400."