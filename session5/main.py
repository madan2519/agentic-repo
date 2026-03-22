import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

# 1. Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 1. Setup Tools
@tool
def book_hotel(city: str) -> str:
    """Useful for booking hotel rooms in a specific city."""
    return f"Successfully booked a 5-star hotel in {city}."

@tool
def book_flight(destination: str) -> str:
    """Useful for finding or booking flights to a destination."""
    return f"Found a direct flight to {destination} for $400."

# 3. Create specialized worker agents
# We give them clear names so the supervisor knows who is who
hotel_agent = create_agent(
    model=llm,
    tools=[book_hotel],
    system_prompt="You are a hotel specialist. Focus only on finding and booking accommodations.",
    name="hotel_specialist",
)

flight_agent = create_agent(
    model=llm,
    tools=[book_flight],
    system_prompt="You are a flight specialist. Focus only on airline bookings and travel routes.",
    name="flight_specialist",
)

# 4. Create the Supervisor Workflow
# This replaces your StateGraph and Router logic entirely
supervisor = create_supervisor(
    [hotel_agent, flight_agent],
    model=llm,
    prompt=(
        "You are a travel supervisor. Based on the user's request, delegate work "
        "to either the hotel_specialist or the flight_specialist. "
        "If a user wants to 'stay' or needs 'accommodation', use the hotel_specialist. "
        "If they want to 'fly' or 'travel', use the flight_specialist. "

        "IMPORTANT:\n"
        "- After the specialist completes their task, read their final answer.\n"
        "- Return the specialist's final answer verbatim to the user.\n"
        "- Only say FINISH if there is nothing useful to return."
    )
)

# 5. Compile the app for compilation
app = supervisor.compile()

from langchain_core.messages import HumanMessage

def run_travel_query(query: str):
    print(f"\n--- Processing Query: {query} ---")

    for event in app.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="updates"
    ):
        # Each event may contain a different payload schema; try to find messages
        messages = event.get("messages") or event.get("outputs") or event.get("data") or []

        # If we still don't have a list, show the raw event once for debugging
        if not isinstance(messages, list):
            print("[DEBUG] Unexpected event structure:", event)
            continue

        for msg in messages:
            # ✅ Print every AI message returned by any LLM
            if getattr(msg, "type", None) == "ai":
                msg.pretty_print()

if __name__ == "__main__":
    # Test with natural language (no hardcoded 'hotel' keyword needed)
    run_travel_query("Find me a hotel in Paris for next weekend.")
    # run_travel_query("I need a nice place to stay in Rome and a way to get there from London.")