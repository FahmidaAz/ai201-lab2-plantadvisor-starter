import json
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, MAX_TOOL_ROUNDS
from tools import lookup_plant, get_seasonal_conditions

_client = Groq(api_key=GROQ_API_KEY)

# ──────────────────────────────────────────────
# Tool definitions
#
# These are the schemas that tell the LLM what tools are available and how to
# call them. The LLM reads these descriptions and decides when (and how) to use
# each tool. They're already complete — your job is to implement the tool
# functions in tools.py and the agent loop below.
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_plant",
            "description": (
                "Look up care information for a specific houseplant by name. "
                "Returns detailed watering, light, humidity, and temperature requirements. "
                "Use this whenever the user asks about a specific plant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_name": {
                        "type": "string",
                        "description": "The plant name to look up. Can be a common name, scientific name, or nickname (e.g., 'pothos', 'devil's ivy', 'Monstera deliciosa').",
                    }
                },
                "required": ["plant_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonal_conditions",
            "description": (
                "Get seasonal care adjustments for houseplants. "
                "Returns guidance on watering, fertilizing, light, and pests for the current or specified season. "
                "Use this when a user asks a season-specific question, or to complement plant care advice with seasonal context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {
                        "type": "string",
                        "description": "The season to get care conditions for. If omitted, the current season is detected automatically.",
                        "enum": ["spring", "summer", "fall", "winter"],
                    }
                },
                "required": [],
            },
        },
    },
]

# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable and friendly plant care advisor. "
    "Help users care for their houseplants by looking up specific plant information "
    "and current seasonal conditions using your available tools.\n\n"
    "Always use your tools to look up plant-specific information before answering — "
    "don't rely on your general knowledge alone. If a plant isn't in your database, "
    "say so clearly and offer general guidance based on what the user describes.\n\n"
    "Keep your advice practical and specific. Cite the source of your information "
    "when you have it (e.g., 'According to the care data for your monstera...')."
    "When lookup_plant returns found: False, do not invent specific care data — "
"acknowledge the gap clearly, then offer general guidance for the plant type "
"and suggest the user consult a dedicated plant care resource."
)

# ──────────────────────────────────────────────
# Tool dispatch
#
# This is already complete. It routes tool calls from the LLM to the actual
# Python functions in tools.py, and returns results as JSON strings (which is
# what the Groq API expects for tool results).
# ──────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """Route a tool call to the correct function and return the result as a JSON string."""
    # Some models send arguments as JSON "null" for no-argument tools, which
    # json.loads() turns into None — normalize so .get() below is always safe.
    if not isinstance(tool_args, dict):
        tool_args = {}
    print(f"  → Tool call: {tool_name}({tool_args})")
    if tool_name == "lookup_plant":
        result = lookup_plant(tool_args["plant_name"])
    elif tool_name == "get_seasonal_conditions":
        result = get_seasonal_conditions(tool_args.get("season"))
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    print(f"  ← Result: {json.dumps(result)[:120]}{'...' if len(json.dumps(result)) > 120 else ''}")
    return json.dumps(result)


# ──────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────

def run_agent(user_message: str, history: list) -> str:
    # 1. Build messages: system prompt + history + new user message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    # 2. Agent loop — runs until no tool calls or MAX_TOOL_ROUNDS hit
    for _ in range(MAX_TOOL_ROUNDS):
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )
        assistant_message = response.choices[0].message

        # No tool calls = final answer
        if not assistant_message.tool_calls:
            return assistant_message.content or ""

        # Append assistant message FIRST (tool results reference its IDs)
        messages.append(assistant_message)

        # Execute each tool and append its result
        for tool_call in assistant_message.tool_calls:
            tool_args = json.loads(tool_call.function.arguments)
            result = dispatch_tool(tool_call.function.name, tool_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # MAX_TOOL_ROUNDS reached — get a final text response without tools
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content or ""