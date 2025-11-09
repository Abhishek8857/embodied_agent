import os
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


def format_message (msg: str) -> dict:
    return {"messages": [{"role" : "user" , "content": msg}]}


def format_response(msg: dict) -> dict:
    data = msg["messages"]
    human_messages: list = []
    ai_messages: list = []
    tool_calls: list = []
    final_response: list = []
    
    for obj in data:
        if isinstance(obj, HumanMessage):
            human_messages.append({"content": obj.content})
        
        if isinstance(obj, AIMessage):
            if obj.content != '':
                final_response.append({"content": obj.content})
            else:
                ai_messages.append(obj.tool_calls[0])
        
        if isinstance(obj, ToolMessage):
            tool_calls.append({"tool": obj.name, "output": obj.content, "tool_call_id": obj.tool_call_id})
    
    
    return {"human_messages": human_messages, "tool_calls": tool_calls, "ai_messages": ai_messages, "final_response": final_response}             


def print_response(data: dict):
    """
    Print only the latest full exchange (Human → AI → Tool(s) → Final AI Response).
    Handles multiple tool calls made during one request.
    Clears the screen before printing for a live-display effect.
    """
    human_msgs = data.get("human_messages", [])
    ai_msgs = data.get("ai_messages", [])
    tool_msgs = data.get("tool_calls", [])
    final_responses = data.get("final_response", [])

    # Clear terminal for live dashboard style output
    os.system("clear")

    width = 80
    def print_centered_header(title: str):
        print(f" {title} ".center(width, "="))

    # Get the latest human message
    h_msg = human_msgs[-1] if human_msgs else None

    # Determine which AI and tool messages belong to this exchange
    last_human_index = len(human_msgs) - 1
    ai_after_human = ai_msgs[last_human_index:] if ai_msgs else []
    final = final_responses[-1] if final_responses else None

    # Create a lookup map for tool call outputs
    tool_map = {t['tool_call_id']: t for t in tool_msgs}

    # === Print Human Message ===
    if h_msg:
        print_centered_header("Human Message")
        print(h_msg["content"])
        print()

    # === Print All AI Messages & Tools ===
    for ai_msg in ai_after_human:
        print_centered_header("AI Message")

        if "name" in ai_msg and "args" in ai_msg:
            print("Tool Calls:")
            print(f"  {ai_msg['name']} (Call ID: {ai_msg['id']})")
            print("  Args:")
            for k, v in ai_msg["args"].items():
                print(f"    {k}: {v}")

            # Find and print corresponding tool message(s)
            t_msg = tool_map.get(ai_msg['id'])
            if t_msg:
                print()
                print_centered_header("Tool Message")
                print(f"Name: {t_msg['tool']}")
                print(f"Output: {t_msg['output']}")
                print(f"Tool Call ID: {t_msg['tool_call_id']}")
        print()

    # === Print Final Response ===
    if final:
        print_centered_header("Final AI Response")
        print(final["content"])
        print()




def get_openai_api_key() -> str: 
    try:
        with open("openai_api_key.config", "r") as f:
            api_key = os.environ["OPENAI_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        raise RuntimeError("Missing openai_api_key.config file")
    
    return api_key


def get_langsmith_api_key() -> str:
    try:
        with open("langsmith_api_key.config", "r") as f:
            api_key = os.environ["LANGSMITH_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        raise RuntimeError("Missing langsmith_api_key.config file")
    
    return api_key