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
    human_msgs = data.get("human_messages", [])
    ai_msgs = data.get("ai_messages", [])
    tool_msgs = data.get("tool_calls", [])
    final_responses = data.get("final_response", [])

    tool_map = {t['tool_call_id']: t for t in tool_msgs}

    ai_index = 0
    width = 80  # total line width

    def print_centered_header(title: str):
        line = f" {title} "
        print(line.center(width, "="))

    for h_msg in human_msgs:
        print_centered_header("Human Message")
        print(h_msg["content"])
        print()

        while ai_index < len(ai_msgs):
            ai_msg = ai_msgs[ai_index]
            ai_index += 1

            print_centered_header("AI Message")

            if "name" in ai_msg and "args" in ai_msg:
                print("Tool Calls:")
                print(f"  {ai_msg['name']} (Call ID: {ai_msg['id']})")
                print("  Args:")
                for k, v in ai_msg["args"].items():
                    print(f"    {k}: {v}")
                
                t_msg = tool_map.get(ai_msg['id'])
                if t_msg:
                    print()
                    print_centered_header("Tool Message")
                    print(f"Name: {t_msg['tool']}")
                    print(f"Output: {t_msg['output']}")
                    print(f"Tool Call ID: {t_msg['tool_call_id']}")
            print()

    for final in final_responses:
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