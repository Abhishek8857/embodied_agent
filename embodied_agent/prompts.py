# Instructions: Base instructions from the developer, commonly referred to as the system prompt. This may be static or dynamic.

# Tools: What tools the agent has access to. The names and descriptions and arguments of these are just as important as the text in the prompt.

# Structured output: What format the agent should respond in. 
#                    The name and description and arguments of these are just as important as the text in the prompt.

# Session context: We also call this “short term memory” in the docs. 
#                  In the context of a conversation, this is most easily thought of the list of messages that make up the conversation. 
#                  But there can often be other, more structured information that you may want the agent to access or update throughout the session. 
#                  The agent can read and write this context. This context is often put directly into the context that is passed to the LLM.
#                  Examples include: messages, files.

# Long term memory: This is information that should persist across sessions (conversations). Examples include: extracted preferences

# Runtime configuration context: This is context that is not the “state” or “memory” of the agent, but rather configuration for a given agent run. 
#                                This is not modified by the agent, and typically isn’t passed into the LLM, but is used to guide the agent’s behavior or look up other context. 
#                                 Examples include: user ID, DB connections


# https://docs.langchain.com/oss/python/langchain/context-engineering


prompt = """You are an expert mathematician who answers maths questions
        
            
            You should understand the questions and trigger the relevant tools"""

def get_prompts() -> str:
    return prompt

