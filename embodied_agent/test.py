from langchain.messages import HumanMessage, AIMessage, ToolMessage

temp ={
    'messages': [
        HumanMessage(
            content='multiply 3 and 4 then divide by 3 and then mutiply by 7 and then divide by 10',
            additional_kwargs={},
            response_metadata={},
            id='65a41acf-9585-418b-b020-92dbca4ad6c6'
        ),

        AIMessage(
            content='',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {'completion_tokens': 18, 'prompt_tokens': 214, 'total_tokens': 232,
                                'completion_tokens_details': None, 'prompt_tokens_details': None},
                'model_provider': 'openai',
                'model_name': 'openai/gpt-4.1',
                'system_fingerprint': None,
                'id': 'gen-1760003593-oxvzs1hBypZMY7ynWHO3',
                'service_tier': None,
                'finish_reason': 'tool_calls',
                'logprobs': None
            },
            id='lc_run--239df96e-1c2d-43ea-ad03-261a3eac27c0-0',
            tool_calls=[{
                'name': 'multiply',
                'args': {'a': 3, 'b': 4},
                'id': 'call_LPSBd7yWv9Xm4wkDTZXUGX1r',
                'type': 'tool_call'
            }],
            usage_metadata={'input_tokens': 214, 'output_tokens': 18, 'total_tokens': 232,
                            'input_token_details': {}, 'output_token_details': {}}
        ),

        ToolMessage(
            content='12',
            name='multiply',
            id='76897fe8-27fa-4ddb-85d9-7eed5ae76f1d',
            tool_call_id='call_LPSBd7yWv9Xm4wkDTZXUGX1r'
        ),

        AIMessage(
            content='',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {'completion_tokens': 18, 'prompt_tokens': 240, 'total_tokens': 258,
                                'completion_tokens_details': None, 'prompt_tokens_details': None},
                'model_provider': 'openai',
                'model_name': 'openai/gpt-4.1',
                'system_fingerprint': None,
                'id': 'gen-1760003594-S63FKy7cqZdEJgt8J5Xn',
                'service_tier': None,
                'finish_reason': 'tool_calls',
                'logprobs': None
            },
            id='lc_run--1eb66f6d-cd11-4d93-865d-670e8cb9c8b6-0',
            tool_calls=[{
                'name': 'divide',
                'args': {'a': 12, 'b': 3},
                'id': 'call_xLWAROtFpzZqj4OsLfgRLtIB',
                'type': 'tool_call'
            }],
            usage_metadata={'input_tokens': 240, 'output_tokens': 18, 'total_tokens': 258,
                            'input_token_details': {}, 'output_token_details': {}}
        ),

        ToolMessage(
            content='4.0',
            name='divide',
            id='6622ff99-4542-4313-ae03-7e749d04d5e9',
            tool_call_id='call_xLWAROtFpzZqj4OsLfgRLtIB'
        ),

        AIMessage(
            content='',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {'completion_tokens': 18, 'prompt_tokens': 268, 'total_tokens': 286,
                                'completion_tokens_details': None, 'prompt_tokens_details': None},
                'model_provider': 'openai',
                'model_name': 'openai/gpt-4.1',
                'system_fingerprint': None,
                'id': 'gen-1760003595-FVW0dsq0w80JoHgDIU5k',
                'service_tier': None,
                'finish_reason': 'tool_calls',
                'logprobs': None
            },
            id='lc_run--1dcae202-f320-424d-b627-6b715b58398b-0',
            tool_calls=[{
                'name': 'multiply',
                'args': {'a': 4, 'b': 7},
                'id': 'call_lhg2CqAK8pz3r4o68mwzvc9B',
                'type': 'tool_call'
            }],
            usage_metadata={'input_tokens': 268, 'output_tokens': 18, 'total_tokens': 286,
                            'input_token_details': {}, 'output_token_details': {}}
        ),

        ToolMessage(
            content='28',
            name='multiply',
            id='66e0ce7b-2a83-404f-a76d-5f5030718218',
            tool_call_id='call_lhg2CqAK8pz3r4o68mwzvc9B'
        ),

        AIMessage(
            content='',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {'completion_tokens': 18, 'prompt_tokens': 294, 'total_tokens': 312,
                                'completion_tokens_details': None, 'prompt_tokens_details': None},
                'model_provider': 'openai',
                'model_name': 'openai/gpt-4.1',
                'system_fingerprint': None,
                'id': 'gen-1760003596-aiZxdDMDg3yLnsu5QM0M',
                'service_tier': None,
                'finish_reason': 'tool_calls',
                'logprobs': None
            },
            id='lc_run--5d582d80-56ad-41ba-8ed1-33e20a4e6639-0',
            tool_calls=[{
                'name': 'divide',
                'args': {'a': 28, 'b': 10},
                'id': 'call_eCBE9aBIiqrYUbQZPaM1PcWl',
                'type': 'tool_call'
            }],
            usage_metadata={'input_tokens': 294, 'output_tokens': 18, 'total_tokens': 312,
                            'input_token_details': {}, 'output_token_details': {}}
        ),

        ToolMessage(
            content='2.8',
            name='divide',
            id='65d1a526-32b6-4259-858e-5c90b3e00560',
            tool_call_id='call_eCBE9aBIiqrYUbQZPaM1PcWl'
        ),

        AIMessage(
            content=(
                'The answer is 2.8.\n\n'
                'Here are the steps:\n'
                '1. Multiply 3 and 4 to get 12.\n'
                '2. Divide 12 by 3 to get 4.\n'
                '3. Multiply 4 by 7 to get 28.\n'
                '4. Divide 28 by 10 to get 2.8.'
            ),
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {'completion_tokens': 69, 'prompt_tokens': 322, 'total_tokens': 391,
                                'completion_tokens_details': None, 'prompt_tokens_details': None},
                'model_provider': 'openai',
                'model_name': 'openai/gpt-4.1',
                'system_fingerprint': None,
                'id': 'gen-1760003598-Zwu3hfARH1PXFAzvw2UG',
                'service_tier': None,
                'finish_reason': 'stop',
                'logprobs': None
            },
            id='lc_run--843dcf55-a641-459d-8e86-9d60a6770699-0',
            usage_metadata={'input_tokens': 322, 'output_tokens': 69, 'total_tokens': 391,
                            'input_token_details': {}, 'output_token_details': {}}
        )
    ]
}


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

    for h_msg in human_msgs:
        print("=" * 32 + " Human Message " + "=" * 32)
        print(h_msg["content"])
        print()

        while ai_index < len(ai_msgs):
            ai_msg = ai_msgs[ai_index]
            ai_index += 1

            print("=" * 32 + " AI Message " + "=" * 32)
            if "name" in ai_msg and "args" in ai_msg:
                print("Tool Calls:")
                print(f"  {ai_msg['name']} (Call ID: {ai_msg['id']})")
                print("  Args:")
                for k, v in ai_msg["args"].items():
                    print(f"    {k}: {v}")
                
                t_msg = tool_map.get(ai_msg['id'])
                if t_msg:
                    print("\n" + "=" * 32 + " Tool Message " + "=" * 32)
                    print(f"Name: {t_msg['tool']}")
                    print(f"Output: {t_msg['output']}")
                    print(f"Tool Call ID: {t_msg['tool_call_id']}")
            print()

    for final in final_responses:
        print("=" * 32 + " Final AI Response " + "=" * 32)
        print(final["content"])
        print()


print_response(format_response(temp))