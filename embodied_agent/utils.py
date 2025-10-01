def format_message (msg: str) -> dict:
    return {"message": [{"role" : "user" , "content": msg}]}