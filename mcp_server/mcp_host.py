from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
import json
import subprocess
import time
import os

########################################
# DEBUG MODE used to debug tool call issues
########################################
DEBUG = False


########################################
# AUTO-START THE MCP SERVER
########################################

print("Starting MCP server...")

server_proc = subprocess.Popen(
    ["python3", "mcp_server.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=os.path.dirname(os.path.abspath(__file__)),
)
#to allow the server to start up
time.sleep(2)
print("MCP Server started.")


########################################
# CONSTANTS
########################################
MCP_SERVER = "http://127.0.0.1:8000"


SYSTEM_INSTRUCTIONS = """
You are an HPC assistant.

You MUST follow these rules:

TOOL CALL RULES:
- If the user asks anything requiring HPC information (nodes, GPUs, running code), you MUST call one of these tools:
    1. list_nodes → {"tool": "list_nodes", "args": {}}
    2. best_node → {"tool": "best_node", "args": {}}
    3. run_on_best_node → {"tool": "run_on_best_node", "args": {"command": ..., "workdir": ...}}

- If the user says anything like:
    "grab a compute node",
    "grab a node",
    "get a compute node",
    "connect to a GPU node",
    "use a compute node",
    "open a GPU node",
    "connect me to a GPU",
  you MUST call exactly:
    {"tool": "run_on_best_node",
     "args": {"command": "nvidia-smi", "workdir": "/home/joh9"}}

- Output ONLY the raw JSON dictionary. No surrounding text. No markdown. No backticks.
- NEVER invent tool names.
- NEVER wrap JSON in ``` fences.

AFTER TOOL EXECUTION:
- When you receive tool results, you MUST NOT call another tool.
- Respond with ONE to THREE short plain English sentences.
- You MUST explicitly mention the actual node names or GPU names returned in the result.
- No markdown, no code blocks, no lists.

Stay strictly within these rules.
"""


########################################
# LOAD GEMMA 2B-IT
########################################

# try gemma3 179M vv
print("Loading Gemma 2B IT...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.float32, # try lower (16, 4) (float)
    device_map="cpu"
)


########################################
# GEMMA CHAT TEMPLATE (FIXED)
########################################

# formats based on chat templates
def format_chat(messages):
    """
    Converts a list of:
       {"role": "user" | "assistant", "content": "..."}
    Into Gemma's required chat template.
    """
    text = ""

    for m in messages:
        role = m["role"]
        content = m["content"]

        # Gemma requires assistant→"model"
        if role == "assistant":
            role = "model"

        text += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"

    # Always add a generation prompt
    text += "<start_of_turn>model\n"
    return text

# builds the prompt for Gemma and sends it to the LLM
def ask_gemma(user_input, is_observation=False):
    # Build combined message
    if not is_observation:
        # Normal user query → give system instructions + user question
        combined = SYSTEM_INSTRUCTIONS + "\n\nUSER QUERY:\n" + user_input
        force_json_prefix = "{"
    else:
        # Observation mode → tool result, natural language response expected
        combined = "TOOL OBSERVATION:\n" + user_input
        force_json_prefix = ""   # DO NOT force JSON during tool result

    # Build prompt for Gemma
    prompt = (
        f"<start_of_turn>user\n{combined}<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{force_json_prefix}"
    )

    if DEBUG:
        print("\n===== DEBUG: PROMPT SENT TO GEMMA =====")
        print(prompt)
        print("========================================\n")

    # Run model
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        eos_token_id=tokenizer.eos_token_id
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if DEBUG:
        print("===== DEBUG: RAW MODEL OUTPUT =====")
        print(raw)
        print("===================================\n")

    # Extract only the last model turn
    if "<start_of_turn>model" in raw:
        cleaned = raw.split("<start_of_turn>model")[-1]
    else:
        cleaned = raw

    cleaned = cleaned.replace("<end_of_turn>", "").replace("<eos>", "").strip()

    # If we are expecting JSON, enforce leading brace
    if not is_observation:
        if not cleaned.startswith("{"):
            cleaned = "{" + cleaned

    if DEBUG:
        print("===== DEBUG: CLEANED MODEL OUTPUT =====")
        print(cleaned)
        print("=======================================\n")

    return cleaned




########################################
# TOOL PARSING (ROBUST)
########################################
# checks if it's a valid tool call and if its one of the allowed tools.
def is_tool_call(text):
    cleaned = (
        text.replace("```json", "")
            .replace("```", "")
            .replace("<eos>", "")
            .replace("None", "null")
            .strip()
    )

    if DEBUG:
        print("===== DEBUG: CLEANED TEXT FOR JSON PARSE =====")
        print(cleaned)
        print("===============================================\n")

    try:
        data = json.loads(cleaned)

        if DEBUG:
            print("===== DEBUG: JSON PARSE SUCCESS =====")
            print(data)
            print("======================================\n")

        if "tool" in data and "args" in data:
            if data["args"] is None:
                data["args"] = {}
            return data

        return None

    except Exception as e:
        if DEBUG:
            print("===== DEBUG: JSON PARSE FAILED =====")
            print(e)
            print("====================================\n")
        return None



########################################
# TOOL EXECUTION
########################################
# posts get requests to the mcp server end points
def execute_tool(tool_name, args):
    if tool_name == "list_nodes":
        return requests.get(f"{MCP_SERVER}/list_nodes").json()

    elif tool_name == "best_node":
        return requests.get(f"{MCP_SERVER}/best_node").json()

    elif tool_name == "run_on_best_node":
        return requests.post(f"{MCP_SERVER}/run_on_best_node", json=args).json()

    return {"error": f"Unknown tool: {tool_name}"}



########################################
# MAIN LOOP
########################################

if __name__ == "__main__":
    print("Gemma HPC Agent Ready.\nType a message.\n")

    VALID_TOOLS = {"list_nodes", "best_node", "run_on_best_node"}

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        # ASK GEMMA
        model_output = ask_gemma(user_input)
        print("Gemma:", model_output)

        # TOOL DETECTION
        tool_call = is_tool_call(model_output)

        if tool_call:
            tool_name = tool_call["tool"]
            args = tool_call["args"] or {}

            # Validate tool
            if tool_name not in VALID_TOOLS:
                print(f"\n!!! INVALID TOOL '{tool_name}' — OVERRIDING TO list_nodes !!!\n")
                tool_name = "list_nodes"
                args = {}

            print(f"\n--- Executing tool: {tool_name} ---")
            tool_result = execute_tool(tool_name, args)
            print("Tool result:", tool_result)

            # 🚀 SPECIAL CASE: If this was a compute-node grab, open SSH and exit host
            if tool_name == "run_on_best_node":
                node = tool_result.get("node")
                if node:
                    print(f"\nOpening interactive SSH session to {node}...\n")
                    # Replace this Python process with ssh <node>
                    os.execlp("ssh", "ssh", node)
                else:
                    print("\nERROR: Tool did not return a 'node' key.\n")

            # FOLLOW-UP FOR NORMAL TOOL RESULTS
            followup = (
                "Tool result: " + json.dumps(tool_result) + ". "
                "Write ONE short plain English sentence. "
                "You MUST quote the node name exactly as provided in the tool result. "
                "For example, if the result contains \"gpu1\", you MUST include \"gpu1\" in your sentence. "
                "If you do not repeat the exact node name, your answer is wrong. "
                "No markdown, no lists, no code blocks."
            )

            final_answer = ask_gemma(followup, is_observation=True)
            print("\nGemma (final):", final_answer)

    print("Shutting down MCP server...")
    server_proc.terminate()
