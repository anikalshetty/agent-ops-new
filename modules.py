import numpy as np
import pandas as pd
import json
import re
import os
import streamlit as st
 
# Function to extract unique model name
def extract_unique_model_name(attribute):
    # If attribute is a string, we need to load it as a dictionary
    if isinstance(attribute, str):
        parsed_data = json.loads(attribute)  # Assuming the 'attributes' column is a JSON string
    else:
        parsed_data = attribute  # Already parsed if it's already a dict
 
    # Convert to string for regex matching
    changed = json.dumps(parsed_data)
    # Regex pattern to match the model name like gpt-<version>-<suffix>
    pattern = r'gpt[-\w\.]+'  # allows dash, word chars, and dots
    # r'(\'gpt[-\w\.]+\'|"gpt[-\w\.]+")'
    matches = re.findall(pattern, changed, flags=re.IGNORECASE)
    return list(set(matches))
    # Keep only unique matches and return the first one (or adjust if you want more logic)
    #return list(set(matches))[0] if matches else None  # If matches found, return the first unique model name
 
def agentCallsPerModel(df):
    df['model_name']=df['attributes'].apply(extract_unique_model_name)

    # Explode if multiple models found
    model_df = df.explode('model_name')

    # Drop rows with no model name
    model_df = model_df[model_df['model_name'].notnull()]

    # --- Group data by model_name and agent_name ---
    grouped = model_df.groupby(['model_name', 'agent_name']).size().reset_index(name='count')

    return grouped


def checkAndProcessdata(model_df):
    ProjFramework=model_df['span_name'].unique()

    if "Crew Created" in ProjFramework:
    # Step 3: Filter only rows with "Crew Created"
        crew_df = model_df[model_df['span_name'] == "Crew Created"]
        processed_df = process_model_data(crew_df)
        #st.dataframe(processed_df)
        processed_df = processed_df.explode('model_name')
        # Drop rows with no model name
        processed_df = processed_df[processed_df['model_name'].notnull()]
        grouped = processed_df.groupby(['model_name', 'agent_name']).size().reset_index(name='count')
        return grouped
    else:
        return None
 
# --- Updated JSON extractor ---
def extract_agents_from_crew_data(data):
    def parse_if_string(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    extracted_agents = []

    if "crew_id" in data and "crew_tasks" in data:
        crew_agents_raw = data["crew_agents"]
        crew_agents = parse_if_string(crew_agents_raw)

        if isinstance(crew_agents, list):
            for agent in crew_agents:
                agent = parse_if_string(agent)
                if isinstance(agent, dict):
                    model = agent.get("llm", "").strip()
                    role = agent.get("role", "").strip()
                    if model or role:
                        extracted_agents.append({
                            "agent_role": role,
                            "model_name": model
                        })
    return extracted_agents


# --- Main logic: extract agents/models if type is Crew AI ---
def process_model_data(model_df):
    rows = []

    for _, row in model_df.iterrows():
        attr_text = row["attributes"]
        span_name = row["span_name"]
        parent_id=row["parent_id"]

        if "crew_id" in attr_text and span_name == "Crew Created":
            try:
                if isinstance(attr_text, str):
                    attr_json = json.loads(attr_text)
                elif isinstance(attr_text, dict):
                     attr_json = attr_text
                else:
                     attr_json = None
                
                if attr_json:

                    agent_info = extract_agents_from_crew_data(attr_json)
                    for info in agent_info:
                        rows.append({
                            "model_name": info["model_name"],
                            "agent_name": info["agent_role"],
                            "parent_id": parent_id,
                            "Type": "Crew AI"  # Add source type here  
                        })
            except json.JSONDecodeError:
                continue  # Skip invalid JSON

    return pd.DataFrame(rows)

def find_llms_under_agents(spans_df):
    output = []

    # Step 1: Group by trace ID
    for trace_id, group in spans_df.groupby('id'):
        span_lookup = group.set_index('span_id').to_dict(orient='index')

        # Step 2: Filter LLM spans
        llm_spans = group[group['span_type'] == 'LLM']

        for _, llm_row in llm_spans.iterrows():
            current_id = llm_row['parent_id']
            agent_info = None

            # Step 3: Walk upward to find nearest AGENT span
            while current_id:
                parent_span = span_lookup.get(current_id)
                if not parent_span:
                    break
                if parent_span.get('span_type') == 'AGENT':
                    agent_info = parent_span
                    break
                current_id = parent_span.get('parent_id')

            if agent_info:
                output.append({
                    "trace_id": trace_id,
                    "llm_span_id": llm_row['span_id'],
                    "llm_name": llm_row['Name'],
                    "agent_span_id": agent_info['span_id'],
                    "agent_name": agent_info['Name']
                })

    return pd.DataFrame(output)

def filter_by_model_regex(df):
    def match_gpt_model_string(attr):
        try:
            # Convert dict to string if needed
            if isinstance(attr, dict):
                attr = json.dumps(attr)

            if not isinstance(attr, str):
                return False

            
            pattern = r'gpt[-\w\.]+'
            return bool(re.search(pattern, attr))
        except Exception:
            return False

    return df[df['attributes'].apply(match_gpt_model_string)]

def getAgentData(df):
    # Filter only rows with span_kind in ['AGENT', 'LLM', 'TOOL']
    filtered_df = df[df['span_kind'].isin(['AGENT', 'LLM', 'TOOL'])].copy()
    return filtered_df

def getagentLLMToolmapping(df):
    # Build a mapping from span_id to its row for quick lookup
    span_lookup = df.set_index('span_id').to_dict(orient='index')
    
    llm_parent_links = []
    agent_tool_links=[]
    for _, row in df.iterrows():
        trace_id=row['trace_rowid']
        # Extract agent -> LLM mappings
        if row['span_kind'] == 'LLM':
            parent_id = row['parent_id']            
            parent_span = span_lookup.get(parent_id, {})
            if parent_span.get('span_kind') == 'AGENT':
                # Extract agent name and model name from attributes
                agent_attr = parent_span.get('attributes')
                status_code=parent_span.get("status_code")
                timestamp=(parent_span.get('start_time').tz_convert('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
                status_message="-"
                if status_code=="ERROR":
                    status_message=parent_span.get("status_message")
                llm_attr = row.get('attributes')
                llm_error=row.get('cumulative_error_count')
                # Try to parse agent name from agent attributes
                agent_name = parseAgentName(agent_attr)

                # Try to extract model name from LLM attributes
                model_name,tokenCount_prompt,tokenCount_completion= getLLMModelName(llm_attr,llm_error)

                llm_parent_links.append({
                    'parent_name': agent_name,
                    'child_name': model_name,
                    'trace_id' :trace_id,
                    "parent_span_kind" : "AGENT",
                    "child_span_kind" : "LLM",
                    "token_input": tokenCount_prompt,
                    "token_output": tokenCount_completion,
                    "token":tokenCount_prompt+tokenCount_completion,
                    "parent_id": parent_id,
                    "error":llm_error,
                    "status_code":status_code,
                    "status_message":status_message,
                    "timestamp":timestamp
                })
            # Extract LLM -> tool mappings      
            elif  parent_span.get('span_kind') == 'TOOL':
                tool_attr = parent_span.get('attributes') 
                timestamp=(parent_span.get('start_time').tz_convert('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S') 
                llm_attr = row.get('attributes')
                llm_error=row.get('cumulative_error_count')
                # Try to extract model name from LLM attributes
                model_name,tokenCount_prompt,tokenCount_completion= getLLMModelName(llm_attr,llm_error)
                status_code=parent_span.get("status_code")
                status_message="-"
                if status_code=="ERROR":
                    status_message=parent_span.get("status_message")
                # Try to extract tool name from TOOL attributes
                tool_name=parseToolName(tool_attr)
                llm_parent_links.append({
                    'parent_name': agent_name,
                    'child_name': model_name,
                    'trace_id' :trace_id,
                    "parent_span_kind" : "TOOL",
                    "child_span_kind" : "LLM",
                    "token_input": tokenCount_prompt,
                    "token_output": tokenCount_completion,
                    "token":tokenCount_prompt+tokenCount_completion,
                    "parent_id": parent_id,
                    "error":llm_error,
                    "status_code":status_code,
                    "status_message":status_message,
                    "timestamp":timestamp
                })
        # Extract agent -> tool mappings        
        elif row['span_kind'] == 'TOOL':
            parent_id = row['parent_id']
            parent_span = span_lookup.get(parent_id, {})
            if parent_span.get('span_kind') == 'AGENT':
                # Extract agent name and model name from attributes
                agent_attr = parent_span.get('attributes')
                status_code=parent_span.get("status_code")
                timestamp=(parent_span.get('start_time').tz_convert('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
                status_message="-"
                if status_code=="ERROR":
                    status_message=parent_span.get("status_message")
                tool_attr = row.get('attributes')
                tool_error= row.get('cumulative_error_count')

                # Try to parse agent name from agent attributes
                agent_name = parseAgentName(agent_attr)

                # Try to extract tool name from TOOL attributes
                tool_name=parseToolName(tool_attr)

                agent_tool_links.append({
                    'parent_name': agent_name,
                    'child_name': tool_name,
                    'trace_id' :trace_id,
                    "parent_span_kind" : "AGENT",
                    "child_span_kind" : "TOOL",
                    "token_input": 0,
                    "token_output": 0,
                    "token":0,
                    "parent_id":parent_id,
                    "error":tool_error,
                    "status_code":status_code,
                    "status_message":status_message,
                    "timestamp":timestamp
                })        

    # Convert the result into a DataFrame
    llm_parent_links_df = pd.DataFrame(llm_parent_links)
    agent_tool_df= pd.DataFrame(agent_tool_links)
    return llm_parent_links_df,agent_tool_df

def parseAgentName(agent_attr):
    agent_name = None
    if isinstance(agent_attr, dict):
        try:
            raw_value = agent_attr["input"]["value"]
            parsed_input = json.loads(raw_value)
            agent_str = parsed_input.get("agent", "")
            match = re.search(r"role='([^']+)'", agent_str)
            if match:
                agent_name = match.group(1)
        except:
            pass
    return agent_name  

def prepGraphData(processed_df):
    processed_df = processed_df.explode('child_name')
    # Drop rows with no model name
    processed_df = processed_df[processed_df['child_name'].notnull()]
    grouped = (processed_df.groupby(['parent_name', 'child_name', 'error'], as_index=False)
      .size()
      .rename(columns={'size': 'count'})
    )
    return grouped  

def getLLMModelName(llm_attr,error):
    model_name = None
    tokenCount_prompt=0
    tokenCount_completion=0
    if isinstance(llm_attr, dict):
        try:
            model_name = llm_attr["llm"]["model_name"]
            if(error==0):
                tokenCount_prompt= llm_attr["llm"]["token_count"]["prompt"]
                tokenCount_completion= llm_attr["llm"]["token_count"]["completion"]
        except:
            pass
    return model_name ,tokenCount_prompt ,tokenCount_completion  

def parseToolName(tool_attr):
    tool_name = None
    if isinstance(tool_attr, dict):
        try:
            tool_name = tool_attr["tool"]["name"]
        except:
            pass
    return tool_name  

def filterByError(filter_option,df):
    if filter_option == "No Errors":
        filtered_df = df[df['error'] == 0]
    elif filter_option == "Errors":
        filtered_df = df[df['error'] != 0]
    else:
        filtered_df = df.copy()
    return filtered_df  

def format_detailed_tool_latency(tool_latencies):
    """
    tool_latencies: list of (tool_name, latency)
    Returns a formatted string with mean, 90th percentile, and count per tool.
    """
    if not tool_latencies:
        return ""

    summary_parts = []
    # Group latencies by tool name
    tool_latency_dict = {}
    for tool_name, latency in tool_latencies:
        tool_latency_dict.setdefault(tool_name, []).append(latency)

    for tool_name, lat_list in tool_latency_dict.items():
        mean_latency = np.mean(lat_list)
        perc_90 = np.percentile(lat_list, 90)
        count = len(lat_list)

        summary_parts.append(
            f"{tool_name}\n"
            f"mean: {mean_latency:.2f}s\n"
            f"90th Percentile: {perc_90:.2f}s\n"
            f"count: {count}"
        )
    return "\n\n".join(summary_parts)


def process_trace_data(trace_data):
    result = []
    trace_ids = trace_data["trace_rowid"].unique()

    for trace_id in trace_ids:
        trace_df = trace_data[trace_data["trace_rowid"] == trace_id]

        # tot_token_trace_prompt = trace_df["cumulative_llm_token_count_prompt"].sum()
        # tot_token_trace_completion = trace_df["cumulative_llm_token_count_completion"].sum()
        # tot_token_trace = tot_token_trace_prompt + tot_token_trace_completion

        status_chain = trace_df[trace_df['span_kind'] == "CHAIN"].iloc[0]
        status_code = status_chain.get("status_code")
        status_message = status_chain.get("status_message")

        # Time modification from GMT to IST
        trace_time = (trace_df["timestamp"].iloc[0].tz_convert('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')

        trace_info = {
            "trace_id": trace_id,
            "agents": "",
            "No. of agent calls": 0,
            "latency": "",
            "LLM calls": "",
            "LLM": "",
            "agent_token_usage": "",
            "total_token_usage": 0,
            "tool_names": "",
            "tool_usage_count": "",
            "llm_models_agent": [],
            "llm_models_tool": [],
            "llm_calls_tool": 0,
            "total_llm_calls": 0,
            "tool_token_usage": 0,
            "error_count": 0,
            "timestamp": trace_time,
            "status_code": status_code,
            "status_message": status_message,
            # "tool_latency": "",
            "agent_tool_latency": "",
            "agent_data": []
        }

        total_llm_calls = 0
        total_token_usage_sum = 0

        agents = trace_df[trace_df["span_kind"] == "AGENT"]
        for _, agent in agents.iterrows():
            # Iterate each agent
            agent_attr = agent.get("attributes")

            # Get agent name and append it to the trace row
            agent_name = parseAgentName(agent_attr)
            if trace_info["agents"]:
                trace_info["agents"] += ";\n " + agent_name
            else:
                trace_info["agents"] = agent_name
            trace_info["No. of agent calls"] += 1

            # get latency and append
            latency = getAgentRunTime(agent.get("start_time"), agent.get("end_time"))
            if trace_info["latency"]:
                trace_info["latency"] += ";\n " + f"{agent_name} - {latency}s"
            else:
                trace_info["latency"] = f"{agent_name} - {latency}s"

            # Extract agent input and output JSON strings safely
            agent_input_str = None
            agent_output_str = None

            # Try to extract input and output JSON strings if present
            try:
                if isinstance(agent_attr, dict):
                    input_field = agent_attr.get("input", {})
                    if isinstance(input_field, dict):
                        agent_input_str = input_field.get("value")
                    elif isinstance(input_field, str):
                        agent_input_str = input_field

                    output_field = agent_attr.get("output", {})
                    if isinstance(output_field, dict):
                        agent_output_str = output_field.get("value")
                    elif isinstance(output_field, str):
                        agent_output_str = output_field
            except Exception:
                pass

            # Normalize input JSON and parse agent input & output details according to known formats
            parsed_agent_input = None
            parsed_agent_output = None

            # First format: top-level "input" key with JSON string in "value"
            # Second format: inside that JSON string, another JSON string containing keys like "input", "messages" etc.

            # Try parsing input
            if agent_input_str:
                try:
                    parsed_input_outer = json.loads(agent_input_str)
                    # If it has nested "input" key as string, try parse that too (second format)
                    if isinstance(parsed_input_outer, dict) and "input" in parsed_input_outer:
                        try:
                            parsed_agent_input = json.loads(parsed_input_outer["input"])
                        except Exception:
                            parsed_agent_input = parsed_input_outer["input"]
                    else:
                        parsed_agent_input = parsed_input_outer
                except Exception:
                    parsed_agent_input = agent_input_str

            # Try parsing output similarly
            if agent_output_str:
                try:
                    parsed_agent_output = json.loads(agent_output_str)
                except Exception:
                    parsed_agent_output = agent_output_str

            # Append parsed info for this agent to agent_data list
            trace_info["agent_data"].append({
                "agent_name": agent_name,
                "input": parsed_agent_input,
                "output": parsed_agent_output
            })

            agent_span_id = agent.get("span_id")
            agent_llm_df = trace_df[(trace_df["span_kind"] == "LLM") & (trace_df["parent_id"] == agent_span_id)]

            if not agent_llm_df.empty:
                # Get LLM call count per agent and append
                agent_llm_call_count = agent_llm_df.shape[0]
                total_llm_calls += agent_llm_call_count  # NEW

                if trace_info["LLM calls"]:
                    trace_info["LLM calls"] += ";\n " + f"{agent_name}-{agent_llm_call_count}"
                else:
                    trace_info["LLM calls"] = f"{agent_name}-{agent_llm_call_count}"

                # LLM Model name per agent
                agent_llm_model_name, token_prompt, token_completion = getLLMModelName(
                    agent_llm_df["attributes"].iloc[0],
                    agent_llm_df["cumulative_error_count"]
                )
                if trace_info["LLM"]:
                    trace_info["LLM"] += ";\n " + f"{agent_name} - {agent_llm_model_name}"
                else:
                    trace_info["LLM"] = f"{agent_name} - {agent_llm_model_name}"

                agent_token_prompt = agent_llm_df["cumulative_llm_token_count_prompt"].sum()
                agent_token_completion = agent_llm_df["cumulative_llm_token_count_completion"].sum()
                tot_token_agent = agent_token_prompt + agent_token_completion
                
                total_token_usage_sum += tot_token_agent

                if trace_info["agent_token_usage"]:
                    trace_info["agent_token_usage"] += ";\n " + f"{agent_name}-{tot_token_agent}"
                else:
                    trace_info["agent_token_usage"] = f"{agent_name}-{tot_token_agent}"

                # 🛠 Tool processing UNDER agent loop
                agent_tool_df = trace_df[(trace_df["span_kind"] == "TOOL") & (trace_df["parent_id"] == agent_span_id)]
                if not agent_tool_df.empty:
                    agent_tool_call_count = agent_tool_df.shape[0]
                    if trace_info["tool_usage_count"]:
                        trace_info["tool_usage_count"] += ";\n " + f"{agent_name}-{agent_tool_call_count}"
                    else:
                        trace_info["tool_usage_count"] = f"{agent_name}-{agent_tool_call_count}"

                    latency = getAgentRunTime(agent_tool_df["start_time"].min(), agent_tool_df["start_time"].max())
                    if trace_info["latency"]:
                        trace_info["latency"] += ";\n " + f"{agent_name} Tools - {latency}s"
                    else:
                        trace_info["latency"] = f"{agent_name} Tools - {latency}s"

                    tool_names_list = []
                    tool_latencies = []

                    for _, tool in agent_tool_df.iterrows():
                        tool_name = parseToolName(tool.get("attributes"))
                        tool_names_list.append(tool_name)

                        tool_span_id = tool.get("span_id")
                        tool_llm_df = trace_df[(trace_df["span_kind"] == "LLM") & (trace_df["parent_id"] == tool_span_id)]
                        if not tool_llm_df.empty:
                            tool_llm_call_count = tool_llm_df.shape[0]
                            trace_info["llm_calls_tool"] += tool_llm_call_count

                            tool_llm_model_name, token_prompt, token_completion = getLLMModelName(
                                tool_llm_df["attributes"].iloc[0],
                                tool_llm_df["cumulative_error_count"]
                            )
                            trace_info["llm_models_tool"].append(tool_llm_model_name)

                            tool_token_prompt = tool_llm_df["cumulative_llm_token_count_prompt"].sum()
                            tool_token_completion = tool_llm_df["cumulative_llm_token_count_completion"].sum()
                            trace_info["tool_token_usage"] += tool_token_prompt + tool_token_completion

                        tool_latency = getAgentRunTime(tool.get("start_time"), tool.get("end_time"))
                        tool_latencies.append((tool_name, tool_latency))

                if trace_info["tool_names"]:
                    trace_info["tool_names"] += ", " + ", ".join(tool_names_list)
                else:
                    trace_info["tool_names"] = ", ".join(tool_names_list)

                agent_tool_latency_str = format_detailed_tool_latency(tool_latencies)
                if trace_info["agent_tool_latency"]:
                    trace_info["agent_tool_latency"] += f"\n\n{agent_name}:\n{agent_tool_latency_str}"
                else:
                    trace_info["agent_tool_latency"] = f"{agent_name}:\n{agent_tool_latency_str}"

        # # Overall tool latency summary for all tools in the trace
        # overall_tools = []
        # overall_tool_latencies = []
        # all_tool_spans = trace_df[trace_df["span_kind"] == "TOOL"]
        # for _, tool in all_tool_spans.iterrows():
        #     tool_name = parseToolName(tool.get("attributes"))
        #     if tool_name:
        #         overall_tools.append(tool_name)
        #         tool_latency = getAgentRunTime(tool.get("start_time"), tool.get("end_time"))
        #         overall_tool_latencies.append((tool_name, tool_latency))

        # if overall_tool_latencies:
        #     trace_info["tool_latency"] = format_latency_summary(overall_tool_latencies)
        #     trace_info["tool_names"] = ", ".join(overall_tools)

        trace_info["total_llm_calls"] = int(total_llm_calls)
        trace_info["total_token_usage"] = int(total_token_usage_sum)
        trace_info["agent_data"] = json.dumps(trace_info["agent_data"], indent=2)

        result.append(trace_info)

    return pd.DataFrame(result)

def extract_agent_data(agent_data_raw):
    agent_data_map = {}

    if not agent_data_raw:
        return agent_data_map

    try:
        if isinstance(agent_data_raw, str):
            agent_list = json.loads(agent_data_raw)
        else:
            agent_list = agent_data_raw

        for agent_obj in agent_list:
            name = agent_obj.get("agent_name", "Unknown Agent")
            input_raw = agent_obj.get("input", {})
            tools_raw = input_raw.get("tools", [])

            # 🛠 Extract tools
            tools = []
            for tool_str in tools_raw:
                if "name='" in tool_str:
                    try:
                        tool_name = tool_str.split("name='")[1].split("'")[0]
                        tools.append(tool_name)
                    except IndexError:
                        pass

            # Extract input summary (goal)
            input_goal = ""
            if isinstance(input_raw, str):
                match = re.search(r"goal='(.*?)'", input_raw)
                if match:
                    input_goal = match.group(1)
            elif isinstance(input_raw, dict):
                # Sometimes 'agent' field contains the long config string
                agent_blob = input_raw.get("agent", "")
                if agent_blob:
                    match = re.search(r"goal='(.*?)'", agent_blob)
                    if match:
                        input_goal = match.group(1)
                else:
                    input_goal = input_raw.get("goal", "")

            # output_summary = agent_obj.get("output", {}).get("description", "")
            output_summary = agent_obj.get("output", {}).get("raw", "")

            agent_data_map[name] = {
                "tools": tools,
                "input_summary": input_goal,
                "output_summary": output_summary,
            }

    except Exception as e:
        st.error(f" Failed to parse agent_data: {e}")

    return agent_data_map

def parse_agent_tool_latency(latency_str: str):
    agents_data = {}

    # Split by agent blocks (e.g., "AgentName:\n...")
    agent_blocks = re.split(r'([A-Za-z0-9_ ]+):\n', latency_str)
    # agent_blocks will look like: ['', 'Travel Planning Expert', 'Weather Forecast Tool\nmean: 0.27s...', 'Trip Justifier', 'Weather Forecast Tool\n...']

    for i in range(1, len(agent_blocks), 2):
        agent_name = agent_blocks[i].strip()
        block = agent_blocks[i+1].strip()

        tools = {}
        # Split by double newlines to separate tool sections
        tool_blocks = re.split(r'\n\s*\n', block)
        for tb in tool_blocks:
            lines = tb.strip().splitlines()
            if not lines:
                continue
            tool_name = lines[0].strip()

            # Extract metrics
            mean_val = None
            perc_90 = None
            count_val = None
            for line in lines[1:]:
                if line.lower().startswith("mean:"):
                    mean_val = line.split(":", 1)[1].strip()
                elif "90th" in line:
                    perc_90 = line.split(":", 1)[1].strip()
                elif line.lower().startswith("count:"):
                    count_val = line.split(":", 1)[1].strip()

            tools[tool_name] = {
                "mean": mean_val,
                "p90": perc_90,
                "count": count_val,
            }

        agents_data[agent_name] = tools

    return agents_data

def getAgentRunTime(start_time,end_time):
    latency = round((end_time - start_time).total_seconds(),2) # In seconds    
    return latency

def calcCost(df):
    # Load the config file
    os.path.abspath(__file__)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    with open('config.json', 'r') as file:
        config = json.load(file)

    # Define a function to calculate cost for each row
    def calculate_row_cost(row):
        model_info = config['models'].get(row['llm_model'])
        
        if model_info:
            # Calculate total cost for this row
            total_cost = (row['token_input'] * model_info['prompt_cost_per_token']/1000000) + (row['token_output'] * model_info['completion_cost_per_token']/1000000)
            return total_cost
        else:
            return 0
    
    # Apply the calculate_row_cost function to each row and store the result in 'total_cost'
    df['total_cost(in dollars)'] = df.apply(calculate_row_cost, axis=1)
    
    return df
