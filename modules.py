import pandas as pd
import json
import re
import os
 
 
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
def process_trace_data(trace_data):
    result = []
    trace_ids = trace_data["trace_rowid"].unique()

    for trace_id in trace_ids:
        trace_df = trace_data[trace_data["trace_rowid"] == trace_id]

        tot_token_trace_prompt = trace_df["cumulative_llm_token_count_prompt"].sum()
        tot_token_trace_completion = trace_df["cumulative_llm_token_count_completion"].sum()
        tot_token_trace = tot_token_trace_prompt + tot_token_trace_completion

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
            "total_token_usage": tot_token_trace,
            "tool_names": [],
            "tool_usage_count": "",
            "llm_models_agent": [],
            "llm_models_tool": [],
            "llm_calls_tool": 0,
            "total_llm_calls": 0,   # will be set below
            "tool_token_usage": 0,
            "error_count": 0,
            "timestamp": trace_time,
            "status_code": status_code,
            "status_message": status_message
        }

        # NEW: running total of LLM calls for this trace
        total_llm_calls = 0 

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
                if trace_info["agent_token_usage"]:
                    trace_info["agent_token_usage"] += ";\n " + f"{agent_name}-{tot_token_agent}"
                else:
                    trace_info["agent_token_usage"] = f"{agent_name}-{tot_token_agent}"

                # Tools under this agent
                agent_tool_df = trace_df[(trace_df["span_kind"] == "TOOL") & (trace_df["parent_id"] == agent_span_id)]
                if not agent_tool_df.empty:
                    agent_tool_call_count = agent_tool_df.shape[0]
                    if trace_info["tool_usage_count"]:
                        trace_info["tool_usage_count"] += ";\n " + f"{agent_name}-{agent_tool_call_count}"
                    else:
                        trace_info["tool_usage_count"] = f"{agent_name}-{agent_tool_call_count}"

                    for _, tool in agent_tool_df.iterrows():
                        tool_name = parseToolName(tool.get("attributes"))
                        trace_info["tool_names"].append(tool_name)

                        tool_span_id = tool.get("span_id")
                        tool_llm_df = trace_df[(trace_df["span_kind"] == "LLM") & (trace_df["parent_id"] == tool_span_id)]
                        if not tool_llm_df.empty:
                            tool_llm_call_count = tool_llm_df.shape[0]
                            trace_info["llm_calls_tool"] += tool_llm_call_count
                            #total_llm_calls += tool_llm_call_count  # NEW

                            tool_llm_model_name, token_prompt, token_completion = getLLMModelName(
                                tool_llm_df["attributes"].iloc[0],
                                tool_llm_df["cumulative_error_count"]
                            )
                            trace_info["llm_models_tool"].append(tool_llm_model_name)

                            tool_token_prompt = tool_llm_df["cumulative_llm_token_count_prompt"].sum()
                            tool_token_completion = tool_llm_df["cumulative_llm_token_count_completion"].sum()
                            trace_info["tool_token_usage"] += tool_token_prompt + tool_token_completion

        # NEW: set the computed total into the row
        trace_info["total_llm_calls"] = int(total_llm_calls)  # NEW

        result.append(trace_info)

    return pd.DataFrame(result)


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
