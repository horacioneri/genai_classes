import streamlit as st
import pandas as pd
import json
import io
import pdfplumber
from ics import Calendar
from openai import AzureOpenAI
import plotly.express as px
import plotly.graph_objects as go
import re
import ast
from login_page import login

# Page config
st.set_page_config(page_title='Using GenAI in practice', page_icon='', layout = 'wide')

# Display LTP logo
st.image(image= "images/Asset 6.png", caption = "Powered by", width = 100)
st.info("This app demonstrates the use of GenAI and agents for document understanding, Q&A, and dynamic visualization generation in practical sessions. It is a simplified version designed for instructional purposes.")

# Session state initialization
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "ready_for_chat" not in st.session_state:
    st.session_state["ready_for_chat"] = False
if "text_data" not in st.session_state:
    st.session_state["text_data"] = ""
if "analysis_result" not in st.session_state:
    st.session_state["text_data"] = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Log in page
if not st.session_state["logged_in"]:
    login()

else:
    #GPT set up - Using Streamlit secrets at the moment
    client = AzureOpenAI(
        api_key=st.secrets["AZURE_OPENAI_KEY"],
        azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
        api_version=st.secrets["AZURE_OPENAI_API_VERSION"]
    )

    #Function to render plotly json
    def render_plotly_json(json_content):
        try:
            # Attempt parsing robustly
            if isinstance(json_content, str):
                json_content = json_content.strip()
                if json_content.startswith("```json"):
                    json_content = json_content.strip("```json").strip("```").strip()
                json_content = ast.literal_eval(json_content)
            
            if not isinstance(json_content, dict):
                raise ValueError("Provided content is not a dictionary.")

            fig = go.Figure()

            for trace in json_content.get("data", []):
                trace_type = trace.get("type", "")
                mode = trace.get("mode", None)
                marker = trace.get("marker", {})

                # Standard data extractions
                x = trace.get("x", None)
                y = trace.get("y", None)
                z = trace.get("z", None)
                labels = trace.get("labels", None)
                values = trace.get("values", None)

                if trace_type == "bar":
                    fig.add_trace(go.Bar(x=x, y=y, marker=marker))
                elif trace_type == "scatter":
                    fig.add_trace(go.Scatter(x=x, y=y, mode=mode if mode else "markers", marker=marker))
                elif trace_type == "box":
                    fig.add_trace(go.Box(x=x, y=y, marker=marker))
                elif trace_type == "heatmap":
                    fig.add_trace(go.Heatmap(x=x, y=y, z=z, colorscale="Viridis"))
                elif trace_type == "pie":
                    fig.add_trace(go.Pie(labels=labels, values=values))
                elif trace_type == "line":
                    fig.add_trace(go.Scatter(x=x, y=y, mode=mode if mode else "lines", marker=marker))
                else:
                    st.warning(f"Unsupported trace type: {trace_type}")
                    return

            layout = json_content.get("layout", {})
            if layout:
                fig.update_layout(
                    title=layout.get("title", ""),
                    xaxis_title=layout.get("xaxis", {}).get("title", None),
                    yaxis_title=layout.get("yaxis", {}).get("title", None)
                )

            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.warning(f"Tried to generate visualization but encountered an issue: {e}")
            st.json(json_content)

    st.title("Gen AI workflow to analyze document and visualize information")

    model_options = [
        "gpt-4o-mini",
        "gpt-4",
        "claude-3-5-sonnet",
        "claude-3-7-sonnet"
    ]

    # File uploader
    st.header('Document upload', divider='rainbow')
    with st.expander('**Click to upload file**', expanded=True):
        uploaded_file = st.file_uploader("Upload CSV, JSON, PDF, or ICS file:", type=["csv", "json", "pdf", "ics"])

        parsed_data = None
        text_data = ""

        if uploaded_file:
            file_type = uploaded_file.name.split(".")[-1]
            if file_type == "csv":
                parsed_data = pd.read_csv(uploaded_file)
                text_data = parsed_data.to_csv(index=False)
            elif file_type == "json":
                parsed_data = json.load(uploaded_file)
                text_data = json.dumps(parsed_data)
            elif file_type == "pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        text_data += page.extract_text() + "\n"
            elif file_type == "ics":
                c = Calendar(uploaded_file.read().decode())
                for event in c.events:
                    text_data += f"Event: {event.name}, Begin: {event.begin}, End: {event.end}, Description: {event.description}\n"

            st.success(f"File '{uploaded_file.name}' uploaded and parsed.")
            st.session_state["text_data"] = text_data

    if st.session_state["text_data"]:
        st.header('Document analyzer', divider='rainbow')
        with st.expander('**Click to analyze document**', expanded=True):
            # Prompt document analysis
            user_analysis_prompt = st.text_area(
                "Provide instructions for how you want the document to be analyzed (structure, type of topics, depth of extraction, etc.)",
                "Please identify key topics, entities, and propose a structured summary relevant for data visualization.",
                key="user_analysis_prompt"
            )
            # Allow to select model
            selected_model_a1 = st.selectbox("Select the model you want to use to analyze file:", model_options, index=0, key="model1")
            st.session_state["selected_model_a1"] = selected_model_a1

            # Agent 1: Analyze document
            if st.button("Run Analysis"):
                with st.spinner("Analyzing document with Agent 1..."):
                    analysis_response = client.chat.completions.create(
                        model=st.session_state["selected_model_a1"],
                        messages=[
                            {"role": "system", "content": "You are an AI document analysis agent that extracts main topics, structure, and metadata from provided text."},
                            {"role": "user", "content": f"{user_analysis_prompt}\n\n{st.session_state["text_data"][:30000]}"}
                        ]
                    )
                    analysis_result = analysis_response.choices[0].message.content
                    st.session_state["analysis_result"] = analysis_result
                    st.session_state["analysis_done"] = True

    if st.session_state["analysis_done"]:
        st.subheader("Document Analysis")
        with st.expander('**Click to see document analysis**', expanded=True):
            st.markdown(st.session_state["analysis_result"])
        
        # Chat interface
        st.header('Chat with AI about the document', divider='rainbow')
        # Allow to select model
        selected_model_a2 = st.selectbox("Select the model you want to use to chat:", model_options, index=0, key="model2")
        st.session_state["selected_model_a2"] = selected_model_a2

        # Allow to select context
        selected_context = st.selectbox("Select the context you want to give to the model:", ["Original file + Analysis", "Original file only", "Analysis only"], index=0, key="context1")
        st.session_state["selected_context"] = selected_context
        
        if not st.session_state["ready_for_chat"]:
            if st.button("Continue to chat"):
                st.session_state["ready_for_chat"] = True

    # Chat UI
    if st.session_state["ready_for_chat"]:

        user_input = st.chat_input("Ask a question about your document, request visualizations, or insights... The available visualizations are simples bar charts, scatterplots and box plots")
        
        if user_input and st.session_state["text_data"]:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Agent 2 is generating a response..."):
                # Agent 2: QA and Visualization Generation
                if st.session_state["selected_context"] == "Original file only":
                    context = f"Here is the document text for context:\n{st.session_state["text_data"][:30000]}"
                elif st.session_state["selected_context"] == "Analysis only":
                    context = f"Here is the result of the analysis of the document text:\n{st.session_state["analysis_result"]}"
                else:
                    context = f"Here is the document text for context:\n{st.session_state["text_data"][:30000]}\n\nAnd here is the result of a previous structural analysis:\n{st.session_state["analysis_result"]}"

                #old prompt {"role": "system", "content": "You are an AI agent specialized in answering questions about documents and generating clear data visualizations using Plotly when requested. If the user requests a visualization, you must generate JSON instructions for the plot. You can use only bar, scatter, or box plots. When generating the JSON, ensure that the sizes of 'x' and 'y' are the same. Mention the word 'plotly' exactly once in your message, and only when generating a plot. Return only one JSON object using this exact structure:\n\n{\n  \"data\": [{\n    \"type\": \"bar\" | \"scatter\" | \"box\",\n    \"x\": [list of x values],\n    \"y\": [list of y values],\n    \"marker\": {\"color\": \"color_value\"}  // optional\n  }],\n  \"layout\": {\n    \"title\": \"Your plot title\"\n  }\n}\n\nDo not include any extra text before or after the JSON. Use only double quotes around all keys and string values in the JSON. If a plot is not requested, provide only a clear text answer without mentioning 'plotly' and without generating JSON."},
                chat_response = client.chat.completions.create(
                    model=st.session_state["selected_model_a2"],
                    messages=[
                        {"role": "system", "content": """
                            You are an AI agent specialized in answering questions about documents and generating clear data visualizations using Plotly when requested.

                            When generating a visualization:
                            - Always generate **valid JSON with double quotes only** so it can be parsed directly.
                            - Structure:
                            - "data": a list of Plotly trace objects with "type" (e.g., "bar", "scatter", "box", "heatmap", "pie", "line", etc.).
                            - Each trace must include necessary fields like "x", "y", "z", "labels", "values", or "mode" as applicable, ensuring all arrays are of matching lengths where required.
                            - "layout":
                                - "title": Title of the chart.
                                - "xaxis": {"title": "label"} if applicable.
                                - "yaxis": {"title": "label"} if applicable.
                            - If generating a plot, include the word "plotly" somewhere in your response so it can be identified as a visualization.
                            - If visualization is not requested, provide a clear structured textual answer instead.
                            - Ensure the JSON is fully parsable, without trailing commas, and with consistent structure.

                            Only produce the JSON for visualization if explicitly requested or if a visualization would clearly improve understanding of the document.
                            """},
                        {"role": "user", "content": context},
                        *[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
                    ]
                )
                agent_reply = chat_response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": agent_reply})
            
        # Display conversation
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.messages:
            if "plotly" in agent_reply.lower():
                render_plotly_json(st.session_state.messages[-1]['content'])

        # # Attempt to parse JSON plot instructions if present
        # if st.session_state.messages:
        #     last_content = st.session_state.messages[-1]['content']
        #     json_block = None

        #     # Prefer extraction inside ```json ``` blocks
        #     json_match = re.search(r'```json\s*([\s\S]*?)\s*```', last_content)
        #     if json_match:
        #         json_block = json_match.group(1).strip()
        #     else:
        #         # Fallback: find the first { ... } and attempt to parse safely
        #         brace_stack = []
        #         start_idx = None
        #         for idx, char in enumerate(last_content):
        #             if char == '{':
        #                 if not brace_stack:
        #                     start_idx = idx
        #                 brace_stack.append('{')
        #             elif char == '}':
        #                 if brace_stack:
        #                     brace_stack.pop()
        #                     if not brace_stack:
        #                         json_block = last_content[start_idx:idx+1]
        #                         break

        #     if json_block:
        #         try:
        #             json_data = json.loads(json_block)
        #             plot_data = json_data['data'][0]
        #             layout_data = json_data.get('layout', {})

        #             x = plot_data.get('x')
        #             y = plot_data.get('y')
        #             color = None
        #             if 'marker' in plot_data and 'color' in plot_data['marker']:
        #                 color = plot_data['marker']['color']
        #             elif 'color' in plot_data:
        #                 color = plot_data['color']
        #             mode = plot_data.get('mode', None)

        #             df = pd.DataFrame({'x': x, 'y': y})
        #             plot_type = plot_data.get('type', 'scatter')

        #             title = layout_data.get('title', 'Generated Plot')
        #             xaxis_title = layout_data.get('xaxis_title', 'x')
        #             yaxis_title = layout_data.get('yaxis_title', 'y')

        #             if plot_type == 'bar':
        #                 fig = px.bar(df, x='x', y='y', title=title, labels={'x': xaxis_title, 'y': yaxis_title},
        #                      color_discrete_sequence=[color] if color else None)
        #             elif plot_type == 'scatter':
        #                 fig = px.scatter(df, x='x', y='y', title=title, labels={'x': xaxis_title, 'y': yaxis_title},
        #                          color_discrete_sequence=[color] if color else None)
        #                 if mode:
        #                     fig.update_traces(mode=mode)
        #             elif plot_type == 'box':
        #                 fig = px.box(df, x='x', y='y', title=title, labels={'x': xaxis_title, 'y': yaxis_title},
        #                      color_discrete_sequence=[color] if color else None)
        #             elif plot_type == 'heatmap':
        #                 z = plot_data.get('z')
        #                 fig = px.imshow(z, x=x, y=y, color_continuous_scale=color if color else 'Viridis',title=title,
        #                     labels={'x': xaxis_title, 'y': yaxis_title, 'color': 'Value'})
        #             else:
        #                 fig = px.scatter(df, x='x', y='y', title=title, labels={'x': xaxis_title, 'y': yaxis_title},
        #                          color_discrete_sequence=[color] if color else None)

        #             st.plotly_chart(fig, use_container_width=True)

        #         except Exception as e:
        #             st.warning(f"Tried to generate visualization but encountered an issue: {e}")
        #     #else:
        #     #    st.info("No JSON visualization instructions detected in the last assistant message.")

    
    if st.button("Reset"):
        for key in ["analysis_done", "ready_for_chat", "messages", "analysis_result", "user_analysis_prompt"]:
            st.session_state.pop(key, None)
        st.rerun()

#with open('json_plot.json', 'r') as f:
#    data = json.load(f)

#json_data = data
#last_content = '{ "data": [{ "type": "bar", "x": ["Gastroenterology", "Pediatrics", "Dermatology", "Billing", "Radiology", "Orthopedics", "Pharmacy", "Emergency"], "y": [10, 7, 12, 10, 2, 6, 2, 2], "marker": {"color": "blue"} }], "layout": { "title": "Number of Complaints per Medical Department" } }'