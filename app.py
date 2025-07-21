import streamlit as st
import pandas as pd
import json
import io
import pdfplumber
from ics import Calendar
from openai import AzureOpenAI
import plotly.express as px
import re
from login_page import login

# Page config
st.set_page_config(page_title='Using GenAI in practice', page_icon='', layout = 'wide')

# Display LTP logo
st.image(image= "images/Asset 6.png", caption = "Powered by", width = 100)

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

    st.title("AI Agent Document Analyzer with Chat and Visualization")
    st.markdown(st.session_state["analysis_done"])
    st.markdown(st.session_state["ready_for_chat"])

    model_options = [
        "gpt-4o-mini",
        "gpt-4",
        "claude-3-5-sonnet",
        "claude-3-7-sonnet"
    ]

    # File uploader
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

        # Prompt document analysis
        user_analysis_prompt = st.text_area(
            "Provide instructions for how you want the document to be analyzed (structure, type of topics, depth of extraction, etc.)",
            "Please identify key topics, entities, and propose a structured summary relevant for data visualization."
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
                        {"role": "user", "content": f"{user_analysis_prompt}\n\n{st.session_state["text_data"][:20000]}"}
                    ]
                )
                analysis_result = analysis_response.choices[0].message.content
                st.session_state["analysis_result"] = analysis_result
                st.session_state["analysis_done"] = True

    if st.session_state["analysis_done"]:
        st.subheader("Document Analysis by Agent 1")
        st.markdown(st.session_state["analysis_result"])
        
        # Chat interface
        st.subheader("Chat with AI Agent for Q&A and Visualization")
        # Allow to select model
        selected_model_a2 = st.selectbox("Select the model you want to use to chat:", model_options, index=0, key="model2")
        st.session_state["selected_model_a2"] = selected_model_a2
        if not st.session_state["ready_for_chat"]:
            if st.button("Continue to chat"):
                st.session_state["ready_for_chat"] = True

    # Chat UI
    if st.session_state["ready_for_chat"]:

        user_input = st.chat_input("Ask a question about your document, request visualizations, or insights...")
        
        if user_input and st.session_state["text_data"]:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Agent 2 is generating a response..."):
                # Agent 2: QA and Visualization Generation
                chat_response = client.chat.completions.create(
                    model=st.session_state["selected_model_a2"],
                    messages=[
                        {"role": "system", "content": "You are an AI agent specialized in answering questions about documents and generating clear data visualizations using plotly when requested. If visualization is requested, provide JSON instructions for the plot, using bar, scatter or boxplots. When generating the json, ensure the size of x and y are the same. When generating a plot, and only when generating a plot, mention the word 'plotly'"},
                        {"role": "user", "content": f"Here is the document text for context:\n{st.session_state["text_data"][:20000]}\n\nAnd here is the result of a previous structural analysis:\n{st.session_state["analysis_result"]}"},
                        *[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
                    ]
                )
                agent_reply = chat_response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": agent_reply})

        # Display conversation
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Attempt to parse JSON plot instructions if present
        if st.session_state.messages and 'plotly' in st.session_state.messages[-1]['content'].lower():
            try:
                json_match = re.search(r'\{[\s\S]*\}', st.session_state.messages[-1]['content'])
                if json_match:
                    json_data = json.loads(json_match.group())
                    plot_data = json_data['data'][0]

                    x = plot_data.get('x')
                    y = plot_data.get('y')
                    color = None
                    if 'marker' in plot_data and 'color' in plot_data['marker']:
                        color = plot_data['marker']['color']
                    elif 'color' in plot_data:
                        color = plot_data['color']

                    df = pd.DataFrame({'x': x, 'y': y})
                    plot_type = plot_data.get('type')

                    if plot_type == 'bar':
                        fig = px.bar(df, x='x', y='y', color_discrete_sequence=[color] if color else None)
                    elif plot_type == 'scatter':
                        fig = px.scatter(df, x='x', y='y', color_discrete_sequence=[color] if color else None)
                    elif plot_type == 'box':
                        fig = px.box(df, x='x', y='y', color_discrete_sequence=[color] if color else None)
                    else:
                        fig = px.scatter(df, x='x', y='y', color_discrete_sequence=[color] if color else None)

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No JSON found in the assistant's message.")

            except Exception as e:
                st.warning(f"Tried to generate visualization but encountered an issue: {e}")

        st.info("This AI agent app demonstrates GenAI + agents for document understanding, Q&A, and dynamic visualization generation for your practical AI in Practice sessions.")
    
    if st.button("Reset"):
        for key in ["analysis_done", "ready_for_chat", "messages", "analysis_result"]:
            st.session_state.pop(key, None)
        st.rerun()

#with open('json_plot.json', 'r') as f:
#    data = json.load(f)

#json_data = data