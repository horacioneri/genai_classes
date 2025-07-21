import streamlit as st
import pandas as pd
import json
import io
import pdfplumber
from ics import Calendar
from openai import AzureOpenAI
import plotly.express as px
from login_page import login

# Page config
st.set_page_config(page_title='Using GenAI in practice', page_icon='', layout = 'wide')

# Display LTP logo
st.image(image= "images/Asset 6.png", caption = "Powered by", width = 100)

# Create log in state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

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

    # Session state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

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

        # Prompt document analysis
        user_analysis_prompt = st.text_area(
            "Provide instructions for how you want the document to be analyzed (structure, type of topics, depth of extraction, etc.)",
            "Please identify key topics, entities, and propose a structured summary relevant for data visualization."
        )
        # Agent 1: Analyze document
        with st.spinner("Analyzing document with Agent 1..."):
            analysis_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI document analysis agent that extracts main topics, structure, and metadata from provided text."},
                    {"role": "user", "content": f"{user_analysis_prompt}\n\n{text_data[:10000]}"}
                ]
            )
            analysis_result = analysis_response.choices[0].message.content
            st.subheader("Document Analysis by Agent 1")
            st.markdown(analysis_result)

    # Chat interface
    st.subheader("Chat with AI Agent for Q&A and Visualization")
    user_input = st.chat_input("Ask a question about your document, request visualizations, or insights...")

    if user_input and text_data:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Agent 2 is generating a response..."):
            # Agent 2: QA and Visualization Generation
            chat_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI agent specialized in answering questions about documents and generating clear data visualizations using plotly when requested. If visualization is requested, provide JSON instructions for the plot."},
                    {"role": "user", "content": f"Here is the document text for context:\n{text_data[:10000]}"},
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
            json_start = st.session_state.messages[-1]['content'].find('{')
            json_data = json.loads(st.session_state.messages[-1]['content'][json_start:])
            plot_data = json_data['data'][0]  # First trace dict

            # Extract x, y, and optional color info
            x = plot_data.get('x')
            y = plot_data.get('y')

            # Sometimes color might be under marker.color or just color
            color = None
            if 'marker' in plot_data and 'color' in plot_data['marker']:
                color = plot_data['marker']['color']
            elif 'color' in plot_data:
                color = plot_data['color']

            # Create DataFrame from x and y
            df = pd.DataFrame({'x': x, 'y': y})

            # Create plotly figure depending on plot type
            if plot_type == 'bar':
                fig = px.bar(df, x='x', y='y', color_discrete_sequence=[color] if color else None)
            elif plot_type == 'scatter':
                fig = px.scatter(df, x='x', y='y', color=color)
            elif plot_type == 'box':
                # For boxplots, typically you plot y vs category x
                fig = px.box(df, x='x', y='y', color=color)
            else:
                fig = px.scatter(df, x='x', y='y', color=color)

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Tried to generate visualization but encountered an issue: {e}")

    st.info("This AI agent app demonstrates GenAI + agents for document understanding, Q&A, and dynamic visualization generation for your practical AI in Practice sessions.")

with open('json_plot.json', 'r') as f:
    data = json.load(f)

json_data = data
df = pd.DataFrame(json_data['data'])
fig = px.scatter(df, x=json_data['x'], y=json_data['y'], color=json_data.get('color'))

