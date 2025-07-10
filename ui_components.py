import os
import streamlit as st
from config import (
    MODEL_OPTIONS, DEFAULT_MODEL, DEFAULT_TEMPERATURE, 
    SAMPLE_QUESTIONS, QUICK_ACTIONS, SECURITY_WARNING
)
from utils import (
    get_data_summary, clear_conversation, validate_csv, validate_api_key
)
from agent import initialize_agent_if_needed


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("Configuration")
        
        _render_security_notice()
        api_key_to_use = _render_api_key_section()
        model, temperature = _render_model_settings()
        uploaded_file = _render_file_upload()
        
        if st.button("Clear Conversation"):
            clear_conversation()
    
    return {
        'api_key': api_key_to_use,
        'model': model,
        'temperature': temperature,
        'uploaded_file': uploaded_file
    }


def _render_security_notice():
    """Render security warning in sidebar."""
    with st.expander("Security Notice", expanded=False):
        st.warning(SECURITY_WARNING)


def _render_api_key_section():
    """Render API key configuration section."""
    st.subheader("API Key Configuration")
    
    # Try to get API key from environment
    try:
        groq_api_key = os.environ['GROQ_API_KEY']
    except KeyError:
        groq_api_key = None
    
    if groq_api_key:
        st.success("API Key loaded from environment (.env file)")
        st.text_input("API Key (from .env):", value="gsk_***", disabled=True)
        api_key_to_use = groq_api_key
    else:
        st.info("No API key found in .env file. Please enter manually.")
        api_key_input = st.text_input("Enter your Groq API Key:", type="password")
        api_key_to_use = api_key_input
    
    if api_key_to_use:
        is_valid, message = validate_api_key(api_key_to_use)
        if is_valid:
            st.success(message)
        else:
            st.error(message)
    
    return api_key_to_use


def _render_model_settings():
    """Render model configuration section."""
    st.subheader("Model Settings")
    model = st.selectbox("Select Model:", MODEL_OPTIONS, index=0)
    temperature = st.slider("Temperature (Creativity):", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1)
    return model, temperature


def _render_file_upload():
    """Render file upload section."""
    st.header("File Upload")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload a CSV file for the agent to analyze."
    )
    
    if uploaded_file:
        is_valid, message = validate_csv(uploaded_file)
        if is_valid:
            st.success(message)
            uploaded_file.seek(0)
            st.session_state.uploaded_file_content = uploaded_file.read()
            st.session_state.uploaded_file_name = uploaded_file.name
        else:
            st.error(message)
            uploaded_file = None
            st.session_state.uploaded_file_content = None
    
    return uploaded_file


def render_main_header():
    """Render the main application header."""
    st.header("AI Agent for Data Analysis & Visualization")
    st.write("Upload your CSV file and ask questions in plain English. Powered by Llama 3 via Groq API.")


def render_api_key_info():
    """Render API key setup instructions."""
    st.info("Please configure your Groq API key to begin.")
    st.markdown("""
    ### How to set up your API key:
    **Method 1: Using .env file (Recommended)**
    1. Create a `.env` file in your project directory
    2. Add your API key: `GROQ_API_KEY=your_api_key_here`
    3. Restart the application
    
    **Method 2: Manual entry**
    1. Visit [Groq Console](https://console.groq.com)
    2. Sign up or log in
    3. Generate an API key
    4. Enter it in the sidebar
    """)


def render_sample_questions():
    """Render sample questions section."""
    st.subheader("Sample Questions You Can Ask:")
    col1, col2 = st.columns(2)
    with col1:
        for question in SAMPLE_QUESTIONS[:4]:
            st.markdown(f"- {question}")
    with col2:
        for question in SAMPLE_QUESTIONS[4:]:
            st.markdown(f"- {question}")


def render_data_overview(df):
    """Render data overview section with metrics and sample data."""
    st.subheader("Dataset Overview")
    _display_data_info(df)
    
    with st.expander("View Data Sample", expanded=False):
        st.dataframe(df.head(10))
    
    with st.expander("Data Types & Info", expanded=False):
        _render_data_types_info(df)


def _display_data_info(df):
    """Display data summary metrics."""
    summary = get_data_summary(df)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{summary['rows']:,}")
        st.metric("Missing Values", f"{summary['missing_values']:,}")
    with col2:
        st.metric("Total Columns", summary['columns'])
        st.metric("Duplicate Rows", f"{summary['duplicate_rows']:,}")
    with col3:
        st.metric("Numeric Columns", summary['numeric_columns'])
        st.metric("Memory Usage", f"{summary['memory_usage'] / 1024:.1f} KB")
    with col4:
        st.metric("Text Columns", summary['categorical_columns'])


def _render_data_types_info(df):
    """Render data types and missing values information."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Types")
        data_types_df = df.dtypes.to_frame('Type')
        data_types_df['Type'] = data_types_df['Type'].astype(str)
        st.dataframe(data_types_df)
    
    with col2:
        st.subheader("Missing Values")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            st.dataframe(missing_data.to_frame('Missing Count'))
        else:
            st.success("No missing values found!")


def render_quick_actions():
    """Render quick action buttons."""
    st.subheader("Quick Actions")
    cols = st.columns(len(QUICK_ACTIONS))
    
    for i, action in enumerate(QUICK_ACTIONS):
        with cols[i]:
            if st.button(action["label"], key=f"quick_action_{i}"):
                st.session_state.quick_query = action["query"]
                st.rerun()


def render_chat_interface(api_key, model, temperature, df):
    """Render the main chat interface."""
    st.subheader("Chat with Your Data")
    
    if not initialize_agent_if_needed(api_key, model, temperature, df):
        return
    
    # Display conversation history
    _render_conversation_history()
    
    # Handle quick query
    if st.session_state.quick_query:
        _handle_user_query(st.session_state.quick_query, df)
        st.session_state.quick_query = None
    
    # Chat input
    if user_query := st.chat_input("Ask a question about your data..."):
        _handle_user_query(user_query, df)


def _render_conversation_history():
    """Render the conversation history."""
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "plot_path" in message:
                if message["plot_path"] and os.path.exists(message["plot_path"]):
                    st.image(message["plot_path"])


def _handle_user_query(user_query, df):
    """Handle a user query and display results."""
    st.session_state.history.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data..."):
            df_info = {'shape': df.shape, 'columns': list(df.columns)}
            response = st.session_state.agent.query(user_query, df_info)
            
            if response['success']:
                st.markdown(response['output'])
            else:
                st.error(response['output'])
            
            # Display plot if it exists
            if response['plot_path'] and os.path.exists(response['plot_path']):
                st.image(response['plot_path'])
                st.success("✅ Visualization generated successfully!")
            
            # Store in history
            history_entry = {"role": "assistant", "content": response['output']}
            if response['plot_path']:
                history_entry["plot_path"] = response['plot_path']
            st.session_state.history.append(history_entry)


def render_error_message(error_msg):
    """Render error message with troubleshooting tips."""
    st.error(f"Failed to process the CSV file. Please ensure it is valid. Error: {error_msg}")
    st.markdown("""
    **Common issues:**
    - File encoding problems (try UTF-8, Latin-1, or Cp1252)
    - Incorrect delimiter (e.g., semicolon instead of comma)
    - Malformed CSV structure
    - Very large file size
    - Special characters in column names
    """)