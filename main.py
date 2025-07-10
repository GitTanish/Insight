import streamlit as st
import matplotlib
from dotenv import load_dotenv
from utils import initialize_session_state, load_dataframe_from_session
from ui_components import (
    render_sidebar, render_main_header, render_api_key_info,
    render_sample_questions, render_data_overview, render_quick_actions,
    render_chat_interface, render_error_message
)

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib backend for Streamlit
matplotlib.use('Agg')


def main():
    """Main application function."""
    initialize_session_state()
    render_main_header()
    config = render_sidebar()
    
    # Check if API key is provided
    if not config['api_key']:
        render_api_key_info()
        return
    
    # Check if file is uploaded
    if 'uploaded_file_content' not in st.session_state or st.session_state.uploaded_file_content is None:
        st.info("Please upload a CSV file using the sidebar to get started.")
        render_sample_questions()
        return
    
    # Load DataFrame from session
    df, error = load_dataframe_from_session()
    if error:
        render_error_message(error)
        return
    
    # Render main application interface
    render_data_overview(df)
    render_quick_actions()
    render_chat_interface(config['api_key'], config['model'], config['temperature'], df)


if __name__ == "__main__":
    main()