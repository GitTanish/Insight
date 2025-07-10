import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent


class DataAnalysisAgent:
    """AI Agent for analyzing pandas DataFrames using natural language queries."""
    
    def __init__(self, api_key: str, model: str, temperature: float):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.agent = None

    def initialize_agent(self, df):
        """Initialize the pandas DataFrame agent with the given DataFrame."""
        try:
            if not self.api_key or not self.api_key.startswith('gsk_'):
                raise ValueError("Invalid or missing Groq API Key.")
            
            llm = ChatGroq(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature
            )
            
            self.agent = create_pandas_dataframe_agent(
                llm,
                df,
                agent_type="openai-tools",
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                allow_dangerous_code=True
            )
            return True
        except Exception as e:
            raise Exception(f"Failed to initialize agent: {str(e)}")

    def query(self, user_query: str, df_info: dict):
        """Process a user query and return analysis results."""
        if not self.agent:
            raise Exception("Agent not initialized")
        
        # Generate unique plot filename
        st.session_state.plot_counter += 1
        plot_filename = f"temp_plot_{st.session_state.plot_counter}.png"
        
        enhanced_query = self._create_enhanced_query(user_query, df_info, plot_filename)
        
        try:
            response = self.agent.invoke({"input": enhanced_query})
            output = self._process_response(response)
            
            # Check for plot file
            plot_path = self._find_plot_file(plot_filename)
            
            return {
                'output': output,
                'plot_path': plot_path,
                'success': True
            }
        except Exception as e:
            return self._handle_query_error(e)

    def _create_enhanced_query(self, user_query: str, df_info: dict, plot_filename: str):
        """Create an enhanced query with visualization instructions."""
        return f"""
        Please analyze the following request about the dataset:
        Query: {user_query}
        
        Dataset info:
        - Shape: {df_info['shape']}
        - Columns: {df_info['columns']}
        
        IMPORTANT VISUALIZATION INSTRUCTIONS:
        1. If creating ANY visualization, you MUST save it using: plt.savefig('{plot_filename}', dpi=300, bbox_inches='tight')
        2. After saving, call plt.close() to free memory
        3. Use proper titles, labels, and formatting for all plots
        4. For matplotlib plots, set figure size: plt.figure(figsize=(10, 6))
        5. Make sure to import matplotlib.pyplot as plt if needed
        
        Example code for visualizations:
        ```python
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        # Your plot code here
        plt.title('Your Plot Title')
        plt.xlabel('X Label')
        plt.ylabel('Y Label')
        plt.savefig('{plot_filename}', dpi=300, bbox_inches='tight')
        plt.close()
        ```
        
        Please provide a comprehensive answer with proper visualizations when requested.
        """

    def _process_response(self, response):
        """Process the agent response and handle formatting."""
        if isinstance(response, dict):
            output = response.get('output', str(response))
        else:
            output = str(response)
        
        # Handle max iterations
        if "Agent stopped due to max iterations" in output:
            output = """
            **Analysis Results:**
            The analysis was extensive and reached the iteration limit, but here's what was processed:
            """ + output.replace("Agent stopped due to max iterations", "").strip()
            output += "\n\n💡 **Tip:** The analysis was comprehensive but hit the iteration limit. The results above show the insights discovered. You can ask more specific questions to get detailed analysis on particular aspects of your data."
        
        return output

    def _find_plot_file(self, plot_filename: str):
        """Find and return the path to the generated plot file."""
        if os.path.exists(plot_filename):
            return plot_filename
        elif os.path.exists("temp_plot.png"):  # Fallback to original name
            return "temp_plot.png"
        return None

    def _handle_query_error(self, error):
        """Handle errors during query processing."""
        error_msg = str(error)
        if "Agent stopped due to max iterations" in error_msg:
            return {
                'output': "The analysis was quite extensive and reached the processing limit. Let me try to provide what insights were discovered. Try asking a more specific question about your data (e.g., 'Show me correlations between columns' or 'Create a histogram of column X') for faster results.",
                'plot_path': None,
                'success': False
            }
        else:
            return {
                'output': f"Error processing query: {error_msg}",
                'plot_path': None,
                'success': False
            }


def initialize_agent_if_needed(api_key: str, model: str, temperature: float, df):
    """Initialize agent if needed or configuration changed."""
    if (st.session_state.agent is None or 
        not st.session_state.agent_initialized or
        st.session_state.get('last_config') != (api_key, model, temperature)):
        
        try:
            agent = DataAnalysisAgent(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
            agent.initialize_agent(df)
            st.session_state.agent = agent
            st.session_state.agent_initialized = True
            st.session_state.last_config = (api_key, model, temperature)
            return True
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            return False
    return True