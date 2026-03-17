from __future__ import annotations
import os
import streamlit as st
import pandas as pd
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

    def query(self, user_query: str, df: pd.DataFrame):
        """Process a user query and return analysis results."""
        if not self.agent:
            raise Exception("Agent not initialized")
        
        # Generate unique plot filename
        st.session_state.plot_counter += 1
        plot_filename = f"temp_plot_{st.session_state.plot_counter}.png"
        
        # Get detailed metadata
        df_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict() if not df.empty else {}
        }
        
        enhanced_query = self._create_enhanced_query(user_query, df_info, plot_filename)
        
        try:
            response = self.agent.invoke({"input": enhanced_query})
            output = self._process_response(response)
            
            # Check for plot files (support multiple)
            plot_paths = self._find_plot_files(st.session_state.plot_counter)
            
            return {
                'output': output,
                'plot_paths': plot_paths,
                'success': True
            }
        except Exception as e:
            return self._handle_query_error(e)

    def _create_enhanced_query(self, user_query: str, df_info: dict, plot_filename: str):
        """Create an enhanced query with visualization instructions."""
        counter = st.session_state.plot_counter
        return f"""
        Please analyze the following request about the dataset:
        Query: {user_query}
        
        Detailed Dataset Metadata:
        - Shape: {df_info['shape']}
        - Columns: {df_info['columns']}
        - Data Types: {df_info['dtypes']}
        - Missing Values: {df_info['null_counts']}
        
        CRITICAL EXECUTION INSTRUCTIONS:
        1. You MUST EXECUTE all Python code using the available tools to perform analysis.
        2. DO NOT include the Python code in your final response to the user.
        3. Only provide the textual analysis and insights in your final response.
        
        VISUALIZATION INSTRUCTIONS:
        1. If creating visualizations, they MUST follow a 'Vintage Newspaper' aesthetic.
        2. Set figure background to the paper color: plt.rcParams['figure.facecolor'] = '#f2ead8'
        3. Use ink-black (#1a1a1b) for all text, labels, and axes.
        4. Use a muted, classic color palette (e.g., 'Greys', 'copper' or 'bone' for continuous, or a custom muted set for categorical).
        5. You MUST save each plot with: plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#f2ead8')
        6. Use the naming convention: 'temp_plot_{counter}_1.png', 'temp_plot_{counter}_2.png'.
        7. No modern neon colors, no glowing effects. Use high contrast ink-on-paper look.
        8. Use 'EB Garamond' or 'serif' for fonts if possible.
        
        Example for vintage plots:
        ```python
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.rcParams['figure.facecolor'] = '#f2ead8'
        plt.rcParams['text.color'] = '#1a1a1b'
        plt.rcParams['axes.labelcolor'] = '#1a1a1b'
        
        # Plot 1
        plt.figure(figsize=(10, 6), facecolor='#f2ead8')
        sns.set_palette(['#1a1a1b', '#8b1d1d', '#3a3a3c', '#525252']) 
        sns.boxplot(...)
        plt.title('Plot Title', fontweight='bold', fontsize=16)
        plt.savefig('temp_plot_{counter}_1.png', dpi=300, bbox_inches='tight', facecolor='#f2ead8')
        plt.close()
        ```
        
        Analysis Strategy:
        - First, understand the column types.
        - Execute your analysis code to get facts.
        - If visualizations are needed, save them as instructed.
        - Provide a comprehensive textual summary of your findings as the final output.
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

    def _find_plot_files(self, counter: int):
        """Find and return a list of paths to all generated plot files."""
        plots = []
        # Check for specific numbered plots
        for i in range(1, 6):  # Check for up to 5 plots
            filename = f"temp_plot_{counter}_{i}.png"
            if os.path.exists(filename):
                plots.append(filename)
        
        # Fallback to the original plot filename if no numbered ones exist
        filename_orig = f"temp_plot_{counter}.png"
        if os.path.exists(filename_orig):
            plots.append(filename_orig)
            
        return plots

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
                'plot_paths': [],
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