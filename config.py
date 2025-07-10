# --- Model Configuration ---
MODEL_OPTIONS = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mistral-saba-24b",
    "compound-beta"
]

DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.0

# --- Sample Questions ---
SAMPLE_QUESTIONS = [
    "Show me the first 10 rows of the data",
    "What are the column names and their data types?",
    "Create a histogram of [column_name]",
    "Show correlation between [column1] and [column2]",
    "What's the average, median, and standard deviation of [column_name]?",
    "Create a scatter plot of [column1] vs [column2]",
    "Show missing values in the dataset",
    "Group the data by [column] and show summary statistics"
]

# --- Quick Actions ---
QUICK_ACTIONS = [
    {"label": "Data Summary", "query": "Show me a comprehensive summary of this dataset including basic statistics"},
    {"label": "Create Visualizations", "query": "Create 2-3 simple but informative visualizations from this data"},
    {"label": "Find Key Patterns", "query": "Identify the top 3 most interesting patterns in this dataset with brief explanations"},
    {"label": "Data Quality Check", "query": "Perform a quick data quality assessment - check for missing values, duplicates, and data types"}
]

# --- Security Warning ---
SECURITY_WARNING = """⚠️ **Security Notice**: This application uses LangChain's pandas agent which can execute Python code to analyze your data. The code execution is sandboxed within the Streamlit environment, but please ensure you only upload trusted CSV files and use this tool in a secure environment."""

# --- File Processing Settings ---
MAX_ROWS = 1_000_000
ENCODINGS_TO_TRY = ['utf-8', 'latin1', 'cp1252']
DELIMITERS_TO_TRY = [',', ';', '\t']
MAX_PLOT_FILES = 10