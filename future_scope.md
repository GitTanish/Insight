# Insight v3 Future Scope

The next major iteration of Insight focuses on structural and functional enhancements to deepen the analytical capabilities of the platform, moving beyond visual improvements.

## 1. Automatic Dataset Profiling
Upon uploading a dataset, Insight will automatically run a comprehensive background analysis (e.g., using `ydata-profiling` or custom automated agents) to generate an immediate statistical report, before the user even asks their first question.

## 2. Multi-File Reasoning
Support for uploading and analyzing multiple datasets simultaneously. The AI agent will be able to perform advanced SQL-like table joins, cross-reference data, and build insights drawn from relationships across multiple disparate CSV files.

## 3. Tool-Based Statistical Analysis
Expanding the LangChain agent's capabilities by equipping it with a dedicated suite of statistical tools (e.g., specialized tools for T-tests, linear regression models, ANOVA, and time-series forecasting) rather than relying solely on pure Python REPL code generation.

## 4. Persistent Session Memory
Implementing a persistent storage layer (e.g., local SQLite, PostgreSQL, or simple JSON/Pickle exports) so users can close the application, return days later, and pick up their data exploration exactly where they left off without losing their chat history or session state.
