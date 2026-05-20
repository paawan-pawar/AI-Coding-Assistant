# streamlit_app.py
import streamlit as st
import sys
import os
import tempfile
import time
from pathlib import Path

# Add the current directory to path to import the assistant
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing assistant
from free_coding_assistant import FreeCodingAssistant, Config, TaskType

# Configure Streamlit page
st.set_page_config(
    page_title="AI Coding Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .code-block {
        background-color: #1e1e1e;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
    }
    .stCodeBlock {
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_code' not in st.session_state:
    st.session_state.current_code = ""
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []
if 'model_status' not in st.session_state:
    st.session_state.model_status = "Not initialized"
if 'ollama_ok' not in st.session_state:
    st.session_state.ollama_ok = None
if 'ollama_diag' not in st.session_state:
    st.session_state.ollama_diag = ""

class StreamlitUI:
    """Streamlit UI for the coding assistant"""
    
    def __init__(self):
        self.assistant = None
        
    def initialize_assistant(self):
        """Initialize the assistant with error handling"""
        try:
            with st.spinner("🚀 Initializing AI Assistant..."):
                self.assistant = FreeCodingAssistant()
                st.session_state.assistant = self.assistant
                self.refresh_ollama_status()
                return True
        except Exception as e:
            st.session_state.model_status = f"❌ Error: {str(e)}"
            st.error(f"Failed to initialize assistant: {e}")
            return False

    def refresh_ollama_status(self):
        """Check if Ollama is reachable and the configured model is available."""
        assistant = st.session_state.assistant
        if not assistant:
            st.session_state.ollama_ok = False
            st.session_state.ollama_diag = "Assistant not initialized."
            st.session_state.model_status = "❌ Not initialized"
            return

        ok, diag = assistant.llm.diagnose()
        st.session_state.ollama_ok = ok
        st.session_state.ollama_diag = diag

        if not ok:
            st.session_state.model_status = "❌ Ollama not reachable"
        elif "not installed" in diag:
            st.session_state.model_status = "⚠️ Model not installed"
        else:
            st.session_state.model_status = "✅ Ready"

    def require_ollama(self) -> bool:
        """Guard to prevent requests when Ollama isn't ready."""
        if st.session_state.ollama_ok is None:
            self.refresh_ollama_status()

        if not st.session_state.ollama_ok:
            st.error(st.session_state.ollama_diag or "Ollama is not reachable.")
            return False

        if "not installed" in (st.session_state.ollama_diag or ""):
            st.warning(st.session_state.ollama_diag)
            return False

        return True
    
    def render_sidebar(self):
        """Render sidebar with configuration and info"""
        with st.sidebar:
            st.title("🤖 AI Coding Assistant")
            st.markdown("---")
            
            # Model Information
            st.subheader("📊 Model Info")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model", Config.OLLAMA_MODEL.split(":")[0])
            with col2:
                st.metric("Status", st.session_state.model_status)

            st.caption(f"Host: {getattr(Config, 'OLLAMA_HOST', 'http://127.0.0.1:11434')}")

            if st.button("🔌 Re-check Ollama"):
                self.refresh_ollama_status()
                st.rerun()

            if st.session_state.model_status != "✅ Ready":
                with st.expander("🔧 Fix Ollama connection", expanded=True):
                    if st.session_state.ollama_diag:
                        st.markdown(st.session_state.ollama_diag)
                    st.markdown(
                        """
1) Install Ollama: https://ollama.com/download
2) Start it:
    - Windows: open the **Ollama** app (Start Menu). This usually starts the local server.
    - Or run `ollama serve`.
    - If `ollama` is "not recognized" in your terminal, try:
      - `%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe serve`
3) Pull the model:
   - `ollama pull deepseek-coder:6.7b-instruct`

If Ollama runs on a different host/port, set `OLLAMA_HOST` (e.g. `http://127.0.0.1:11434`) and restart Streamlit.
                        """
                    )
            
            st.markdown("---")
            
            # Knowledge Base Stats
            st.subheader("📚 Knowledge Base")
            if st.session_state.assistant:
                try:
                    if getattr(st.session_state.assistant, "vector_store_ready", False):
                        collection = st.session_state.assistant.vector_store.collection
                        count = collection.count()
                        st.metric("Files Indexed", count)
                    else:
                        st.metric("Files Indexed", "Not loaded")
                except Exception:
                    st.metric("Files Indexed", "N/A")
            
            st.markdown("---")
            
            # Quick Actions
            st.subheader("⚡ Quick Actions")
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("🔄 Reset Assistant"):
                st.session_state.assistant = None
                st.session_state.messages = []
                st.rerun()
            
            st.markdown("---")
            
            # Help Section
            st.subheader("ℹ️ How to Use")
            st.markdown("""
            **Chat Tab**: Ask any coding question
            **Code Generation**: Generate code from prompts
            **Code Review**: Get feedback on your code
            **Debug**: Fix errors in your code
            **Refactor**: Improve code structure
            **Test**: Generate unit tests
            **Document**: Add documentation
            **Knowledge**: Index your codebase
            """)
            
            st.markdown("---")
            st.caption(f"Using model: {Config.OLLAMA_MODEL}")
            st.caption("Powered by Ollama (Local, Free)")
    
    def render_chat_tab(self):
        """Render chat interface"""
        st.header("💬 Chat with AI Assistant")
        st.markdown("Ask any coding question, and I'll help you solve it!")

        # If we need to clear the chat prompt, do it BEFORE the widget is created.
        if st.session_state.get("_reset_chat_prompt", False):
            st.session_state["chat_prompt"] = ""
            st.session_state["_reset_chat_prompt"] = False
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Streamlit restriction: st.chat_input() can't be used inside st.tabs.
        # Use a standard input + button to keep the same UX within the Chat tab.
        prompt = st.text_input(
            "Ask me anything about coding...",
            key="chat_prompt",
            placeholder="Example: Why am I getting a ModuleNotFoundError in Python?"
        )
        send = st.button("Send", type="primary")

        if send and prompt.strip():
            if not self.require_ollama():
                return
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.assistant.process_request(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # Clear the text input on the next run (can't modify widget state after instantiation).
                    st.session_state["_reset_chat_prompt"] = True
                    st.rerun()
    
    def render_code_generation_tab(self):
        """Render code generation interface"""
        st.header("✨ Code Generation")
        st.markdown("Describe what you want to create, and I'll generate the code for you.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            prompt = st.text_area(
                "What code do you want to generate?",
                placeholder="Example: Create a REST API endpoint for user authentication with JWT tokens",
                height=150
            )
        
        with col2:
            language = st.selectbox(
                "Programming Language",
                ["python", "javascript", "typescript", "java", "cpp", "go", "rust"],
                index=0
            )
            
            include_comments = st.checkbox("Include comments", value=True)
            error_handling = st.checkbox("Add error handling", value=True)
        
        if st.button("🚀 Generate Code", type="primary"):
            if prompt:
                if not self.require_ollama():
                    return
                with st.spinner("Generating code..."):
                    context = {
                        "language": language,
                        "include_comments": include_comments,
                        "error_handling": error_handling
                    }
                    code = st.session_state.assistant._handle_code_generation(prompt, context)
                    
                    # Display generated code
                    st.subheader("📝 Generated Code")
                    st.code(code, language=language)
                    
                    # Copy button
                    if st.button("📋 Copy to Clipboard"):
                        st.write("Code copied to clipboard!")
                    
                    # Save button
                    if st.button("💾 Save to File"):
                        save_path = st.text_input("Save as:", f"generated_code.{language}")
                        if save_path:
                            with open(save_path, 'w') as f:
                                f.write(code)
                            st.success(f"Saved to {save_path}")
            else:
                st.warning("Please enter a prompt first.")
    
    def render_code_review_tab(self):
        """Render code review interface"""
        st.header("🔍 Code Review")
        st.markdown("Paste your code and I'll review it for bugs, security issues, and improvements.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            code = st.text_area(
                "Paste your code here:",
                value=st.session_state.current_code,
                height=300,
                placeholder="def example_function():\n    pass"
            )
        
        with col2:
            language = st.selectbox(
                "Language",
                ["python", "javascript", "typescript", "java", "cpp"],
                index=0
            )
            review_type = st.selectbox(
                "Review Focus",
                ["Complete Review", "Security Only", "Performance Only", "Best Practices Only"]
            )
        
        if st.button("🔍 Review Code", type="primary"):
            if code:
                if not self.require_ollama():
                    return
                with st.spinner("Analyzing code..."):
                    context = {"code": code, "language": language}
                    review = st.session_state.assistant._handle_code_review("Review this code", context)
                    
                    # Display review
                    st.subheader("📊 Code Review Results")
                    
                    # Parse and display review in sections
                    if "Security" in review:
                        st.warning("🔒 Security Issues")
                        st.markdown(review)
                    else:
                        st.info("Code Analysis")
                        st.markdown(review)
                    
                    # Add to current code
                    st.session_state.current_code = code
            else:
                st.warning("Please paste some code to review.")
    
    def render_debug_tab(self):
        """Render debugging interface"""
        st.header("🐛 Debug Assistant")
        st.markdown("Paste your error message and code, and I'll help you fix it.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            error = st.text_area(
                "Error Message:",
                placeholder="Traceback (most recent call last):\n  File \"app.py\", line 10, in <module>\n    result = 10 / 0\nZeroDivisionError: division by zero",
                height=150
            )
        
        with col2:
            debug_code = st.text_area(
                "Related Code (optional):",
                placeholder="def calculate(x, y):\n    return x / y\n\nresult = calculate(10, 0)",
                height=150
            )
        
        if st.button("🐛 Debug", type="primary"):
            if error:
                if not self.require_ollama():
                    return
                with st.spinner("Analyzing error..."):
                    context = {"error": error, "code": debug_code}
                    solution = st.session_state.assistant._handle_debugging(error, context)
                    
                    st.subheader("🔧 Solution")
                    st.success("Here's how to fix it:")
                    st.markdown(solution)
            else:
                st.warning("Please paste the error message.")
    
    def render_refactor_tab(self):
        """Render refactoring interface"""
        st.header("♻️ Code Refactoring")
        st.markdown("Improve your code structure and quality.")
        
        code = st.text_area(
            "Code to refactor:",
            height=300,
            placeholder="def bad_function(a,b,c):\n    if a==1:\n        return b\n    elif a==2:\n        return c\n    else:\n        return None"
        )
        
        refactor_type = st.selectbox(
            "Refactoring Type",
            ["Improve Readability", "Optimize Performance", "Extract Functions", "Simplify Logic", "Custom"]
        )
        
        if st.button("♻️ Refactor Code", type="primary"):
            if code:
                if not self.require_ollama():
                    return
                with st.spinner("Refactoring..."):
                    context = {"code": code}
                    refactored = st.session_state.assistant._handle_refactoring(refactor_type, context)
                    
                    st.subheader("✨ Refactored Code")
                    st.code(refactored, language="python")
                    
                    # Show diff
                    st.subheader("📊 Changes")
                    st.markdown("**Before:**")
                    st.code(code, language="python")
            else:
                st.warning("Please paste code to refactor.")
    
    def render_test_tab(self):
        """Render test generation interface"""
        st.header("🧪 Test Generator")
        st.markdown("Generate unit tests for your code.")
        
        code = st.text_area(
            "Code to test:",
            height=300,
            placeholder="def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b"
        )
        
        test_framework = st.selectbox(
            "Test Framework",
            ["pytest", "unittest", "jest", "mocha"]
        )
        
        if st.button("🧪 Generate Tests", type="primary"):
            if code:
                if not self.require_ollama():
                    return
                with st.spinner("Generating tests..."):
                    context = {"code": code}
                    tests = st.session_state.assistant._handle_testing(f"Generate {test_framework} tests", context)
                    
                    st.subheader("✅ Generated Tests")
                    st.code(tests, language="python")
                    
                    # Run option
                    if st.button("▶️ Run Tests"):
                        st.info("Running tests... (This feature coming soon)")
            else:
                st.warning("Please paste code to test.")
    
    def render_document_tab(self):
        """Render documentation interface"""
        st.header("📚 Documentation Generator")
        st.markdown("Add comprehensive documentation to your code.")
        
        code = st.text_area(
            "Code to document:",
            height=300,
            placeholder="def complex_algorithm(data, threshold=0.5):\n    # Implementation here\n    pass"
        )
        
        doc_type = st.selectbox(
            "Documentation Type",
            ["Docstrings", "README", "API Documentation", "Inline Comments"]
        )
        
        if st.button("📝 Generate Documentation", type="primary"):
            if code:
                if not self.require_ollama():
                    return
                with st.spinner("Generating documentation..."):
                    context = {"code": code}
                    docs = st.session_state.assistant._handle_documentation(doc_type, context)
                    
                    st.subheader("📖 Documentation")
                    st.markdown(docs)
            else:
                st.warning("Please paste code to document.")
    
    def render_knowledge_tab(self):
        """Render knowledge base management"""
        st.header("🗄️ Knowledge Base Management")
        st.markdown("Index your codebase for semantic search.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 Add Single File")
            uploaded_file = st.file_uploader(
                "Upload a code file",
                type=['py', 'js', 'ts', 'java', 'cpp', 'go', 'rs']
            )
            
            if uploaded_file:
                with tempfile.NamedTemporaryFile(mode='w', suffix=f".{uploaded_file.name.split('.')[-1]}", delete=False) as f:
                    content = uploaded_file.read().decode()
                    f.write(content)
                    temp_path = f.name
                
                if st.button("➕ Add to Knowledge Base"):
                    with st.spinner("Indexing file..."):
                        result = st.session_state.assistant.add_to_knowledge_base(temp_path)
                        st.success(result)
                    os.unlink(temp_path)
            
            st.subheader("🌐 Add from URL")
            repo_url = st.text_input("GitHub Repository URL")
            if st.button("Clone and Index"):
                st.info("This feature will clone and index the repository.")
        
        with col2:
            st.subheader("📂 Index Directory")
            dir_path = st.text_input("Directory Path", placeholder="./src")
            
            if st.button("📚 Index Directory", type="primary"):
                if os.path.exists(dir_path):
                    with st.spinner(f"Indexing {dir_path}..."):
                        result = st.session_state.assistant.index_directory(dir_path)
                        st.success(result)
                else:
                    st.error("Directory does not exist.")
            
            st.subheader("🔍 Search Knowledge Base")
            search_query = st.text_input("Search query", placeholder="database connection")
            if st.button("Search"):
                if search_query:
                    with st.spinner("Searching..."):
                        results = st.session_state.assistant.vector_store.search(search_query)
                        
                        if results:
                            st.success(f"Found {len(results)} results")
                            for i, result in enumerate(results, 1):
                                with st.expander(f"Result {i} - Score: {result['distance']:.2f}"):
                                    st.code(result['code'][:500], language="python")
                                    if 'metadata' in result:
                                        st.json(result['metadata'])
                        else:
                            st.info("No results found.")

def main():
    """Main function to run the Streamlit app"""
    
    # Initialize UI
    ui = StreamlitUI()
    
    # Initialize assistant if not already
    if st.session_state.assistant is None:
        if ui.initialize_assistant():
            st.rerun()
    
    # Render sidebar
    ui.render_sidebar()
    
    # Main content area with tabs
    if st.session_state.assistant:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "💬 Chat", 
            "✨ Generate", 
            "🔍 Review", 
            "🐛 Debug",
            "♻️ Refactor",
            "🧪 Test",
            "📚 Document",
            "🗄️ Knowledge"
        ])
        
        with tab1:
            ui.render_chat_tab()
        
        with tab2:
            ui.render_code_generation_tab()
        
        with tab3:
            ui.render_code_review_tab()
        
        with tab4:
            ui.render_debug_tab()
        
        with tab5:
            ui.render_refactor_tab()
        
        with tab6:
            ui.render_test_tab()
        
        with tab7:
            ui.render_document_tab()
        
        with tab8:
            ui.render_knowledge_tab()
    else:
        st.error("Failed to initialize assistant. Please check the error message in the sidebar.")
        
        # Show troubleshooting tips
        with st.expander("🔧 Troubleshooting Tips"):
            st.markdown("""
            **Common Issues:**
            1. **Ollama not running**: Make sure Ollama is running with `ollama serve`
            2. **Model not downloaded**: Run `ollama pull deepseek-coder:6.7b-instruct`
            3. **Memory issues**: Try a smaller model like `codellama:7b-instruct`
            4. **Python dependencies**: Install required packages with `pip install ollama chromadb streamlit`
            
            **Quick Fix:**
            ```bash
            # Check Ollama status
            ollama list
            
            # Pull the model
            ollama pull deepseek-coder:6.7b-instruct
            
            # Restart Ollama
            ollama serve""")

if __name__ == "__main__":
    # Streamlit scripts are meant to be launched with `streamlit run`.
    # When executed with plain `python streamlit_app.py`, Streamlit runs in
    # "bare mode" and session_state/widgets will not behave correctly.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
    except Exception:
        get_script_run_ctx = None

    if get_script_run_ctx is not None and get_script_run_ctx() is not None:
        main()
    else:
        print("Run this app with Streamlit:")
        print("  python -m streamlit run streamlit_app.py")