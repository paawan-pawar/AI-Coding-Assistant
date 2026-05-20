# free_coding_assistant.py (No Gradio dependency)
import os
import json
import asyncio
import subprocess
import ast
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
from datetime import datetime

# Free, local LLM via Ollama
import ollama

# NOTE: ChromaDB (and its optional embedding backends like onnxruntime/torch)
# can fail to import on some Windows setups due to missing native DLL deps.
# Keep it as an optional, lazy import so the assistant can still run without
# knowledge-base features.

# File watching
import watchdog.events
import watchdog.observers

class Config:
    """Configuration for the free assistant"""
    # Ollama settings
    OLLAMA_MODEL = "deepseek-coder:6.7b-instruct"  # Change to your preferred model
    # Alternative models: "deepseek-coder:6.7b-instruct", "mistral:7b-instruct", "llama2:7b"
    # Ollama server URL. Override with env var, e.g.:
    #   Windows PowerShell: $env:OLLAMA_HOST = "http://127.0.0.1:11434"
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    
    # Vector database
    CHROMA_DB_PATH = "./chroma_db"
    
    # Storage
    CODE_STORE_PATH = "./code_store"
    CONTEXT_PATH = "./contexts"
    
    # Maximum tokens for context
    MAX_CONTEXT_TOKENS = 4096

# Create necessary directories
os.makedirs(Config.CODE_STORE_PATH, exist_ok=True)
os.makedirs(Config.CONTEXT_PATH, exist_ok=True)

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    EXPLANATION = "explanation"

@dataclass
class CodeContext:
    """Context for code understanding"""
    file_path: str
    content: str
    language: str
    imports: List[str]
    functions: List[str]
    classes: List[str]
    docstrings: Dict[str, str]

class LocalLLM:
    """Wrapper for local Llama models via Ollama"""
    
    def __init__(self, model_name: str = Config.OLLAMA_MODEL):
        self.model_name = model_name
        self.conversation_history = []
        self.host = Config.OLLAMA_HOST
        self._client = self._create_client()

    def _create_client(self):
        """Create an Ollama client when available (allows explicit host).

        The upstream `ollama` Python package exposes both module-level helpers
        (`ollama.chat`, `ollama.list`) and a `Client` class in most versions.
        We prefer the client so we can target a specific host, but fall back
        gracefully for older versions.
        """
        client_cls = getattr(ollama, "Client", None)
        if client_cls is None:
            # Fall back to module-level functions; they usually respect OLLAMA_HOST.
            os.environ.setdefault("OLLAMA_HOST", self.host)
            return None

        try:
            return client_cls(host=self.host)
        except TypeError:
            # Older client signatures may not accept `host`.
            os.environ.setdefault("OLLAMA_HOST", self.host)
            try:
                return client_cls()
            except Exception:
                return None

    def _list_models(self) -> Dict[str, Any]:
        if self._client is not None:
            return self._client.list()
        return ollama.list()

    def _chat(self, **kwargs) -> Dict[str, Any]:
        if self._client is not None:
            return self._client.chat(**kwargs)
        return ollama.chat(**kwargs)

    def diagnose(self) -> Tuple[bool, str]:
        """Return (ok, message) describing Ollama availability.

        - ok=False: Ollama isn't reachable at the configured host.
        - ok=True with warning text: service reachable but model missing.
        """
        try:
            listing = self._list_models()
        except Exception as e:
            msg = (
                f"Failed to connect to Ollama at {self.host}. "
                "Start Ollama (Windows: open the Ollama app or run `ollama serve`), "
                f"then pull the model with `ollama pull {self.model_name}`. "
                "If Ollama is running on a different machine/port, set OLLAMA_HOST. "
                f"\n\nUnderlying error: {e}"
            )
            return False, msg

        models = []
        try:
            models = [m.get("name") for m in (listing.get("models") or []) if isinstance(m, dict)]
        except Exception:
            models = []

        if models and self.model_name not in models:
            return True, (
                f"Ollama is running at {self.host}, but model '{self.model_name}' is not installed. "
                f"Run `ollama pull {self.model_name}`."
            )

        return True, f"Ollama is reachable at {self.host}."
        
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response from local model"""
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation history (last 5 exchanges)
        for msg in self.conversation_history[-10:]:
            messages.append(msg)
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            response = self._chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": 0.2,  # Lower for more deterministic code
                    "top_p": 0.9,
                    "num_predict": 2048
                },
            )
            
            result = response['message']['content']
            
            # Update history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": result})
            
            return result
            
        except Exception as e:
            # Provide a more actionable error for the common "Ollama not running" case.
            ok, diag = self.diagnose()
            if not ok:
                return f"Error generating response: {diag}"
            return (
                "Error generating response: "
                f"{e}\n\nOllama host: {self.host}\nModel: {self.model_name}\n\n"
                "If the model isn't installed, run: "
                f"ollama pull {self.model_name}"
            )
    
    def generate_code(self, prompt: str, language: str = "python") -> str:
        """Specialized code generation"""
        system_prompt = f"""You are an expert {language} developer. Generate clean, efficient, and well-documented code.
        Follow best practices, include error handling, and add comments for complex logic.
        Only output the code, no explanations unless specifically asked."""
        
        full_prompt = f"Write {language} code for: {prompt}\n\nCode:"
        return self.generate(full_prompt, system_prompt)
    
    def review_code(self, code: str, language: str = "python") -> str:
        """Review code for issues and improvements"""
        system_prompt = """You are a senior code reviewer. Analyze code for:
        1. Bugs and logical errors
        2. Security vulnerabilities
        3. Performance issues
        4. Best practices violations
        5. Code style and readability
        
        Provide specific, actionable feedback."""
        
        prompt = f"Review this {language} code:\n\n```{language}\n{code}\n```\n\nReview:"
        return self.generate(prompt, system_prompt)
    
    def debug_error(self, error: str, code: str = None) -> str:
        """Debug errors and suggest fixes"""
        system_prompt = "You are a debugging expert. Analyze errors and provide solutions."
        
        context = f"Error: {error}\n"
        if code:
            context += f"\nCode:\n{code}\n"
        
        prompt = context + "\nWhat's the issue and how to fix it?"
        return self.generate(prompt, system_prompt)

class CodeAnalyzer:
    """Free code analysis with AST"""
    
    def analyze(self, file_path: str) -> CodeContext:
        """Analyze code file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ext = file_path.split('.')[-1]
        language = self._detect_language(ext)
        
        if language == "python":
            return self._analyze_python(file_path, content)
        elif language in ["javascript", "typescript", "js", "ts"]:
            return self._analyze_generic(file_path, content, language)
        else:
            return self._analyze_generic(file_path, content, language)
    
    def _detect_language(self, ext: str) -> str:
        mapping = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'go': 'go',
            'rs': 'rust'
        }
        return mapping.get(ext, 'text')
    
    def _analyze_python(self, file_path: str, content: str) -> CodeContext:
        """Analyze Python code with AST"""
        try:
            tree = ast.parse(content)
            functions = []
            classes = []
            imports = []
            docstrings = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                    docstrings[node.name] = ast.get_docstring(node) or ""
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    docstrings[node.name] = ast.get_docstring(node) or ""
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            return CodeContext(
                file_path=file_path,
                content=content,
                language="python",
                imports=imports,
                functions=functions,
                classes=classes,
                docstrings=docstrings
            )
        except Exception as e:
            print(f"AST parsing error: {e}")
            return self._analyze_generic(file_path, content, "python")
    
    def _analyze_generic(self, file_path: str, content: str, language: str) -> CodeContext:
        """Simple analysis for other languages"""
        lines = content.split('\n')
        imports = [line for line in lines if 'import' in line or 'require' in line]
        functions = [line for line in lines if 'def ' in line or 'function' in line]
        classes = [line for line in lines if 'class ' in line]
        
        return CodeContext(
            file_path=file_path,
            content=content,
            language=language,
            imports=imports[:10],  # Limit size
            functions=functions[:10],
            classes=classes[:10],
            docstrings={}
        )

class VectorCodeStore:
    """Free vector database for code search using ChromaDB"""
    
    def __init__(self):
        try:
            import chromadb  # type: ignore
            from chromadb.utils import embedding_functions  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "ChromaDB is not available in this environment. "
                "Install/fix its optional native dependencies (e.g. onnxruntime) "
                "or disable knowledge-base features. Original error: "
                f"{e}"
            )

        # Create client with persistent storage
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # Free, local embedding model
        )
        self.collection = self.client.get_or_create_collection(
            name="code_embeddings",
            embedding_function=self.embedding_fn
        )
    
    def add_code(self, code_id: str, code: str, metadata: Dict):
        """Add code to vector store"""
        try:
            # Split code into chunks for better retrieval
            chunks = self._chunk_code(code)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{code_id}_{i}"
                self.collection.upsert(
                    documents=[chunk],
                    metadatas=[{**metadata, "chunk_index": i}],
                    ids=[chunk_id]
                )
            return True
        except Exception as e:
            print(f"Error adding code: {e}")
            return False
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant code"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            return [
                {
                    "code": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _chunk_code(self, code: str, max_chunk_size: int = 1000) -> List[str]:
        """Split code into chunks"""
        if len(code) <= max_chunk_size:
            return [code]
        
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > max_chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = len(line)
            else:
                current_chunk.append(line)
                current_size += len(line)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks


class NullVectorStore:
    """Fallback vector store used when optional embedding deps aren't installed."""

    def __init__(self, reason: str = ""):
        self.reason = reason
        self.collection = None

    def add_code(self, code_id: str, code: str, metadata: Dict):
        return False

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        return []

class FreeCodingAssistant:
    """Main coding assistant with local Llama models"""
    
    def __init__(self):
        self.llm = LocalLLM()
        self.analyzer = CodeAnalyzer()
        self._vector_store = None
        self.context_cache = {}

    @property
    def vector_store(self) -> 'VectorCodeStore':
        """Lazily initialize the vector store.

        This avoids pulling in heavier ML dependencies (e.g., torch) during
        Streamlit startup until knowledge-base features are actually used.
        """
        if self._vector_store is None:
            try:
                self._vector_store = VectorCodeStore()
            except Exception as e:
                # Common case: chromadb SentenceTransformer embedding requires
                # the optional `sentence-transformers` dependency.
                self._vector_store = NullVectorStore(reason=str(e))
        return self._vector_store

    @property
    def vector_store_ready(self) -> bool:
        return self._vector_store is not None
        
    def process_request(self, request: str, context: Dict = None) -> str:
        """Process a user request"""
        
        # Detect task type
        task_type = self._detect_task_type(request)
        
        # Build prompt based on task type
        if task_type == TaskType.CODE_GENERATION:
            return self._handle_code_generation(request, context)
        elif task_type == TaskType.CODE_REVIEW:
            return self._handle_code_review(request, context)
        elif task_type == TaskType.DEBUGGING:
            return self._handle_debugging(request, context)
        elif task_type == TaskType.REFACTORING:
            return self._handle_refactoring(request, context)
        elif task_type == TaskType.TESTING:
            return self._handle_testing(request, context)
        elif task_type == TaskType.DOCUMENTATION:
            return self._handle_documentation(request, context)
        else:
            return self._handle_explanation(request, context)
    
    def _detect_task_type(self, request: str) -> TaskType:
        """Detect task type from request"""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ["write", "create", "implement", "generate"]):
            return TaskType.CODE_GENERATION
        elif any(word in request_lower for word in ["review", "check", "analyze", "improve"]):
            return TaskType.CODE_REVIEW
        elif any(word in request_lower for word in ["fix", "debug", "error", "bug"]):
            return TaskType.DEBUGGING
        elif any(word in request_lower for word in ["refactor", "restructure"]):
            return TaskType.REFACTORING
        elif any(word in request_lower for word in ["test", "unit test"]):
            return TaskType.TESTING
        elif any(word in request_lower for word in ["document", "docstring", "comment"]):
            return TaskType.DOCUMENTATION
        else:
            return TaskType.EXPLANATION
    
    def _handle_code_generation(self, request: str, context: Dict = None) -> str:
        """Generate code based on request"""
        
        # Search for similar code patterns
        try:
            similar_code = self.vector_store.search(request)
        except Exception:
            similar_code = []
        
        # Build context prompt
        context_prompt = ""
        if similar_code:
            context_prompt = "\nSimilar code examples:\n"
            for i, code in enumerate(similar_code[:2], 1):
                context_prompt += f"\nExample {i}:\n```\n{code['code'][:500]}\n```\n"
        
        # Detect language
        language = "python"  # default
        if context and "language" in context:
            language = context["language"]
        
        full_prompt = f"""Generate {language} code for: {request}
{context_prompt}

Provide only the code, with brief comments explaining key parts.
Make sure the code is complete, runnable, and follows best practices.

Code:"""
        
        return self.llm.generate(full_prompt)
    
    def _handle_code_review(self, request: str, context: Dict = None) -> str:
        """Review code for issues"""
        
        # Try to get code from context or current file
        code = None
        if context and "code" in context:
            code = context["code"]
        elif context and "file" in context:
            try:
                with open(context["file"], 'r') as f:
                    code = f.read()
            except:
                pass
        
        if not code:
            return "Please provide the code to review (either in context or specify a file)."
        
        # Analyze code structure
        try:
            # Create temp file for analysis
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            code_context = self.analyzer.analyze(temp_file)
            analysis_info = f"""
Code Statistics:
- Functions: {len(code_context.functions)}
- Classes: {len(code_context.classes)}
- Imports: {len(code_context.imports)}
- Lines: {len(code.split(chr(10)))}  # Using chr(10) instead of \n
"""
            os.unlink(temp_file)
        except Exception as e:
            analysis_info = f"\nAnalysis error: {e}\n"
        
        # Get LLM review
        review = self.llm.review_code(code)
        
        return f"{analysis_info}\n{review}"
    
    def _handle_debugging(self, request: str, context: Dict = None) -> str:
        """Debug errors"""
        
        error = request
        code = None
        
        # Extract error from request if not provided separately
        if "error:" in request.lower():
            parts = request.split("\n")
            error = parts[0]
            if len(parts) > 1:
                code = "\n".join(parts[1:])
        
        if context and "code" in context:
            code = context["code"]
        elif context and "error" in context:
            error = context["error"]
        
        return self.llm.debug_error(error, code)
    
    def _handle_refactoring(self, request: str, context: Dict = None) -> str:
        """Refactor code"""
        
        code = None
        if context and "code" in context:
            code = context["code"]
        elif context and "file" in context:
            try:
                with open(context["file"], 'r') as f:
                    code = f.read()
            except:
                pass
        
        if not code:
            return "Please provide the code to refactor."
        
        system_prompt = "You are a code refactoring expert. Improve code quality while maintaining functionality."
        prompt = f"Refactor this code: {request}\n\nCode:\n{code}\n\nRefactored code:"
        
        return self.llm.generate(prompt, system_prompt)
    
    def _handle_testing(self, request: str, context: Dict = None) -> str:
        """Generate tests"""
        
        code = None
        if context and "code" in context:
            code = context["code"]
        elif context and "file" in context:
            try:
                with open(context["file"], 'r') as f:
                    code = f.read()
            except:
                pass
        
        if not code:
            return "Please provide the code to test."
        
        system_prompt = "You are a testing expert. Generate comprehensive unit tests."
        prompt = f"Generate unit tests for:\n\n{code}\n\nTests:"
        
        return self.llm.generate(prompt, system_prompt)
    
    def _handle_documentation(self, request: str, context: Dict = None) -> str:
        """Generate documentation"""
        
        code = None
        if context and "code" in context:
            code = context["code"]
        elif context and "file" in context:
            try:
                with open(context["file"], 'r') as f:
                    code = f.read()
            except:
                pass
        
        if not code:
            return "Please provide the code to document."
        
        system_prompt = "You are a technical writer. Create clear, comprehensive documentation."
        prompt = f"Document this code: {request}\n\nCode:\n{code}\n\nDocumentation:"
        
        return self.llm.generate(prompt, system_prompt)
    
    def _handle_explanation(self, request: str, context: Dict = None) -> str:
        """Explain code or concepts"""
        
        code = None
        if context and "code" in context:
            code = context["code"]
        
        if code:
            prompt = f"Explain this code: {request}\n\nCode:\n{code}\n\nExplanation:"
        else:
            prompt = f"Explain: {request}"
        
        return self.llm.generate(prompt)
    
    def add_to_knowledge_base(self, file_path: str):
        """Add code file to knowledge base"""
        try:
            code_context = self.analyzer.analyze(file_path)
            
            # Store in vector database
            code_id = hashlib.md5(file_path.encode()).hexdigest()
            metadata = {
                "file": file_path,
                "language": code_context.language,
                "functions": ", ".join(code_context.functions[:5]),
                "classes": ", ".join(code_context.classes[:5])
            }
            
            self.vector_store.add_code(code_id, code_context.content, metadata)
            return f"Added {file_path} to knowledge base"
        except Exception as e:
            return f"Error adding file: {e}"
    
    def index_directory(self, directory: str):
        """Index entire directory"""
        indexed = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.java', '.cpp')):
                    file_path = os.path.join(root, file)
                    result = self.add_to_knowledge_base(file_path)
                    if "Added" in result:
                        indexed += 1
        
        return f"Indexed {indexed} files"

# Remove the WebInterface class since we're using Streamlit
# Keep only the CLI interface for command-line usage

class CLIInterface:
    """Command-line interface"""
    
    def __init__(self):
        self.assistant = FreeCodingAssistant()
    
    def run(self):
        print("=" * 60)
        print("🤖 Free AI Coding Assistant - Powered by Local Llama")
        print("=" * 60)
        print("\nCommands:")
        print("  /help     - Show this help")
        print("  /file     - Analyze a file")
        print("  /index    - Index a directory")
        print("  /clear    - Clear conversation")
        print("  /exit     - Exit")
        print("\nJust type your coding question or request!\n")
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() == '/exit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == '/help':
                    self._show_help()
                elif user_input.lower().startswith('/file '):
                    file_path = user_input[6:].strip()
                    self._handle_file(file_path)
                elif user_input.lower().startswith('/index '):
                    dir_path = user_input[7:].strip()
                    self._handle_index(dir_path)
                elif user_input.lower() == '/clear':
                    self.assistant.llm.conversation_history = []
                    print("Conversation cleared!")
                elif user_input:
                    print("\n🤖 Assistant: ")
                    response = self.assistant.process_request(user_input)
                    print(response)
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_help(self):
        print("""
Available Commands:
  /file <path>     - Analyze and add a code file to knowledge base
  /index <path>    - Index entire directory
  /clear           - Clear conversation history
  /exit            - Exit the assistant

Examples:
  /file app.py
  /index ./src
  Write a function to sort a list
  Review the code in main.py
  Debug this error: NameError: name 'x' is not defined
        """)
    
    def _handle_file(self, file_path: str):
        if os.path.exists(file_path):
            result = self.assistant.add_to_knowledge_base(file_path)
            print(f"📁 {result}")
        else:
            print(f"File not found: {file_path}")
    
    def _handle_index(self, dir_path: str):
        if os.path.exists(dir_path):
            result = self.assistant.index_directory(dir_path)
            print(f"📚 {result}")
        else:
            print(f"Directory not found: {dir_path}")

# Main function for CLI only
def main():
    """Main entry point for CLI"""
    import sys
    
    print("Starting Free AI Coding Assistant...")
    print(f"Using model: {Config.OLLAMA_MODEL}")
    print(f"Ollama host: {Config.OLLAMA_HOST}")
    print("Make sure Ollama is running (Windows: open the Ollama app or run `ollama serve`)")
    
    # Check if Ollama is running
    ok, diag = LocalLLM().diagnose()
    if not ok:
        print(f"\n❌ {diag}")
        sys.exit(1)
    if "not installed" in diag:
        print(f"\n⚠️  {diag}")
        print("You can continue, but requests will fail until the model is pulled.")
    
    cli = CLIInterface()
    cli.run()

if __name__ == "__main__":
    main()