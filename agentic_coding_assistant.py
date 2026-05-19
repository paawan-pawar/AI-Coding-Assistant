# agentic_coding_assistant.py
import os
import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import ast
import difflib

# Install required packages:
# pip install openai langchain langgraph python-dotenv gitpython watchdog

import openai
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from git import Repo
import watchdog.events
import watchdog.observers

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    EXPLANATION = "explanation"

@dataclass
class Task:
    id: str
    type: TaskType
    description: str
    context: Dict[str, Any]
    status: str = "pending"
    result: Optional[Any] = None

@dataclass
class CodeContext:
    file_path: str
    content: str
    language: str
    dependencies: List[str]
    imports: List[str]
    functions: List[str]
    classes: List[str]

class CodeAnalyzer:
    """Analyzes code structure and dependencies"""
    
    def __init__(self):
        self.supported_languages = ["python", "javascript", "typescript"]
    
    def analyze(self, file_path: str) -> CodeContext:
        with open(file_path, 'r') as f:
            content = f.read()
        
        ext = file_path.split('.')[-1]
        language = self._detect_language(ext)
        
        if language == "python":
            return self._analyze_python(file_path, content)
        elif language in ["javascript", "typescript"]:
            return self._analyze_js(content)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def _detect_language(self, ext: str) -> str:
        mapping = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript'
        }
        return mapping.get(ext, 'unknown')
    
    def _analyze_python(self, file_path: str, content: str) -> CodeContext:
        try:
            tree = ast.parse(content)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module)
            
            return CodeContext(
                file_path=file_path,
                content=content,
                language="python",
                dependencies=imports,
                imports=imports,
                functions=functions,
                classes=classes
            )
        except Exception as e:
            print(f"Error analyzing Python file: {e}")
            return CodeContext(
                file_path=file_path,
                content=content,
                language="python",
                dependencies=[],
                imports=[],
                functions=[],
                classes=[]
            )
    
    def _analyze_js(self, content: str) -> CodeContext:
        # Simplified JS analysis - in production, use a proper parser
        imports = []
        functions = []
        classes = []
        
        lines = content.split('\n')
        for line in lines:
            if 'import' in line:
                imports.append(line.strip())
            if 'function' in line or '=>' in line:
                functions.append(line.strip())
            if 'class' in line:
                classes.append(line.strip())
        
        return CodeContext(
            file_path="",
            content=content,
            language="javascript",
            dependencies=imports,
            imports=imports,
            functions=functions,
            classes=classes
        )

class ToolRegistry:
    """Manages tools available to the agent"""
    
    @staticmethod
    @tool
    def read_file(file_path: str) -> str:
        """Read the contents of a file"""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    
    @staticmethod
    @tool
    def write_file(file_path: str, content: str) -> str:
        """Write content to a file"""
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    @staticmethod
    @tool
    def run_command(command: str) -> str:
        """Run a shell command"""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout if result.stdout else result.stderr
        except Exception as e:
            return f"Error running command: {e}"
    
    @staticmethod
    @tool
    def search_code(pattern: str, directory: str = ".") -> str:
        """Search for code patterns using grep"""
        try:
            result = subprocess.run(
                f"grep -r '{pattern}' {directory}",
                shell=True,
                capture_output=True,
                text=True
            )
            return result.stdout if result.stdout else "No matches found"
        except Exception as e:
            return f"Error searching code: {e}"
    
    @staticmethod
    @tool
    def git_commit(message: str) -> str:
        """Create a git commit"""
        try:
            repo = Repo('.')
            repo.index.commit(message)
            return f"Committed with message: {message}"
        except Exception as e:
            return f"Error committing: {e}"

class AgenticCodingAssistant:
    """Main AI coding assistant with agentic capabilities"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.llm = ChatOpenAI(api_key=api_key, model=model, temperature=0.2)
        self.analyzer = CodeAnalyzer()
        self.tools = [
            ToolRegistry.read_file,
            ToolRegistry.write_file,
            ToolRegistry.run_command,
            ToolRegistry.search_code,
            ToolRegistry.git_commit
        ]
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.agent = self._create_agent()
        self.workflow = self._create_workflow()
    
    def _create_agent(self):
        """Create the LangChain agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI coding assistant with agentic capabilities. 
            You can read/write files, run commands, search code, and perform git operations.
            
            Your goal is to help users with coding tasks autonomously. Break down complex 
            tasks into steps, use available tools when needed, and provide explanations 
            for your actions.
            
            When writing code:
            - Follow best practices and coding standards
            - Add appropriate error handling
            - Include comments for complex logic
            - Consider edge cases
            
            For code review:
            - Check for bugs, security issues, and performance problems
            - Suggest improvements
            - Verify best practices
            
            Be proactive and autonomous while keeping the user informed."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def _create_workflow(self):
        """Create the workflow graph for complex tasks"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("analyze", self._analyze_task)
        workflow.add_node("plan", self._plan_steps)
        workflow.add_node("execute", self._execute_step)
        workflow.add_node("review", self._review_output)
        workflow.add_node("refine", self._refine_output)
        
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "plan")
        workflow.add_conditional_edges(
            "plan",
            self._should_execute,
            {
                "execute": "execute",
                "end": END
            }
        )
        workflow.add_edge("execute", "review")
        workflow.add_conditional_edges(
            "review",
            self._should_refine,
            {
                "refine": "refine",
                "end": END
            }
        )
        workflow.add_edge("refine", "review")
        
        return workflow.compile()
    
    async def process_request(self, request: str, context: Optional[Dict] = None) -> str:
        """Process a user request with agentic workflow"""
        
        # Detect task type
        task_type = self._detect_task_type(request)
        
        # Build context
        full_context = context or {}
        full_context["request"] = request
        full_context["task_type"] = task_type
        
        # Run workflow
        state = AgentState(
            task=Task(
                id=os.urandom(8).hex(),
                type=task_type,
                description=request,
                context=full_context
            ),
            current_step=0,
            steps=[],
            outputs=[]
        )
        
        result = await self.workflow.ainvoke(state)
        
        return result["outputs"][-1] if result["outputs"] else "Task completed"
    
    def _detect_task_type(self, request: str) -> TaskType:
        """Detect the type of task from the request"""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ["write", "create", "implement", "generate"]):
            return TaskType.CODE_GENERATION
        elif any(word in request_lower for word in ["review", "check", "analyze"]):
            return TaskType.CODE_REVIEW
        elif any(word in request_lower for word in ["fix", "debug", "error", "bug"]):
            return TaskType.DEBUGGING
        elif any(word in request_lower for word in ["refactor", "improve", "optimize"]):
            return TaskType.REFACTORING
        elif any(word in request_lower for word in ["test", "unit test"]):
            return TaskType.TESTING
        elif any(word in request_lower for word in ["document", "docstring", "comment"]):
            return TaskType.DOCUMENTATION
        else:
            return TaskType.EXPLANATION
    
    def _analyze_task(self, state: AgentState) -> AgentState:
        """Analyze the task and gather necessary context"""
        task = state.task
        
        # If there's a file mentioned, analyze it
        if "file" in task.context:
            try:
                code_context = self.analyzer.analyze(task.context["file"])
                task.context["code_context"] = code_context
            except Exception as e:
                task.context["analysis_error"] = str(e)
        
        return state
    
    def _plan_steps(self, state: AgentState) -> AgentState:
        """Plan the steps needed to complete the task"""
        task = state.task
        
        # Use LLM to plan steps
        plan_prompt = f"""
        Task: {task.description}
        Task Type: {task.type.value}
        Context: {task.context}
        
        Create a step-by-step plan to accomplish this task. Return as JSON array of steps.
        Each step should have: action, description, and required files.
        """
        
        # For simplicity, we'll create basic steps
        # In production, use LLM to generate detailed plans
        steps = [
            {"action": "analyze", "description": "Understand the current codebase", "files": []},
            {"action": "implement", "description": "Implement the required changes", "files": []},
            {"action": "test", "description": "Test the implementation", "files": []},
            {"action": "document", "description": "Document the changes", "files": []}
        ]
        
        state.steps = steps
        return state
    
    def _execute_step(self, state: AgentState) -> AgentState:
        """Execute the current step using the agent"""
        current_step = state.steps[state.current_step]
        
        # Execute step using the agent
        response = self.agent.invoke({
            "input": f"Execute step: {current_step['description']}\nContext: {state.task.context}",
            "chat_history": self.memory.chat_memory.messages
        })
        
        state.outputs.append(response["output"])
        self.memory.chat_memory.add_user_message(current_step["description"])
        self.memory.chat_memory.add_ai_message(response["output"])
        
        state.current_step += 1
        return state
    
    def _review_output(self, state: AgentState) -> AgentState:
        """Review the output for quality"""
        last_output = state.outputs[-1] if state.outputs else ""
        
        review_prompt = f"""
        Review the following output for quality, correctness, and best practices:
        
        {last_output}
        
        Provide a review with any issues found and suggestions for improvement.
        """
        
        # In production, use LLM for review
        # For now, just pass through
        state.task.context["review"] = "Output looks good"
        
        return state
    
    def _should_execute(self, state: AgentState) -> str:
        """Determine if we should execute or end"""
        return "execute" if state.current_step < len(state.steps) else "end"
    
    def _should_refine(self, state: AgentState) -> str:
        """Determine if we should refine the output"""
        # In production, check review results
        return "end"
    
    def _refine_output(self, state: AgentState) -> AgentState:
        """Refine the output based on review"""
        # Implementation for refining output
        return state

class AgentState:
    """State for the workflow graph"""
    def __init__(self, task: Task, current_step: int, steps: List, outputs: List):
        self.task = task
        self.current_step = current_step
        self.steps = steps
        self.outputs = outputs

class FileWatcher(watchdog.events.FileSystemEventHandler):
    """Watch for file changes and trigger assistant actions"""
    
    def __init__(self, assistant: AgenticCodingAssistant):
        self.assistant = assistant
    
    def on_modified(self, event):
        if not event.is_directory:
            print(f"File modified: {event.src_path}")
            # Could trigger automatic analysis or actions

# Example usage and CLI interface
class CodingAssistantCLI:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        self.assistant = AgenticCodingAssistant(self.api_key)
        self.watcher = None
    
    async def run(self):
        print("🚀 Agentic AI Coding Assistant")
        print("Type 'exit' to quit, 'watch' to watch files, 'help' for commands\n")
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower() == 'watch':
                    self._start_watching()
                else:
                    print("\n🤖 Assistant: ", end="")
                    response = await self.assistant.process_request(user_input)
                    print(response)
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_help(self):
        print("""
        Available commands:
        - 'exit': Exit the assistant
        - 'watch': Watch for file changes
        - 'help': Show this help message
        
        You can also ask:
        - "Write a function to calculate fibonacci numbers"
        - "Review the code in app.py"
        - "Debug the error in main.py"
        - "Add unit tests for utils.py"
        """)
    
    def _start_watching(self):
        """Start watching for file changes"""
        if self.watcher:
            print("Already watching files")
            return
        
        event_handler = FileWatcher(self.assistant)
        self.watcher = watchdog.observers.Observer()
        self.watcher.schedule(event_handler, '.', recursive=False)
        self.watcher.start()
        print("Watching for file changes...")

# Example of extended capabilities
class AutonomousCodingAgent:
    """Extended agent with autonomous capabilities"""
    
    def __init__(self, assistant: AgenticCodingAssistant):
        self.assistant = assistant
        self.active_projects = {}
    
    async def create_project(self, project_name: str, description: str):
        """Create a new project autonomously"""
        prompt = f"""
        Create a new project called '{project_name}'.
        Description: {description}
        
        Create the necessary file structure, write initial code, and set up the project.
        Follow best practices and include proper documentation.
        """
        
        response = await self.assistant.process_request(prompt)
        self.active_projects[project_name] = {"status": "created", "description": description}
        return response
    
    async def refactor_codebase(self, project_path: str, refactor_type: str):
        """Refactor an entire codebase"""
        prompt = f"""
        Refactor the codebase in {project_path} for {refactor_type}.
        Analyze all files, identify areas for improvement, and apply refactoring.
        Ensure no functionality is broken and maintain backward compatibility.
        """
        
        return await self.assistant.process_request(prompt)
    
    async def generate_tests(self, file_path: str):
        """Generate comprehensive tests for a file"""
        prompt = f"""
        Generate comprehensive unit tests for {file_path}.
        Include edge cases, error scenarios, and maintain high coverage.
        Use appropriate testing framework for the language.
        """
        
        return await self.assistant.process_request(prompt)

# Main entry point
async def main():
    """Main entry point for the coding assistant"""
    
    # Initialize the assistant
    assistant_cli = CodingAssistantCLI()
    
    # Run the CLI
    await assistant_cli.run()

if __name__ == "__main__":
    asyncio.run(main())