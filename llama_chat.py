import tkinter as tk
from tkinter import messagebox, filedialog, ttk, simpledialog, scrolledtext
import os
import json
import datetime
import subprocess

# ----------------------------
# Persistent Session Memory
# ----------------------------
MEMORY_DIR = "./session_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

def load_session(project_name):
    path = os.path.join(MEMORY_DIR, f"{project_name}_session.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"history": []}

def save_session(project_name, message, response):
    session = load_session(project_name)
    timestamp = datetime.datetime.now().isoformat()
    session["history"].append({"timestamp": timestamp, "user": message, "ai": response})
    path = os.path.join(MEMORY_DIR, f"{project_name}_session.json")
    with open(path, "w") as f:
        json.dump(session, f, indent=2)

# ----------------------------
# Placeholder for Llama3
# ----------------------------
def llama3_generate(prompt, project_name=None):
    """
    Replace with your local Llama3 inference call.
    """
    response = f"# AI suggestion for prompt:\n{prompt[:300]}...\n"
    if project_name:
        save_session(project_name, prompt, response)
    return response

# ----------------------------
# Project Management
# ----------------------------
PROJECTS_DIR = "./projects"
os.makedirs(PROJECTS_DIR, exist_ok=True)

def create_project(project_name):
    project_path = os.path.join(PROJECTS_DIR, project_name)
    os.makedirs(project_path, exist_ok=True)
    return project_path

def list_project_files(project_name):
    project_path = os.path.join(PROJECTS_DIR, project_name)
    if not os.path.exists(project_path):
        return []
    return [f for f in os.listdir(project_path) if f.endswith(".py")]

def load_project_file(project_name, filename):
    path = os.path.join(PROJECTS_DIR, project_name, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return ""

def save_project_file(project_name, filename, content):
    project_path = os.path.join(PROJECTS_DIR, project_name)
    os.makedirs(project_path, exist_ok=True)
    with open(os.path.join(project_path, filename), "w") as f:
        f.write(content)

# ----------------------------
# Real-Time Code Assistance
# ----------------------------
def code_assist(project_name, current_code, cursor_context):
    prompt = f"""
You are a real-time AI coding assistant.
Project: {project_name}

Current code around cursor:
{cursor_context}

Full current code:
{current_code}

Provide the next lines of code or edits with correct indentation.
Return only code.
"""
    return llama3_generate(prompt, project_name=project_name)

# ----------------------------
# Project-Wide AI Search & Refactor
# ----------------------------
def project_ai_search(project_name, query):
    results = {}
    for filename in list_project_files(project_name):
        code = load_project_file(project_name, filename)
        prompt = f"""
Project-wide search query: "{query}"
File: {filename}

Return the code snippet matching the query with line numbers.
"""
        response = llama3_generate(prompt, project_name=project_name)
        results[filename] = response
    return results

def project_ai_refactor(project_name, instruction):
    results = {}
    for filename in list_project_files(project_name):
        code = load_project_file(project_name, filename)
        prompt = f"""
Project: {project_name}
File: {filename}
Instruction: {instruction}

Suggest refactor or improvements for the code. Return full code with proper indentation.
"""
        refactored = llama3_generate(prompt, project_name=project_name)
        results[filename] = refactored
        save_project_file(project_name, filename, refactored)
    return results

# ----------------------------
# Run Python File and capture output
# ----------------------------
def run_python_file(project_name, filename):
    path = os.path.join(PROJECTS_DIR, project_name, filename)
    if not os.path.exists(path):
        return "File does not exist."
    try:
        result = subprocess.run(
            ["python3", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        output = f"--- Output ---\n{result.stdout}\n--- Errors ---\n{result.stderr}"
    except Exception as e:
        output = f"Execution failed: {str(e)}"
    return output

# ----------------------------
# Auto-Completion Feature
# ----------------------------
class AutoCompletePopup:
    def __init__(self, text_widget, project_name):
        self.text_widget = text_widget
        self.project_name = project_name
        self.popup = None
        self.suggestions = []
        self.selected_index = 0

    def show(self, suggestions):
        if not suggestions:
            self.hide()
            return
        self.suggestions = suggestions
        self.selected_index = 0

        bbox = self.text_widget.bbox("insert")
        if bbox:
            x, y, _, _ = bbox
            x += self.text_widget.winfo_rootx()
            y += self.text_widget.winfo_rooty() + 20

            if self.popup:
                self.popup.destroy()
            self.popup = tk.Toplevel(self.text_widget)
            self.popup.wm_overrideredirect(True)
            self.popup.geometry(f"+{x}+{y}")

            self.listbox = tk.Listbox(self.popup, width=50, height=min(6, len(suggestions)))
            self.listbox.pack()
            for item in suggestions:
                self.listbox.insert(tk.END, item)
            self.listbox.select_set(0)
            self.listbox.bind("<Return>", self.select)
            self.listbox.bind("<Escape>", lambda e: self.hide())
            self.listbox.bind("<Double-Button-1>", self.select)

    def hide(self):
        if self.popup:
            self.popup.destroy()
            self.popup = None

    def select(self, event=None):
        if self.suggestions:
            suggestion = self.suggestions[self.selected_index]
            self.text_widget.insert("insert", suggestion)
        self.hide()

def get_autocomplete_suggestions(project_name, code_context):
    prompt = f"""
Project: {project_name}
Code context around cursor:
{code_context}

Suggest next lines of code or completions. Return only code suggestions as separate lines.
"""
    response = llama3_generate(prompt, project_name=project_name)
    suggestions = [line for line in response.split("\n") if line.strip()]
    return suggestions

def bind_autocomplete(text_widget, project_name):
    popup = AutoCompletePopup(text_widget, project_name)

    def on_key(event):
        if event.keysym in ("Up", "Down", "Return", "Escape"):
            return
        code_context = text_widget.get("1.0", "insert")
        suggestions = get_autocomplete_suggestions(project_name, code_context)
        popup.show(suggestions)

    text_widget.bind("<KeyRelease>", on_key)

# ----------------------------
# Tkinter IDE UI
# ----------------------------
class Llama3IDE:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultimate Llama3 IDE - AI Assistant")
        self.root.geometry("1400x900")

        self.project_name_var = tk.StringVar()

        # Top Frame: Project controls
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(top_frame, text="Project Name:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        tk.Entry(top_frame, textvariable=self.project_name_var, width=30).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Load/Create Project", command=self.load_project).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Project Search", command=self.project_search).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Project Refactor", command=self.project_refactor).pack(side=tk.LEFT, padx=5)

        # Left Frame: File Browser
        self.left_frame = tk.Frame(root, width=200)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(self.left_frame, text="Project Files", font=("Arial", 12, "bold")).pack(pady=5)
        self.file_listbox = tk.Listbox(self.left_frame)
        self.file_listbox.pack(expand=True, fill=tk.Y, padx=5, pady=5)
        self.file_listbox.bind("<<ListboxSelect>>", self.open_selected_file)
        tk.Button(self.left_frame, text="New File", command=self.create_new_file).pack(pady=5)

        # Right Frame: Tabbed Code Editor
        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.tab_control = ttk.Notebook(self.right_frame)
        self.tab_control.pack(expand=True, fill=tk.BOTH)

    def load_project(self):
        project_name = self.project_name_var.get().strip()
        if not project_name:
            messagebox.showwarning("Input Needed", "Enter a project name.")
            return
        create_project(project_name)
        self.refresh_file_list()

    def refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        project_name = self.project_name_var.get().strip()
        for f in list_project_files(project_name):
            self.file_listbox.insert(tk.END, f)

    def open_selected_file(self, event):
        selection = self.file_listbox.curselection()
        if not selection:
            return
        filename = self.file_listbox.get(selection[0])
        self.open_file_tab(filename)

    def create_new_file(self):
        filename = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python Files","*.py")])
        if filename:
            project_name = self.project_name_var.get().strip()
            save_project_file(project_name, os.path.basename(filename), "")
            self.refresh_file_list()
            self.open_file_tab(os.path.basename(filename))

    def open_file_tab(self, filename):
        project_name = self.project_name_var.get().strip()
        content = load_project_file(project_name, filename)

        tab = tk.Frame(self.tab_control)
        self.tab_control.add(tab, text=filename)

        code_text = tk.Text(tab, wrap=tk.NONE)
        code_text.insert("1.0", content)
        code_text.pack(expand=True, fill=tk.BOTH)

        # Bind AI Auto-Completion
        bind_autocomplete(code_text, project_name)

        bottom_frame = tk.Frame(tab)
        bottom_frame.pack(fill=tk.X, pady=5)

        cursor_context_input = tk.Text(bottom_frame, wrap=tk.NONE, height=4)
        cursor_context_input.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)

        # Output console
        output_console = scrolledtext.ScrolledText(tab, height=10, bg="#1e1e1e", fg="#d4d4d4")
        output_console.pack(expand=False, fill=tk.X, padx=5, pady=5)
        output_console.insert(tk.END, "# Output console\n")

        def get_ai_suggestion():
            current_code = code_text.get("1.0", tk.END)
            cursor_context = cursor_context_input.get("1.0", tk.END)
            response = code_assist(project_name, current_code, cursor_context)
            code_text.insert(tk.END, "\n" + response)

        def copy_ai_suggestion():
            self.root.clipboard_clear()
            self.root.clipboard_append(code_text.get("1.0", tk.END))
            messagebox.showinfo("Copied", "Code copied to clipboard!")

        def save_file():
            save_project_file(project_name, filename, code_text.get("1.0", tk.END))
            messagebox.showinfo("Saved", f"{filename} saved successfully.")

        def run_file():
            save_file()  # Save before running
            output = run_python_file(project_name, filename)
            output_console.delete("1.0", tk.END)
            output_console.insert(tk.END, output)

        tk.Button(bottom_frame, text="AI Suggestion", command=get_ai_suggestion).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_frame, text="Copy Code", command=copy_ai_suggestion).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_frame, text="Save File", command=save_file).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_frame, text="Run File", command=run_file).pack(side=tk.LEFT, padx=5)

    # ----------------------------
    # Project-wide AI Search
    # ----------------------------
    def project_search(self):
        project_name = self.project_name_var.get().strip()
        if not project_name:
            messagebox.showwarning("Input Needed", "Enter project name.")
            return
        query = simpledialog.askstring("Project Search", "Enter search query:")
        if not query:
            return
        results = project_ai_search(project_name, query)
        result_window = tk.Toplevel(self.root)
        result_window.title(f"Search Results for '{query}'")
        text = tk.Text(result_window)
        text.pack(expand=True, fill=tk.BOTH)
        for file, snippet in results.items():
            text.insert(tk.END, f"--- {file} ---\n{snippet}\n\n")

    # ----------------------------
    # Project-wide AI Refactor
    # ----------------------------
    def project_refactor(self):
        project_name = self.project_name_var.get().strip()
        if not project_name:
            messagebox.showwarning("Input Needed", "Enter project name.")
            return
        instruction = simpledialog.askstring("Project Refactor", "Enter refactor instruction:")
        if not instruction:
            return
        results = project_ai_refactor(project_name, instruction)
        messagebox.showinfo("Refactor Complete", f"Refactor applied to {len(results)} files.")

# ----------------------------
# Run IDE
# ----------------------------
root = tk.Tk()
ide = Llama3IDE(root)
root.mainloop()
