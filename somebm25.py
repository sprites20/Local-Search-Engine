import os
import subprocess
import whoosh.index as index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import MultifieldParser
from whoosh.scoring import BM25F
from tqdm import tqdm
import fitz  # PyMuPDF for PDFs
from docx import Document  # python-docx for Word files
from bs4 import BeautifulSoup  # BeautifulSoup for HTML
import markdown  # Markdown parsing

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.progressbar import ProgressBar

from kivy.graphics.opengl import glGetString, GL_VERSION

from threading import Thread
from kivy.clock import Clock

def is_opengl_available():
    try:
        version = glGetString(GL_VERSION)
        print(version)
        return version is not None
    except Exception:
        return False

# Function to set up Mesa software rendering
def setup_mesa():
    if is_opengl_available():
        return  # Skip Mesa setup if OpenGL is available
    
    if os.name == "nt":  # Windows
        mesa_path = "mesa"  # Change this to where you placed the Mesa DLLs
        opengl_dll = os.path.join(mesa_path, "opengl32.dll")
        
        if os.path.exists(opengl_dll):
            os.environ["KIVY_GL_BACKEND"] = "sdl2"
            os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "swrast"
            os.environ["PATH"] += f";{mesa_path}"
        else:
            print("Mesa3D not found! OpenGL may not work properly.")
    
    else:  # Linux/macOS
        print("Mesa not found! Installing it...")
        subprocess.run(["sudo", "apt", "install", "-y", "mesa-utils", "libgl1-mesa-glx", "libegl1-mesa"], check=True)
        os.environ["KIVY_GL_BACKEND"] = "sdl2"
        os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "swrast"

# Run Mesa setup before launching Kivy
setup_mesa()

INDEX_DIR = "whoosh_index"

class BM25WhooshSearch:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.index = self.setup_index()

    def setup_index(self, force_update=False):
        if not os.path.exists(INDEX_DIR):
            os.mkdir(INDEX_DIR)
        schema = Schema(
            filepath=ID(stored=True, unique=True),  # Unique identifier (not searchable)
            filepath_search=TEXT(stored=False),  # Searchable version (not stored)
            content=TEXT(stored=False),
            modified=TEXT(stored=True)
        )
        if not index.exists_in(INDEX_DIR):
            idx = index.create_in(INDEX_DIR, schema)
        else:
            idx = index.open_dir(INDEX_DIR)
        if force_update:
            self.update_index(idx)
        return idx

    def get_all_files(self):
        """Recursively collects all supported files in the folder with modification times."""
        file_data = {}
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                file_data[full_path] = str(os.path.getmtime(full_path))  # Store mod time as string
        return file_data

    def extract_text(self, file_path):
        """Extracts text from PDFs, Word docs, HTML, Markdown, text files, and source code."""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                with fitz.open(file_path) as doc:
                    return "\n".join(page.get_text("text") for page in doc)
            elif ext in {".docx"}:
                return "\n".join([p.text for p in Document(file_path).paragraphs])
            elif ext in {".html", ".htm"}:
                with open(file_path, "r", encoding="utf-8") as f:
                    return BeautifulSoup(f.read(), "html.parser").get_text()
            elif ext in {".md"}:
                with open(file_path, "r", encoding="utf-8") as f:
                    return markdown.markdown(f.read())
            elif ext in {".txt", ".csv", ".json", ".xml"}:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif ext in {".py", ".cpp", ".java", ".js", ".c", ".cs", ".html", ".css", ".sh"}:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
                #return self.extract_code(file_path)  # Extract meaningful code
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
        return None
    
    def extract_code(self, file_path):
        """Extracts meaningful lines from code files, ignoring comments and empty lines."""
        code_lines = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith(("#", "//", "/*", "*", "--")):  # Ignore comments
                    code_lines.append(stripped)
        return "\n".join(code_lines)

    def update_index(self, idx):
        """Updates index by adding new/modified files and removing missing ones."""
        writer = idx.writer()
        indexed_files = set()
        files_to_index = self.get_all_files()
        total_files = len(files_to_index)

        with idx.searcher() as searcher:
            for doc in searcher.documents():
                indexed_files.add(doc["filepath"])

        # Add or update files with progress
        for i, (file, mod_time) in enumerate(files_to_index.items(), start=1):
            if file not in indexed_files or mod_time != self.get_file_mod_time(idx, file):
                content = self.extract_text(file)
                if content:
                    writer.update_document(
                        filepath=file,
                        filepath_search=file,
                        content=content,
                        modified=mod_time
                    )

            # Yield progress
            yield (i / total_files) * 100  # Return progress percentage

        # Remove deleted files
        for indexed_file in indexed_files:
            if indexed_file not in files_to_index:
                writer.delete_by_term("filepath", indexed_file)

        writer.commit()
        yield 100  # Ensure progress reaches 100% when done


    def get_file_mod_time(self, idx, filepath):
        """Retrieves the stored modification time of a file from the index."""
        with idx.searcher() as searcher:
            result = searcher.document(filepath=filepath)
            return result["modified"] if result else None

    def search(self, query, top_n=10):
        """Performs a BM25 search on both content and file paths."""
        with self.index.searcher(weighting=BM25F()) as searcher:
            parser = MultifieldParser(["content", "filepath_search"], schema=self.index.schema)
            parsed_query = parser.parse(query)
            results = searcher.search(parsed_query, limit=top_n)

            return [(hit["filepath"], hit.score) for hit in results]

RESULTS_PER_PAGE = 20

class FolderSelectionScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        layout = BoxLayout(orientation='vertical')
        
        # Manual Input for Folder Path
        self.folder_input = TextInput(hint_text='Enter folder path manually', multiline=False)
        self.folder_input.bind(on_text_validate=self.update_folder)
        
        # File Chooser
        filechooser = FileChooserIconView(dirselect=True)
        filechooser.bind(selection=lambda instance, selection: self.update_folder_selection(instance, selection))
        
        # Button to Confirm Selection
        select_button = Button(text='Select Folder', size_hint_y=None, height=50)
        select_button.bind(on_press=self.confirm_selection)
        
        layout.add_widget(self.folder_input)
        layout.add_widget(filechooser)
        layout.add_widget(select_button)
        self.add_widget(layout)
    
    def update_folder(self, instance):
        self.app.selected_folder = self.folder_input.text
    
    def update_folder_selection(self, instance, selection):
        if selection:
            self.app.selected_folder = selection[0]
            self.folder_input.text = selection[0]
            
            # Update the folder label in SearchScreen
            if hasattr(self.app, 'search_screen'):
                self.app.search_screen.folder_label.text = f"Selected Folder: {selection[0]}"

    def confirm_selection(self, instance):
        if not self.app.selected_folder:
            return

        print(f"Selected folder: {self.app.selected_folder}")  # Debugging print

        # Disable UI elements to prevent selection during indexing
        self.folder_input.disabled = True
        instance.disabled = True  # Disable 'Select Folder' button

        # Update the search engine
        self.app.search_engine = BM25WhooshSearch(self.app.selected_folder)
        
        # UI updates
        self.progress_label = Label(text="Indexing Started...")
        self.progress_bar = ProgressBar(max=100, value=0)
        self.add_widget(self.progress_label)
        self.add_widget(self.progress_bar)
            
        def update_progress(progress):
            Clock.schedule_once(lambda dt: setattr(self.progress_bar, "value", progress))
            Clock.schedule_once(lambda dt: setattr(self.progress_label, "text", f"Indexing Progress: {int(progress)}%"))

        def run_indexing():
            for progress in self.app.search_engine.update_index(self.app.search_engine.index):
                Clock.schedule_once(lambda dt, p=progress: update_progress(p))

            # Ensure it reaches 100% and unlock UI when done
            Clock.schedule_once(lambda dt: update_progress(100))
            Clock.schedule_once(lambda dt: unlock_ui())

        def unlock_ui():
            self.remove_widget(self.progress_label)
            self.remove_widget(self.progress_bar)
            self.folder_input.disabled = False
            instance.disabled = False  # Re-enable 'Select Folder' button
            self.app.switch_to_search_screen()

        # Start indexing in a new thread
        indexing_thread = Thread(target=run_indexing, daemon=True)
        indexing_thread.start()

class SearchScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        layout = BoxLayout(orientation='vertical')
        
        # Folder Selection Navigation
        folder_layout = BoxLayout(size_hint_y=None, height=50)
        self.folder_label = Label(text='Selected Folder: temp', size_hint_x=0.8)
        folder_button = Button(text='Change Folder', size_hint_x=0.2)
        folder_button.bind(on_press=self.app.switch_to_folder_selection)
        
        folder_layout.add_widget(self.folder_label)
        folder_layout.add_widget(folder_button)
        layout.add_widget(folder_layout)
        
        # Search Bar
        search_layout = BoxLayout(size_hint_y=None, height=50)
        self.query_input = TextInput(hint_text='Enter search query', multiline=False)
        search_button = Button(text='Search', size_hint_x=None, width=100)
        search_button.bind(on_press=self.perform_search)
        
        search_layout.add_widget(self.query_input)
        search_layout.add_widget(search_button)
        layout.add_widget(search_layout)
        
        # Scrollable Results
        self.results_container = GridLayout(cols=1, size_hint_y=None)
        self.results_container.bind(minimum_height=self.results_container.setter('height'))
        self.scroll = ScrollView()
        self.scroll.add_widget(self.results_container)
        layout.add_widget(self.scroll)
        
        # Pagination
        pagination_layout = BoxLayout(size_hint_y=None, height=50)
        self.prev_button = Button(text='Previous', size_hint_x=0.2)
        self.prev_button.bind(on_press=self.prev_page)
        
        self.page_label = Label(size_hint_x=0.2)
        self.page_input = TextInput(size_hint_x=0.2, text='1', multiline=False)
        self.page_input.bind(on_text_validate=self.select_page)
        
        self.next_button = Button(text='Next', size_hint_x=0.2)
        self.next_button.bind(on_press=self.next_page)
        
        pagination_layout.add_widget(self.prev_button)
        pagination_layout.add_widget(self.page_label)
        pagination_layout.add_widget(self.page_input)
        pagination_layout.add_widget(self.next_button)
        
        layout.add_widget(pagination_layout)
        
        self.add_widget(layout)
    
    def perform_search(self, instance):
        query = self.query_input.text.strip()
        if query:
            self.app.current_query = query
            self.app.current_page = 0
            self.app.results = self.app.search_engine.search(query, top_n=1000)
            self.update_results()
    
    def update_results(self):
        self.results_container.clear_widgets()
        total_pages = max(1, (len(self.app.results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)
        start = self.app.current_page * RESULTS_PER_PAGE
        end = start + RESULTS_PER_PAGE
        page_results = self.app.results[start:end]
        
        for filepath, score in page_results:
            btn = Button(text=f"{filepath} - Score: {score:.4f}", size_hint_y=None, height=30)
            btn.bind(on_press=lambda instance, path=filepath: self.app.show_file_content(path))
            self.results_container.add_widget(btn)
        
        self.page_label.text = f"Page {self.app.current_page + 1} / {total_pages}"
    
    def next_page(self, instance):
        if (self.app.current_page + 1) * RESULTS_PER_PAGE < len(self.app.results):
            self.app.current_page += 1
            self.update_results()
    
    def prev_page(self, instance):
        if self.app.current_page > 0:
            self.app.current_page -= 1
            self.update_results()
    
    def select_page(self, instance):
        try:
            selected_page = int(self.page_input.text) - 1
            total_pages = max(1, (len(self.app.results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)
            if 0 <= selected_page < total_pages:
                self.app.current_page = selected_page
                self.update_results()
        except ValueError:
            pass

class FileContentScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.filepath = None
        
        layout = BoxLayout(orientation='vertical')
        
        self.file_name_label = Label(text='', size_hint_y=0.1)
        self.scroll_view = ScrollView()
        self.content_text = TextInput(text='', readonly=True, size_hint_y=None, height=500)
        self.content_text.bind(minimum_height=self.content_text.setter('height'))
        
        layout.add_widget(self.file_name_label)
        self.scroll_view.add_widget(self.content_text)
        layout.add_widget(self.scroll_view)
        
        button_layout = BoxLayout(size_hint_y=None, height=50)
        
        back_button = Button(text='Back')
        back_button.bind(on_press=self.go_back)
        button_layout.add_widget(back_button)
        
        open_button = Button(text='Open Externally')
        open_button.bind(on_press=self.open_externally)
        button_layout.add_widget(open_button)
        
        show_button = Button(text='Show in File Manager')
        show_button.bind(on_press=self.show_in_file_manager)
        button_layout.add_widget(show_button)
        
        layout.add_widget(button_layout)
        self.add_widget(layout)
    
    def display_content(self, content, filepath):
        self.filepath = filepath
        self.file_name_label.text = filepath
        self.content_text.text = content
    
    def go_back(self, instance):
        self.app.screen_manager.current = 'search'
    
    def open_externally(self, instance):
        if self.filepath:
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(self.filepath)
                elif os.uname().sysname == 'Darwin':  # macOS
                    subprocess.run(['open', self.filepath])
                else:  # Linux
                    subprocess.run(['xdg-open', self.filepath])
            except Exception as e:
                print(f"Error opening file: {e}")
    
    def show_in_file_manager(self, instance):
        if self.filepath:
            folder_path = os.path.dirname(self.filepath)
            try:
                if os.name == 'nt':  # Windows
                    subprocess.run(['explorer', folder_path])
                elif os.uname().sysname == 'Darwin':  # macOS
                    subprocess.run(['open', folder_path])
                else:  # Linux
                    subprocess.run(['xdg-open', folder_path])
            except Exception as e:
                print(f"Error showing file in file manager: {e}")

class SearchApp(App):
    def __init__(self, search_engine, **kwargs):
        super().__init__(**kwargs)
        self.search_engine = search_engine
        self.current_query = ""
        self.current_page = 0
        self.results = []
        self.selected_folder = "temp"
    
    def build(self):
        self.screen_manager = ScreenManager()
        
        self.search_screen = SearchScreen(self, name='search')
        self.folder_selection_screen = FolderSelectionScreen(self, name='folder_selection')
        self.file_content_screen = FileContentScreen(self, name='content')

        self.screen_manager.add_widget(self.search_screen)
        self.screen_manager.add_widget(self.folder_selection_screen)
        self.screen_manager.add_widget(self.file_content_screen)
        
        return self.screen_manager

    def switch_to_folder_selection(self, instance=None):
        self.screen_manager.current = 'folder_selection'

    def switch_to_search_screen(self):
        # Update the folder label in SearchScreen
        self.search_screen.folder_label.text = f"Selected Folder: {self.selected_folder}"
        self.screen_manager.current = 'search'

    def show_file_content(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            content = f"Error reading file: {str(e)}"
        
        self.file_content_screen.display_content(content, filepath)
        self.screen_manager.current = 'content'

if __name__ == '__main__':
    folder_path = "temp"  # Ensure this folder contains files
    search_engine = BM25WhooshSearch(folder_path)
    print("Indexing files...")
    search_engine.update_index(search_engine.index)
    print("Indexing complete!")
    SearchApp(search_engine).run()