import os
import whoosh.index as index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F
from tqdm import tqdm
import time

INDEX_DIR = "whoosh_index"

class BM25WhooshSearch:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.index = self.setup_index()

    def setup_index(self):
        """Creates or loads a Whoosh index."""
        if not os.path.exists(INDEX_DIR):
            os.mkdir(INDEX_DIR)

        schema = Schema(filepath=ID(stored=True, unique=True), content=TEXT(stored=False), modified=TEXT(stored=True))

        if not index.exists_in(INDEX_DIR):
            idx = index.create_in(INDEX_DIR, schema)
        else:
            idx = index.open_dir(INDEX_DIR)

        self.update_index(idx)
        return idx

    def get_all_text_files(self):
        """Recursively collects all text files in the folder and subfolders with modification time."""
        file_data = {}
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                file_data[full_path] = str(os.path.getmtime(full_path))  # Store mod time as string
        return file_data

    def update_index(self, idx):
        """Updates index by adding new/modified files and removing missing ones."""
        writer = idx.writer()
        indexed_files = set()
        files_to_index = self.get_all_text_files()

        with idx.searcher() as searcher:
            # Check existing files in the index
            for doc in searcher.documents():
                indexed_files.add(doc["filepath"])

        # Add or update files
        for file, mod_time in tqdm(files_to_index.items(), desc="Updating Index"):
            if file not in indexed_files or mod_time != self.get_file_mod_time(idx, file):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        content = f.read()
                        writer.update_document(filepath=file, content=content, modified=mod_time)
                except Exception as e:
                    print(f"Skipping {file}: {e}")

        # Remove deleted files
        for indexed_file in indexed_files:
            if indexed_file not in files_to_index:
                writer.delete_by_term("filepath", indexed_file)

        writer.commit()

    def get_file_mod_time(self, idx, filepath):
        """Retrieves the stored modification time of a file from the index."""
        with idx.searcher() as searcher:
            result = searcher.document(filepath=filepath)
            return result["modified"] if result else None

    def search(self, query, top_n=10):
        """Performs a BM25 search."""
        with self.index.searcher(weighting=BM25F()) as searcher:
            parser = QueryParser("content", schema=self.index.schema)
            parsed_query = parser.parse(query)
            results = searcher.search(parsed_query, limit=top_n)

            return [(hit["filepath"], hit.score) for hit in results]

if __name__ == "__main__":
    folder_path = "temp"  # Change this to your folder
    search_engine = BM25WhooshSearch(folder_path)

    while True:
        query = input("\nEnter search query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        results = search_engine.search(query)
        print("\nTop results:")
        for filepath, score in results:
            print(f"{filepath} - Score: {score:.4f}")
