import queue
import threading
from contextlib import redirect_stdout

import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from preprocessing import preprocess_text
from israel import vectorize_documents
from load_dataset import load_dataset
from test_movie_names_with_indices import movies
from advanced_alg import extract_keywords

_cache_lock = threading.Lock()
_dataset_cache = None
_movie_cache = None


class QueueWriter:
    def __init__(self, log_queue):
        self.log_queue = log_queue
        self._buffer = ""

    def write(self, text):
        if not text:
            return
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.log_queue.put(line + "\n")

    def flush(self):
        if self._buffer:
            self.log_queue.put(self._buffer)
            self._buffer = ""


def _get_dataset():
    global _dataset_cache
    with _cache_lock:
        cached = _dataset_cache
    if cached is not None:
        return cached

    df = load_dataset()

    with _cache_lock:
        if _dataset_cache is None:
            _dataset_cache = df
        return _dataset_cache


def _get_movie_data():
    global _movie_cache
    with _cache_lock:
        cached = _movie_cache
    if cached is not None:
        return cached

    df = pd.read_json("datasett/test.jsonl", lines=True)

    with _cache_lock:
        if _movie_cache is None:
            _movie_cache = df
        return _movie_cache


def _is_dataset_cached():
    with _cache_lock:
        return _dataset_cache is not None


def _is_movie_cached():
    with _cache_lock:
        return _movie_cache is not None


def _parse_movie_list(movies_text):
    lines = []
    for line in movies_text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def _parse_movie_index(movie_line):
    if not movie_line:
        return None
    try:
        index_str = movie_line.split("-", 1)[0]
        return int(index_str)
    except ValueError:
        return None


def _jaccard_score(items_a, items_b):
    set_a = {word for word, _ in items_a} if items_a else set()
    set_b = {word for word, _ in items_b} if items_b else set()
    if not set_a and not set_b:
        return None
    return len(set_a & set_b) / len(set_a | set_b)


def run_gui():
    bg_color = "#f4f1ea"
    panel_bg = "#ffffff"
    accent = "#1f3b4d"
    accent_soft = "#dfe7ea"
    text_color = "#2b2b2b"
    muted_text = "#6b6b6b"
    title_font = "Liberation Serif"
    body_font = "Liberation Sans"
    mono_font = "DejaVu Sans Mono"

    root = tk.Tk()
    root.title("Keyword Extraction")
    root.minsize(1020, 700)
    root.configure(bg=bg_color)

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("App.TFrame", background=bg_color)
    style.configure("Panel.TFrame", background=panel_bg)
    style.configure("Panel.TLabel", background=panel_bg, foreground=text_color, font=(body_font, 11))
    style.configure("Section.TLabel", background=panel_bg, foreground=accent, font=(body_font, 12, "bold"))
    style.configure("Panel.TRadiobutton", background=panel_bg, foreground=text_color, font=(body_font, 11))
    style.configure("App.TCombobox", padding=5)
    style.configure("Accent.TButton", background="#d36a4c", foreground="#ffffff", font=(body_font, 11, "bold"), padding=(12, 8))
    style.configure("App.TButton", padding=(12, 8), font=(body_font, 11))
    style.map("Accent.TButton", background=[("active", "#c85f43")], foreground=[("disabled", "#f3e7e2")])
    style.map("Panel.TRadiobutton", background=[("active", panel_bg)])

    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    header = tk.Frame(root, bg=accent)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)

    title_label = tk.Label(
        header,
        text="Keyword Extraction",
        bg=accent,
        fg="#fdf7f0",
        font=(title_font, 22, "bold"),
        padx=18,
        pady=0,
    )
    title_label.grid(row=0, column=0, sticky="w", pady=(12, 0))

    subtitle_label = tk.Label(
        header,
        text="TF-IDF + KeyBERT on movie summaries",
        bg=accent,
        fg=accent_soft,
        font=(body_font, 11),
        padx=18,
        pady=0,
    )
    subtitle_label.grid(row=1, column=0, sticky="w", pady=(0, 12))

    content = ttk.Frame(root, style="App.TFrame", padding=16)
    content.grid(row=1, column=0, sticky="nsew")
    content.columnconfigure(0, weight=0)
    content.columnconfigure(1, weight=1)
    content.rowconfigure(0, weight=1)

    left_panel = ttk.Frame(content, style="Panel.TFrame", padding=14)
    left_panel.grid(row=0, column=0, sticky="nsw")
    left_panel.columnconfigure(0, weight=1)

    right_panel = ttk.Frame(content, style="Panel.TFrame", padding=14)
    right_panel.grid(row=0, column=1, sticky="nsew", padx=(16, 0))
    right_panel.columnconfigure(0, weight=1)
    right_panel.rowconfigure(1, weight=1)

    ttk.Label(left_panel, text="Controls", style="Section.TLabel").grid(row=0, column=0, sticky="w")

    ttk.Label(left_panel, text="Algorithm", style="Panel.TLabel").grid(
        row=1, column=0, sticky="w", pady=(12, 4)
    )

    algorithm_var = tk.StringVar(value="tfidf")
    ttk.Radiobutton(
        left_panel,
        text="TF-IDF",
        value="tfidf",
        variable=algorithm_var,
        style="Panel.TRadiobutton",
    ).grid(row=2, column=0, sticky="w")
    ttk.Radiobutton(
        left_panel,
        text="KeyBERT (Advanced)",
        value="keybert",
        variable=algorithm_var,
        style="Panel.TRadiobutton",
    ).grid(row=3, column=0, sticky="w")
    ttk.Radiobutton(
        left_panel,
        text="Both",
        value="both",
        variable=algorithm_var,
        style="Panel.TRadiobutton",
    ).grid(row=4, column=0, sticky="w")

    ttk.Label(left_panel, text="Movie", style="Panel.TLabel").grid(
        row=5, column=0, sticky="w", pady=(16, 4)
    )

    movie_var = tk.StringVar()
    movie_list = _parse_movie_list(movies)
    movie_combo = ttk.Combobox(
        left_panel,
        textvariable=movie_var,
        values=movie_list,
        state="readonly",
        style="App.TCombobox",
    )
    movie_combo.grid(row=6, column=0, sticky="ew")
    if movie_list:
        movie_combo.current(0)

    ttk.Label(left_panel, text="Actions", style="Panel.TLabel").grid(
        row=7, column=0, sticky="w", pady=(16, 4)
    )

    button_frame = ttk.Frame(left_panel, style="Panel.TFrame")
    button_frame.grid(row=8, column=0, sticky="ew")
    button_frame.columnconfigure(1, weight=1)

    ttk.Label(right_panel, text="Output", style="Section.TLabel").grid(row=0, column=0, sticky="w")

    output_text = tk.Text(
        right_panel,
        wrap="word",
        bg="#fbfaf6",
        fg=text_color,
        insertbackground=text_color,
        font=(mono_font, 11),
        relief="flat",
        borderwidth=0,
    )
    output_text.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
    output_scroll = ttk.Scrollbar(right_panel, orient="vertical", command=output_text.yview)
    output_scroll.grid(row=1, column=1, sticky="ns", pady=(10, 0))
    output_text.configure(yscrollcommand=output_scroll.set, state="disabled")

    metrics_var = tk.StringVar(value="Jaccard (TF-IDF vs KeyBERT): N/A")
    metrics_label = ttk.Label(right_panel, textvariable=metrics_var, style="Panel.TLabel")
    metrics_label.grid(row=2, column=0, sticky="w", pady=(10, 0))

    status_var = tk.StringVar(value="Ready.")
    status_label = tk.Label(
        root,
        textvariable=status_var,
        bg=bg_color,
        fg=muted_text,
        font=(body_font, 10),
        padx=16,
        pady=8,
        anchor="w",
    )
    status_label.grid(row=2, column=0, sticky="ew")

    log_queue = queue.Queue()

    def set_status(text):
        status_var.set(text)

    def set_metrics(text):
        metrics_var.set(text)

    def append_output(text):
        output_text.configure(state="normal")
        output_text.insert("end", text)
        output_text.see("end")
        output_text.configure(state="disabled")

    def clear_output():
        output_text.configure(state="normal")
        output_text.delete("1.0", "end")
        output_text.configure(state="disabled")

    def process_log_queue():
        try:
            while True:
                chunk = log_queue.get_nowait()
                append_output(chunk)
        except queue.Empty:
            pass
        root.after(100, process_log_queue)

    def on_worker_done(error):
        run_button.config(state="normal")
        clear_button.config(state="normal")
        movie_combo.config(state="readonly")
        if error:
            messagebox.showerror("Error", error)
            set_status("Error.")
            return
        set_status("Done.")

    def run_worker(algorithm, movie_index, movie_label):
        writer = QueueWriter(log_queue)
        try:
            with redirect_stdout(writer):
                if _is_dataset_cached():
                    print("Using cached dataset.")
                else:
                    print("Loading dataset and preprocessing if needed...")
                df = _get_dataset()
                print("Dataset ready.")

                if _is_movie_cached():
                    print("Using cached movie summaries.")
                else:
                    print("Loading movie summaries...")
                movie_df = _get_movie_data()
                if movie_index < 0 or movie_index >= len(movie_df):
                    raise ValueError("Movie index is out of range.")

                print(f"\nSelected movie: {movie_label}")
                print("Preprocessing selected movie...")
                summary = movie_df["summary"].iloc[movie_index]
                processed = preprocess_text(summary)
                print("Preprocessing done.\n")

                movie_series = pd.Series(processed)
                combined = pd.concat([df, movie_series], keys=["original", "new"])

                tfidf_items = None
                keybert_items = None

                if algorithm in {"tfidf", "both"}:
                    print("TF-IDF Results:")
                    tfidf_items = vectorize_documents(combined, top_n=10)
                    print("")

                if algorithm in {"keybert", "both"}:
                    print("KeyBERT Results:")
                    keybert_items = extract_keywords(movie_series[0], top_n=10)
                    print("")

                if algorithm == "both":
                    score = _jaccard_score(tfidf_items, keybert_items)
                    if score is None:
                        metrics_text = "Jaccard (TF-IDF vs KeyBERT): N/A"
                    else:
                        metrics_text = f"Jaccard (TF-IDF vs KeyBERT): {score:.3f}"
                    root.after(0, lambda: set_metrics(metrics_text))

            root.after(0, lambda: on_worker_done(None))
        except Exception as exc:
            root.after(0, lambda: on_worker_done(str(exc)))
        finally:
            writer.flush()

    def on_run():
        movie_line = movie_var.get()
        movie_index = _parse_movie_index(movie_line)
        if movie_index is None:
            messagebox.showerror("Selection Required", "Please select a movie.")
            return

        clear_output()
        if algorithm_var.get() == "both":
            set_metrics("Jaccard (TF-IDF vs KeyBERT): running...")
        else:
            set_metrics("Jaccard (TF-IDF vs KeyBERT): N/A (run Both)")
        set_status("Running extraction...")
        run_button.config(state="disabled")
        clear_button.config(state="disabled")
        movie_combo.config(state="disabled")

        worker = threading.Thread(
            target=run_worker,
            args=(algorithm_var.get(), movie_index, movie_line),
            daemon=True,
        )
        worker.start()

    run_button = ttk.Button(button_frame, text="Run Extraction", style="Accent.TButton", command=on_run)
    run_button.grid(row=0, column=0, padx=(0, 8))

    clear_button = ttk.Button(button_frame, text="Clear Output", style="App.TButton", command=clear_output)
    clear_button.grid(row=0, column=1, padx=(0, 8))

    process_log_queue()

    root.mainloop()


if __name__ == "__main__":
    run_gui()