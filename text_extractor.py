from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers

# For folder dialog
import tkinter as tk
from tkinter import filedialog, messagebox


def choose_directory(title: str) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    path_str = filedialog.askdirectory(title=title)
    root.destroy()
    if not path_str:
        return None
    return Path(path_str)


def choose_file(initial_dir: Path, title: str) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    file_str = filedialog.askopenfilename(
        title=title,
        initialdir=str(initial_dir),
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    root.destroy()
    if not file_str:
        return None
    return Path(file_str)


def configure_hf_logging(quiet: bool = True) -> None:
    if not quiet:
        return
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from transformers import logging as tlogging  # type: ignore

        tlogging.set_verbosity_error()
    except Exception:
        pass


def resolve_hf_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


def iter_chunks(text: str, max_chars: int = 800) -> Iterable[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current: List[str] = []
    current_len = 0

    for p in paragraphs:
        if current_len + len(p) + 2 <= max_chars:
            current.append(p)
            current_len += len(p) + 2
        else:
            if current:
                yield "\n\n".join(current)
            if len(p) <= max_chars:
                current = [p]
                current_len = len(p)
            else:
                for i in range(0, len(p), max_chars):
                    yield p[i : i + max_chars]
                current = []
                current_len = 0

    if current:
        yield "\n\n".join(current)


def build_index(
    txt_files: List[Path],
    out_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    hf_token: str | None = None,
    quiet: bool = True,
) -> None:
    configure_hf_logging(quiet=quiet)

    model_kwargs: Dict[str, Any] = {}
    if hf_token:
        model_kwargs["token"] = hf_token
    model = SentenceTransformer(model_name, **model_kwargs)

    chunks: List[str] = []
    meta: List[Dict[str, Any]] = []

    txt_files = sorted(txt_files)
    print(f"Found {len(txt_files)} .txt file(s) for indexing")
    if not txt_files:
        print("No .txt files found; exiting.")
        return

    for txt_path in txt_files:
        raw = txt_path.read_text(encoding="utf-8", errors="ignore")
        book_name = txt_path.stem

        current_page = None
        for block in raw.split("===== PAGE "):
            block = block.strip()
            if not block:
                continue
            if block[0].isdigit():
                header, _, rest = block.partition("=====")
                page_num = header.split("/")[0].strip()
                try:
                    current_page = int(page_num)
                except ValueError:
                    current_page = None
                text_body = rest
            else:
                text_body = block

            for ch in iter_chunks(text_body):
                chunks.append(ch)
                meta.append(
                    {
                        "book": book_name,
                        "source_file": txt_path.name,
                        "page": current_page,
                        "chars": len(ch),
                    }
                )

    print(f"Total chunks: {len(chunks)}")
    if not chunks:
        print("No chunks to embed; exiting.")
        return

    print("Embedding chunks...")
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    out_dir.mkdir(exist_ok=True, parents=True)

    np.save(out_dir / "embeddings.npy", embeddings)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for ch, m in zip(chunks, meta):
            rec = {"text": ch, **m}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    try:
        tk.Tk().withdraw()
        messagebox.showinfo("Embeddings complete", f"Saved index to:\n{out_dir}")
    except Exception:
        pass

    print(f"Saved embeddings to {out_dir / 'embeddings.npy'}")
    print(f"Saved chunk metadata to {out_dir / 'chunks.jsonl'}")
    print("Index ready for retrieval / logic-binding.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build embeddings over extracted book texts.")
    ap.add_argument(
        "--txt_dir",
        type=str,
        default=None,
        help="Directory containing extracted .txt files (if omitted, a folder dialog will appear).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2).",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["all", "single"],
        default=None,
        help="Input mode: 'all' for all .txt files in folder, 'single' for one file.",
    )
    ap.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific .txt file path (used when mode=single).",
    )
    ap.add_argument(
        "--non_recursive",
        action="store_true",
        help="If set, search only top-level directory for .txt files.",
    )
    ap.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face token. If omitted, reads HF_TOKEN env var.",
    )
    ap.add_argument(
        "--verbose_hf",
        action="store_true",
        help="Show Hugging Face / transformers warnings and model load logs.",
    )
    args = ap.parse_args()

    if args.txt_dir:
        txt_dir = Path(args.txt_dir)
    else:
        txt_dir = choose_directory("Select folder containing extracted .txt astrology books")
        if txt_dir is None:
            print("No folder selected. Exiting.")
            return

    if not txt_dir.is_dir():
        raise SystemExit(f"Text directory does not exist: {txt_dir}")

    mode = args.mode
    if mode is None:
        while True:
            choice = input("Choose mode: [A]ll files in folder or [S]ingle file? ").strip().lower()
            if choice in {"a", "all"}:
                mode = "all"
                break
            if choice in {"s", "single"}:
                mode = "single"
                break
            print("Please enter A for all files or S for single file.")

    selected_files: List[Path]
    if mode == "single":
        if args.file:
            selected = Path(args.file)
        else:
            selected = choose_file(txt_dir, "Select ONE .txt file to index")
            if selected is None:
                print("No file selected. Exiting.")
                return
        if not selected.is_file():
            raise SystemExit(f"Selected file does not exist: {selected}")
        if selected.suffix.lower() != ".txt":
            raise SystemExit(f"Selected file is not a .txt file: {selected}")
        selected_files = [selected]
    else:
        search_iter = txt_dir.glob("*.txt") if args.non_recursive else txt_dir.rglob("*.txt")
        selected_files = sorted(p for p in search_iter if p.is_file())

    if not selected_files:
        scope = "top-level" if args.non_recursive else "recursive"
        print(f"Found 0 .txt files in {txt_dir} ({scope} search).")
        print("Tip: choose mode 'single' to pick one file, or verify that .txt files exist.")
        return

    hf_token = resolve_hf_token(args.hf_token)
    if hf_token is None:
        print("HF token not set. Model download may be rate-limited. Set HF_TOKEN to avoid this.")

    build_index(
        selected_files,
        out_dir=txt_dir / ".index",
        model_name=args.model,
        hf_token=hf_token,
        quiet=not args.verbose_hf,
    )


if __name__ == "__main__":
    main()