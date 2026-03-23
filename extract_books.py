from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, List

from pypdf import PdfReader  # pip install pypdf

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore

# For folder dialog
import tkinter as tk
from tkinter import filedialog, messagebox


def choose_directory(title: str) -> Optional[Path]:
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    path_str = filedialog.askdirectory(title=title)
    root.destroy()
    if not path_str:
        return None
    return Path(path_str)


def choose_pdf_files(title: str, initial_dir: Optional[Path] = None) -> List[Path]:
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    selected = filedialog.askopenfilenames(
        title=title,
        initialdir=str(initial_dir) if initial_dir else None,
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
    )
    root.destroy()
    return [Path(p) for p in selected if p]


def ocr_page(pdf_path: Path, page_index: int, dpi: int = 300, lang: str = "eng") -> str:
    if fitz is None or Image is None or pytesseract is None:
        return ""

    zoom = dpi / 72.0
    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(page_index)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return (pytesseract.image_to_string(img, lang=lang) or "").strip()
    finally:
        doc.close()


def ocr_dependencies_ready() -> bool:
    if fitz is None or Image is None or pytesseract is None:
        return False
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def configure_tesseract_cmd(cli_tesseract_cmd: Optional[str] = None) -> None:
    if pytesseract is None:
        return

    candidates: List[str] = []
    if cli_tesseract_cmd:
        candidates.append(cli_tesseract_cmd)

    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd:
        candidates.append(env_cmd)

    which_cmd = shutil.which("tesseract")
    if which_cmd:
        candidates.append(which_cmd)

    candidates.extend(
        [
            r"B:\n8n\astro\ocr\tes\tesseract.exe",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"B:\Tesseract-OCR\tesseract.exe",
        ]
    )

    for cmd in candidates:
        if cmd and Path(cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = cmd
            return


def extract_pdf(
    pdf_path: Path,
    out_dir: Path,
    use_ocr: bool = False,
    ocr_lang: str = "eng",
    ocr_dpi: int = 300,
) -> tuple[Path, int, int, int]:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    text_pages = 0
    ocr_pages = 0
    total_chars = 0

    total_pages = len(reader.pages)
    for i, page in enumerate(reader.pages):
        parts.append(f"\n\n===== PAGE {i + 1} / {total_pages} =====\n\n")
        try:
            txt = page.extract_text() or ""
        except Exception as e:
            txt = f"[EXTRACTION_ERROR] {type(e).__name__}: {e}\n"

        if txt.strip():
            text_pages += 1
            total_chars += len(txt.strip())
        elif use_ocr:
            ocr_txt = ocr_page(pdf_path, i, dpi=ocr_dpi, lang=ocr_lang)
            if ocr_txt:
                txt = ocr_txt + "\n"
                ocr_pages += 1
                total_chars += len(ocr_txt)

        parts.append(txt)

    out_path = out_dir / (pdf_path.stem + ".txt")
    out_path.write_text("".join(parts), encoding="utf-8", errors="ignore")

    if text_pages == 0 and ocr_pages == 0:
        print(
            f"Warning: {pdf_path.name} has no machine-readable text layer. "
            "Use --ocr and install PyMuPDF + Pillow + pytesseract + Tesseract OCR."
        )
    else:
        print(f"    Text pages: {text_pages}, OCR pages: {ocr_pages}, chars: {total_chars}")

    return out_path, text_pages, ocr_pages, total_chars


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract text from selected PDFs or from all PDFs in a directory.")
    ap.add_argument(
        "--src",
        type=str,
        default=None,
        help="Source directory with PDF books (CLI folder mode).",
    )
    ap.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="One or more specific PDF files (CLI file mode).",
    )
    ap.add_argument(
        "--non_recursive",
        action="store_true",
        help="In folder mode, scan only top-level directory (default is recursive).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for .txt files (default: <base>/.extracted).",
    )
    ap.add_argument(
        "--ocr",
        action="store_true",
        help="Force OCR fallback during extraction pass.",
    )
    ap.add_argument(
        "--no_auto_ocr",
        action="store_true",
        help="Disable automatic OCR retry when normal extraction is empty/too small.",
    )
    ap.add_argument(
        "--ocr_lang",
        type=str,
        default="eng",
        help="Tesseract language code for OCR (default: eng).",
    )
    ap.add_argument(
        "--ocr_dpi",
        type=int,
        default=300,
        help="Render DPI for OCR pages (default: 300).",
    )
    ap.add_argument(
        "--min_chars",
        type=int,
        default=50,
        help="Minimum extracted chars required to accept non-OCR output (default: 50).",
    )
    ap.add_argument(
        "--tesseract_cmd",
        type=str,
        default=None,
        help="Optional full path to tesseract executable.",
    )
    args = ap.parse_args()

    configure_tesseract_cmd(args.tesseract_cmd)

    pdfs: List[Path] = []
    base_dir: Path

    # CLI file mode
    if args.files:
        pdfs = [Path(p) for p in args.files]
        missing = [p for p in pdfs if not p.is_file()]
        if missing:
            raise SystemExit(f"These files do not exist: {', '.join(str(p) for p in missing)}")
        base_dir = pdfs[0].parent

    # CLI folder mode
    elif args.src:
        src_dir = Path(args.src)
        if not src_dir.is_dir():
            raise SystemExit(f"Source directory does not exist: {src_dir}")
        search_iter = src_dir.glob("*.pdf") if args.non_recursive else src_dir.rglob("*.pdf")
        pdfs = sorted(p for p in search_iter if p.is_file())
        base_dir = src_dir

    # GUI mode: single dialog where user can pick one or multiple files
    else:
        pdfs = choose_pdf_files("Select one or multiple PDF books")
        if not pdfs:
            print("No files selected. Exiting.")
            return
        base_dir = pdfs[0].parent

    # Choose output dir
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = base_dir / ".extracted"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdfs:
        if args.src:
            scope = "top-level" if args.non_recursive else "recursive"
            print(f"No PDFs found in {base_dir} ({scope} search)")
        else:
            print("No PDFs selected.")
        return

    print(f"Found {len(pdfs)} PDF(s)")

    if (args.ocr or not args.no_auto_ocr) and not ocr_dependencies_ready():
        print("OCR requested but dependencies are missing.")
        print("Install: pip install pymupdf pillow pytesseract")
        print("Also install Tesseract OCR engine and add it to PATH.")

    for pdf in pdfs:
        out_path, text_pages, ocr_pages, total_chars = extract_pdf(
            pdf,
            out_dir,
            use_ocr=args.ocr,
            ocr_lang=args.ocr_lang,
            ocr_dpi=args.ocr_dpi,
        )
        if not args.no_auto_ocr and not args.ocr and total_chars < args.min_chars:
            if ocr_dependencies_ready():
                print(
                    f"    Auto OCR triggered for {pdf.name}: low text output "
                    f"({total_chars} chars, text_pages={text_pages})."
                )
                out_path, text_pages, ocr_pages, total_chars = extract_pdf(
                    pdf,
                    out_dir,
                    use_ocr=True,
                    ocr_lang=args.ocr_lang,
                    ocr_dpi=args.ocr_dpi,
                )
            else:
                print(
                    f"    Auto OCR needed for {pdf.name} but OCR dependencies are missing. "
                    "Install PyMuPDF, Pillow, pytesseract and Tesseract OCR."
                )
        print(f"Extracted: {pdf.name} -> {out_path}")

    try:
        tk.Tk().withdraw()
        messagebox.showinfo("Extraction complete", f"Extracted {len(pdfs)} PDFs to:\n{out_dir}")
    except Exception:
        pass  # ignore GUI errors in headless mode

    print("Done.")


if __name__ == "__main__":
    main()