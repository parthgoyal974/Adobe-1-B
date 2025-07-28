# pdf_to_t5.py
#
# First-time run (internet required):
#   python pdf_to_t5.py "<persona>" "<job>"
#
# Later runs work completely offline.
#
# Requirements:
#   pip install --quiet pymupdf transformers torch

import os
import sys
from pathlib import Path

import fitz                       # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --------------------------------------------------------------------------- #
# 1.  PDF text extraction                                                     #
# --------------------------------------------------------------------------- #
def extract_text(pdf_path: Path) -> str:
    """
    Return the full plain-text content of a single PDF.
    """
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages).strip()


# --------------------------------------------------------------------------- #
# 2.  Prompt template                                                         #
# --------------------------------------------------------------------------- #
def build_prompt(persona: str, job: str, resume_text: str) -> str:
    """
    Compose the final prompt handed to T5.
    """
    return (
        f"You are {persona}.\n"
        f"Job to do by you: {job}.\n\n"
        "Using the info below, write a customised response.\n\n"
        "response:\n"
        f"{resume_text}\n\n"
        "response letter:"
    )


# --------------------------------------------------------------------------- #
# 3.  Model retrieval (download-on-demand)                                    #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 3.  Model retrieval (download-on-demand)                                    #
# --------------------------------------------------------------------------- #
def ensure_model(model_name: str, model_dir: Path):
    """
    Return (tokenizer, model), downloading once if necessary, then
    forcing Transformers into offline mode.
    """
    # --- make sure we always deal with *string* paths -----------------------
    model_dir_str = str(model_dir)          # ← NEW: single source of truth

    if not model_dir.exists():
        print(f"[INFO] Local model not found -> downloading '{model_name}' …")
        model_dir.mkdir(parents=True, exist_ok=True)

        # download directly into model_dir
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  cache_dir=model_dir_str)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                      cache_dir=model_dir_str)

    else:
        print(f"[INFO] Using local model found in {model_dir_str}")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # ---------- ALWAYS pass plain strings + use_fast=False --------------
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_dir_str,
            local_files_only=True,
            use_fast=False
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=model_dir_str,
            local_files_only=True
        )

    # once we have the files, stay offline
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return tokenizer, model



# --------------------------------------------------------------------------- #
# 4.  Main driver                                                             #
# --------------------------------------------------------------------------- #
def main() -> None:
    # ----------------------- argument parsing --------------------------------
    if len(sys.argv) != 3:
        sys.exit("Usage: python pdf_to_t5.py <persona> <job>")

    persona = sys.argv[1]
    job     = sys.argv[2]

    # ----------------------- I/O locations -----------------------------------
    root_dir   = Path(__file__).resolve().parent
    input_dir  = root_dir / "app" / "input"
    output_dir = root_dir / "app" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        sys.exit(f"No PDF files found in {input_dir}")

    # ----------------------- text extraction ---------------------------------
    texts = []
    for pdf in pdf_paths:
        print(f"Extracting text from {pdf.name} …")
        texts.append(extract_text(pdf))

    resume_text = "\n\n".join(texts)          # merge all PDFs into one blob
    prompt      = build_prompt(persona, job, resume_text)

    # ----------------------- model & tokenizer -------------------------------
    model_name = "t5-small"
    model_dir  = root_dir / "models" / model_name
    tokenizer, model = ensure_model(model_name, model_dir)

    device = "cpu"
    model.to(device)

    # ----------------------- encode / generate -------------------------------
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,        # within t5-small limits
    ).to(device)

    generated_ids = model.generate(
        **encoded,
        max_length=256,
        num_beams=5,
        early_stopping=True,
    )

    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # ----------------------- write output ------------------------------------
    persona_file = persona.replace(" ", "_") + ".txt"
    out_path     = output_dir / persona_file

    out_path.write_text(result, encoding="utf-8")
    print(f"\nResult written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
