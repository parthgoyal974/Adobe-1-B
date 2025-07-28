# pdf_to_t5.py
# pip install pymupdf transformers torch --quiet

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
    Return the full plain-text content of a PDF.

    Parameters
    ----------
    pdf_path : Path
        Path to a single PDF file.

    Returns
    -------
    str
        All text contained in the PDF, stripped of leading / trailing blanks.
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

    Feel free to adjust the wording here – the clearer the instructions,
    the better the small T5 model will behave.
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
# 3.  Main driver                                                             #
# --------------------------------------------------------------------------- #
def main() -> None:
    # ----------------------- argument parsing --------------------------------
    if len(sys.argv) < 4:
        sys.exit(
            "Usage: python pdf_to_t5.py <job> <persona> <pdf1> [<pdf2> ...]"
        )

    job     = sys.argv[1]
    persona = sys.argv[2]
    pdf_paths = [Path(p) for p in sys.argv[3:]]

    # Validate files exist
    missing = [str(p) for p in pdf_paths if not p.is_file()]
    if missing:
        sys.exit(f"Error: file(s) not found – {', '.join(missing)}")

    # ----------------------- text extraction ---------------------------------
    texts = []
    for pdf in pdf_paths:
        print(f"Extracting text from {pdf} ...")
        texts.append(extract_text(pdf))

    resume_text = "\n\n".join(texts)      # merge all PDFs into one blob
    prompt      = build_prompt(persona, job, resume_text)

    # ----------------------- model & tokenizer -------------------------------
    model_name = "t5-small"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ----------------------- encode / generate -------------------------------
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,        # keep within t5-small limits
    ).to(device)

    generated_ids = model.generate(
        **encoded,
        max_length=256,
        num_beams=5,
        early_stopping=True,
    )

    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # ----------------------- output ------------------------------------------
    print("\n" + "=" * 60 + "\nGenerated output:\n")
    print(result)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
