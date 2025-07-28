T5-powered PDF ⇨ custom-response generator
------------------------------------------

This repo contains `analyzer.py` (formerly `pdf_to_t5.py`).
It extracts text from one or more PDF files and feeds it, together with your
instructions, to a **T5-small** language model to create a customised reply.

Quick start
-----------

1. Install the dependencies (one-off):

```bash
pip install pymupdf transformers torch
```

2. Run the script with **all input supplied as positional command-line
arguments**:

```bash
python .\analyzer.py "<job>" "<persona description>" <pdf1> [<pdf2> ...]
```


Arguments in order
------------------

| Position | `sys.argv` index | Example value | Meaning |
| :-- | :-- | :-- | :-- |
| 1st | `sys.argv` | `"Travel Planner"` | **Job** – the role or task you want the AI to perform |
| 2nd | `sys.argv` | `"Plan a 4-day trip for 10 college friends"` | **Persona / detailed instruction** |
| 3rd-N | `sys.argv[3+]` | `"South of France - Cities.pdf"`, more PDFs optional | **One or more PDF files** whose text will be used |

Example
-------

```bash
python .\analyzer.py \
  "Travel Planner" \
  "Plan a 4-day trip for 10 college friends" \
  "South of France - Cities.pdf"
```

The script will:

1. Extract all text from *South of France - Cities.pdf*
2. Build a prompt that says you are a Travel Planner and must plan a 4-day trip for 10 college friends
3. Send the prompt to T5-small and print the generated itinerary/response to your terminal.
