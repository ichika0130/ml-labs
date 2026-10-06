# ML Systems Labs

Coursework for the ML Systems lab course (textbook: Reddi, *Introduction to Machine Learning Systems*).

## Setup

Python 3.11.8:

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Layout

- `lab01/` … `lab11/` — each has `report.md`, `src/`, `results/`
- `project/` — final project (`README.md`, `src/`, `results/`, `data_card.md`, `model_card.md`)
- `data/` — datasets, git-ignored; scripts download them if missing
