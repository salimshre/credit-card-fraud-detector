# Project Documentation

This directory contains the project report, diagrams, screenshots, and research references for the Credit Card Fraud Detection dashboard.

## Documentation Structure

| Path | Purpose |
| --- | --- |
| `CreditCard/` | Main project report source, compiled PDF, report assets, and bibliography. |
| `CreditCard/Creditcard.tex` | Main LaTeX source file for the complete report. |
| `CreditCard/Creditcard.pdf` | Generated report PDF ready for review or submission. |
| `CreditCard/CreditCard.bib` | Bibliography entries used by the LaTeX report. |
| `CreditCard/assets/` | Logo, screenshots, diagrams, and visual assets used in the report. |
| `CreditCard/build/` | Generated LaTeX auxiliary files and build outputs. |
| `CreditCard/reference/` | Reference PDF used for report front-page formatting. |
| `ResearchPaper/` | Supporting research papers used during literature review. |
| `archive/` | Older documentation drafts and reference material kept for traceability. |
| `fraud_shield_architecture_overview.svg` | Root-level architecture diagram draft. |
| `PredictionWorkflowFlowchart.svg` | Root-level prediction workflow diagram draft. |

## Main Report Contents

The main report is maintained in `CreditCard/Creditcard.tex` and compiled to `CreditCard/Creditcard.pdf`.

| Chapter | Topic |
| --- | --- |
| Front Matter | Declaration, acknowledgement, abstract, approvals, table of contents, list of figures, and list of tables. |
| Chapter 1 | Introduction, problem statement, objectives, features, significance, scope, and limitations. |
| Chapter 2 | Literature review, machine learning techniques, research gaps, and summary. |
| Chapter 3 | System analysis, requirements, feasibility, and ethical/security considerations. |
| Chapter 4 | Methodology, dataset, feature engineering, model training, prediction pipeline, and risk scoring. |
| Chapter 5 | System design, architecture, use case design, data flow, API design, persistence, and UI screenshots. |
| Chapter 6 | Implementation details for authentication, validation, encoding, behavior analysis, prediction, alerts, CSV detection, and reporting. |
| Chapter 7 | Testing, results, evaluation metrics, sample outputs, issues, and discussion. |
| Chapter 8 | Conclusion and future enhancements. |
| Chapter 9 | Installation and user guide. |

## Build The PDF

Run these commands from `Documentation/CreditCard`:

```powershell
pdflatex -interaction=nonstopmode -aux-directory=build Creditcard.tex
bibtex build/Creditcard
pdflatex -interaction=nonstopmode -aux-directory=build Creditcard.tex
pdflatex -interaction=nonstopmode -aux-directory=build Creditcard.tex
```

The compiled PDF is written to:

```text
Documentation/CreditCard/Creditcard.pdf
```

## Recommended Workflow

1. Edit `Documentation/CreditCard/Creditcard.tex`.
2. Add or update report images inside `Documentation/CreditCard/assets/`.
3. Add literature sources to `Documentation/CreditCard/CreditCard.bib`.
4. Compile the PDF from `Documentation/CreditCard`.
5. Review `Documentation/CreditCard/Creditcard.pdf`.

## Notes

- Keep generated files in `CreditCard/build/` where possible.
- Keep older drafts in `archive/` instead of mixing them with active report files.
- Keep research PDFs in `ResearchPaper/` so the report source remains focused.
