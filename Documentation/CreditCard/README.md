# Credit Card Documentation

This folder contains the LaTeX report for the Credit Card Fraud Detection project.

## Structure

| Path | Purpose |
| --- | --- |
| `Creditcard.tex` | Main LaTeX report source |
| `CreditCard.bib` | Bibliography database |
| `Creditcard.pdf` | Compiled report for submission |
| `assets/` | Images, logo, and screenshots used by the report |
| `assets/screenshots/` | Dashboard screenshots included in Chapter 5 |
| `build/` | LaTeX auxiliary files generated during compilation |
| `reference/` | Reference front-page PDF used for formatting |
| `archive/` | Older documentation files kept for reference |

## Compile

Run from this directory:

```powershell
pdflatex -interaction=nonstopmode -aux-directory=build Creditcard.tex
```
