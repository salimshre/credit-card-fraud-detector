Installed MiKTeX 25.12, which provides both:

```powershell
pdflatex
xelatex
```

Verified both executables are available at:

```text
C:\Users\StudyAcer\AppData\Local\Programs\MiKTeX\miktex\bin\x64\
```

That path is now in your user `PATH`; reopen PowerShell/VS Code terminal if `pdflatex` is not recognized in an already-open terminal.

I also compiled the documentation successfully. The PDF is here:

[Creditcard.pdf](C:/Users/StudyAcer/OneDrive/Documents/GitHub/credit-card-fraud-detector/Documentation/CreditCard/Creditcard.pdf)

I made one small LaTeX fix while compiling: added `amsmath` to [Creditcard.tex](C:/Users/StudyAcer/OneDrive/Documents/GitHub/credit-card-fraud-detector/Documentation/CreditCard/Creditcard.tex:5). Bibliography resolved, no fatal LaTeX errors remain.