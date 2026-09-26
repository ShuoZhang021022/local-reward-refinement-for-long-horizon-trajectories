# LaTeX source

The main.tex file was adapted from the supplied GeneLi_Template/main.tex.
It retains arxiv_style.tex, macros.tex, project-macros.tex, and refs.bib
as project inputs. The title and date are set in main.tex.

The local copies omit two unused template dependencies: the blindtext
package import and the prettyref helper block. Indicator functions in
the paper use the math bold 1 symbol so compilation does not require
the missing bbm font files. These changes do not alter the method equations.

Build from this directory with a TeX Live installation:

    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex

The published English PDF is at
../output/pdf/two_step_gate_grpo_en.pdf.
