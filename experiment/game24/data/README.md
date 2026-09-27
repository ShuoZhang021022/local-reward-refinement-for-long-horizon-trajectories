# Game24 source data

`24.csv` is an unchanged copy of the ToT repository's public dataset:

- Upstream revision: `733b009f627f8e5c81c3e5461391d3aa3e0dd18f`
- Source: https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/733b009f627f8e5c81c3e5461391d3aa3e0dd18f/src/tot/data/24/24.csv
- SHA256: `b9f12b3e36d987a3c714c4cef17d89a137d7c59da26532fcfb93b4821d8111b5`
- Size: 48,235 bytes; 1,362 puzzle rows.

The loader verifies these exact bytes before creating any split. Git newline
conversion is disabled for this file. Do not resave it in a spreadsheet editor.
The original metadata columns are preserved, but only `Puzzles` is supplied to
the task environment. The original zero-based indices 900 through 999 are held
out for testing.
