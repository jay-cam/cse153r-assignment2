## Project Structure

This project contains separate task notebooks during development. Each task has an `.ipynb` notebook with saved outputs and a `.py` export for GitHub readability.

### Task 1: Symbolic Unconditioned Generation

- Notebook: `task1_symbolic_unconditioned.ipynb`
- GitHub-readable export: `task1_symbolic_unconditioned.py`
- Optional HTML preview: `task1_symbolic_unconditioned.html`
- Presentation section: 0:00–14:36 in the video

This notebook loads the POP909 dataset, extracts the `MELODY` tracks, trains a trigram Markov baseline and an LSTM next-token model, generates MIDI outputs, and evaluates the generated melodies using loss, perplexity, generated statistics, and listening discussion.

### Task 2: <Task Name>

- Notebook: `task2_<name>.ipynb`
- GitHub-readable export: `task2_<name>.py`
- Optional HTML preview: `task2_<name>.html`
- Presentation section: 14:36–25:00 in the video

<Brief description of the second task.>

## Viewing Notes

GitHub may fail to preview `.ipynb` notebooks with saved outputs. If this happens, open the `.py` export on GitHub to review the code and markdown, or download/open the `.ipynb` file in VS Code, Jupyter, or Google Colab.

The `.ipynb` notebooks are the primary development files because they preserve outputs, tables, plots, and generated results used in the video presentation. The `.py` files are included as readable exports for GitHub preview. The `.html` files are browser-viewable notebook exports and are useful for checking how the final submission will look.

For the final Gradescope submission, the task notebooks will be combined into one final notebook and exported as `workbook.html`.

The final submission will also include:

- `video_url.txt`
- `symbolic_unconditioned.mid`
- `<second task generated output file>`
- `video_url.txt`
