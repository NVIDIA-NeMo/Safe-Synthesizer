# Tutorials

Interactive Jupyter notebook tutorials for NeMo Safe Synthesizer.

## Available Tutorials

_No tutorials have been added yet._

## Adding a Tutorial

To add a new tutorial:

1. Create a Jupyter notebook (`.ipynb`) in the `docs/tutorials/` directory
2. Add it to the `nav` section in `mkdocs.yml` under **Tutorials**
3. The notebook will be automatically rendered as a documentation page

!!! tip
    Notebooks are rendered with `mkdocs-jupyter`. Cell outputs are included as-is (notebooks are **not** re-executed during the docs build). Make sure to run your notebook and save it with outputs before committing.

## Guidelines

- Use clear markdown cells to explain each step
- Include expected outputs so readers can follow along without running the notebook
- Keep notebooks focused on a single topic or workflow
- Name files descriptively, e.g., `basic_pipeline.ipynb`, `custom_evaluation.ipynb`
