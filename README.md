# Environmental Sound Classification (ESC-50)

Deep Learning course project. Compares a small CNN trained from scratch with a
fine-tuned pretrained CNN14, with and without augmentation, on ESC-50.

## Structure
- `src/`: shared code (config, preprocessing, data loading)
- `notebooks/`: exploration and experiments
- `tests/`: unit tests
- `figures/`, `results/`, `docs/`: outputs and notes

## Running in Colab
1. Add a Colab secret named `GITHUB_TOKEN`.
2. Run the setup cell at the top of any notebook.

## Data
ESC-50 is not stored in the repo. It is downloaded in the notebooks and cached on Drive.