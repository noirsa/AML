# AML-MDS notebooks

## YouTube viral classification

Open [youtube_viral_baseline.ipynb](youtube_viral_baseline.ipynb) with the project's `.venv` kernel and run all cells from the project root. It uses all 12,000 synthetic videos, the supplied `viral` label, and all 12 non-view, non-ID predictors. `views` is excluded because it defines the label. Current `likes`, `comments` and `shares` are retained for the current-snapshot task. The notebook documents their synthetic dependence on views.

The notebook includes data checks, a stratified 2/3–1/3 split, training-only EDA, an L2 logistic baseline tuned by five-fold training CV log loss, and a class-prior reference. Results and split membership are saved under `outputs/youtube_viral_baseline/`. Dependencies are pinned in `requirements-youtube-viral.txt`; the version-1 Kaggle archive is cached under `data/youtube_viral/raw/` and downloaded only if absent. Source: [Ved's YouTube viral dataset](https://www.kaggle.com/datasets/vedauv/what-makes-a-yt-video-go-viral-12000-vids), CC0.

## Job Dataset proposal baseline

Open [job_dataset_baseline.ipynb](job_dataset_baseline.ipynb) for the data preparation, leakage checks, training-CV Ridge baseline and actual test results. It contains short operational notes; proposal/report prose is kept outside the notebook. Run from the project root with the `.venv` kernel and the original dataset ZIP.

The previous Dry Bean starter remains below for reference.

# AML-MDS proposal starter: Dry Bean

Open `proposal_dry_bean.ipynb` using the Python environment in `.venv` and run the cells from the project root (`D:\AML`).

If dependencies are missing, install the starter requirements in that environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The notebook fetches UCI dataset 602 using `ucimlrepo`, displays the first rows and source variable descriptions, and saves the unmodified table to `data/dry_bean_raw.csv`. Fetching requires an internet connection. An executed copy retains the output of the successful retrieval; the CSV is a local snapshot for subsequent work.

The requirements cover only these starter cells. Add any further analysis dependencies and record their versions as you extend the notebook.

Dataset source: https://archive.ics.uci.edu/dataset/602/dry+bean

Dataset DOI: https://doi.org/10.24432/C50S4B

Dataset license: CC BY 4.0. Preserve the dataset attribution in submitted work.

The notebook includes an AI-use acknowledgment for the introduction and data-loading code. Review it before submission.
