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
