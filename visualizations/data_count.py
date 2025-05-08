import tqdm
import pathlib
import pandas as pd

def summarize_csv_rows(directory: str = ".") -> pd.DataFrame:
    """
    Scan `directory` for CSV files, count rows in each, and return a summary DataFrame.
    """
    path = pathlib.Path(directory)
    csv_files = sorted(path.glob("*.csv"))                 # change pattern if you need recursion

    summaries = []
    for f in tqdm.tqdm(csv_files):
        try:
            n_rows = sum(1 for _ in f.open()) - 1          # ‑1 to exclude header row
            # If you prefer pandas for counting:
            # n_rows = pd.read_csv(f, usecols=[0]).shape[0]
            summaries.append({"file": f.name, "rows": n_rows})
        except Exception as e:
            summaries.append({"file": f.name, "rows": None, "error": str(e)})

    summary_df = pd.DataFrame(summaries)
    return summary_df

if __name__ == "__main__":
    df = summarize_csv_rows("/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/csvs/")        # or any path you like
    print("\nRow counts per CSV file\n" + "-"*30)
    print(df.to_string(index=False))
