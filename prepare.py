"""Turn the raw scrape into the training set.

Only authors with at least MIN_QUOTES quotes survive, whitespace is normalised,
exact duplicates are dropped. train.py calls prepare() on whatever it loads, so
the committed raw file (100 quotes, 50 authors) trains as 48 quotes, 7 authors.

    uv run prepare.py            # rewrites data/quotes.csv in place: 100 rows -> 48 rows
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data" / "quotes.csv"
MIN_QUOTES = 5


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("text", "author", "tags"):
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df.drop_duplicates(subset=["text"])
    counts = df["author"].value_counts()
    keep = counts[counts >= MIN_QUOTES].index
    return df[df["author"].isin(keep)].reset_index(drop=True)


def main() -> None:
    before = pd.read_csv(DATA)
    after = prepare(before)
    after.to_csv(DATA, index=False)
    print(f"{DATA.name}: {len(before)} rows, {before['author'].nunique()} authors")
    print(f"      -> {len(after)} rows, {after['author'].nunique()} authors (>= {MIN_QUOTES} quotes each)")


if __name__ == "__main__":
    main()
