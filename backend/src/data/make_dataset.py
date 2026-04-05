import os

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def main():
    raw_path = os.path.join(ROOT_DIR, "data", "external", "bank-full.csv")
    df = pd.read_csv(raw_path)

    # Encode target
    df["y_numeric"] = df["y"].map({"no": 0, "yes": 1})
    df = df.drop(columns=["y"])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["y_numeric"])

    processed_dir = os.path.join(ROOT_DIR, "data", "processed")
    train_df.to_csv(os.path.join(processed_dir, "bank_train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "bank_test.csv"), index=False)
    test_df.drop(columns=["y_numeric"]).to_csv(os.path.join(processed_dir, "bank_test_sample.csv"), index=False)

    print(f"Saved {len(train_df)} rows to bank_train.csv")
    print(f"Saved {len(test_df)} rows to bank_test.csv (with labels)")
    print(f"Saved {len(test_df)} rows to bank_test_sample.csv (no labels, for frontend upload)")


if __name__ == "__main__":
    main()
