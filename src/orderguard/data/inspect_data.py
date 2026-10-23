from pathlib import Path

import pandas as pd

# Locate the project root regardless of the terminal's directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}. "
            "Download and extract the Olist dataset first."
        )

    table_summaries = []
    column_profiles = []

    for csv_path in csv_files:
        print(f"Inspecting: {csv_path.name}")

        # Process one table at a time to avoid loading every table
        # into memory simultaneously.
        df = pd.read_csv(csv_path, low_memory=False)

        total_cells = df.shape[0] * df.shape[1]
        missing_cells = int(df.isna().sum().sum())

        table_summaries.append(
            {
                "table": csv_path.name,
                "rows": len(df),
                "columns": len(df.columns),
                "duplicate_rows": int(df.duplicated().sum()),
                "missing_cells": missing_cells,
                "missing_percentage": (
                    100 * missing_cells / total_cells if total_cells else 0.0
                ),
            }
        )

        for column in df.columns:
            missing_count = int(df[column].isna().sum())

            column_profiles.append(
                {
                    "table": csv_path.name,
                    "column": column,
                    "dtype": str(df[column].dtype),
                    "missing_count": missing_count,
                    "missing_percentage": (
                        100 * missing_count / len(df) if len(df) else 0.0
                    ),
                }
            )

        print(f"  Shape: {df.shape}")
        print(f"  Missing cells: {missing_cells:,}")
        print()

        del df

    table_report = pd.DataFrame(table_summaries)
    column_report = pd.DataFrame(column_profiles)

    table_report.to_csv(
        REPORT_DIR / "data_inventory.csv",
        index=False,
    )

    column_report.to_csv(
        REPORT_DIR / "column_profile.csv",
        index=False,
    )

    print("Inspection complete.")
    print(table_report.to_string(index=False))
    print("\nReports saved to:")
    print(f"  {REPORT_DIR / 'data_inventory.csv'}")
    print(f"  {REPORT_DIR / 'column_profile.csv'}")


if __name__ == "__main__":
    main()
