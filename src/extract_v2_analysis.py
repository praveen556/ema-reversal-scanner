from pathlib import Path
import pandas as pd

# =========================================================
# FIND THE NEWEST V2 BACKTEST WORKBOOK
# =========================================================
project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "output"

files = sorted(
    output_dir.glob("EMA_Backtest_V2_SP500_*.xlsx"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not files:
    raise FileNotFoundError(
        "No EMA_Backtest_V2_SP500_*.xlsx file found in the output folder."
    )

source_file = files[0]

print("=" * 70)
print("V2 BACKTEST - SMALL ANALYSIS EXTRACT")
print("=" * 70)
print(f"Reading: {source_file.name}")

# =========================================================
# READ ONLY THE SMALL SHEETS WE NEED
# =========================================================
sheets_to_extract = [
    "RUN INFO",
    "SUMMARY",
    "HC DIAGNOSTICS",
    "HIGH CONFIDENCE",
]

extracted = {}

for sheet in sheets_to_extract:
    print(f"Reading sheet: {sheet}")
    extracted[sheet] = pd.read_excel(
        source_file,
        sheet_name=sheet,
        engine="openpyxl",
    )

# =========================================================
# SAVE A SMALL WORKBOOK
# =========================================================
output_file = output_dir / "EMA_Backtest_V2_Analysis_Extract.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for sheet_name, df in extracted.items():
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
        )

        ws = writer.sheets[sheet_name]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        for cells in ws.columns:
            max_length = max(
                (
                    len(str(cell.value))
                    for cell in cells
                    if cell.value is not None
                ),
                default=0,
            )
            ws.column_dimensions[cells[0].column_letter].width = min(
                max_length + 3,
                30,
            )

print()
print("=" * 70)
print("EXTRACT COMPLETE")
print("=" * 70)
print(f"Created: {output_file}")
print()
print("Upload this smaller Excel file to ChatGPT:")
print("EMA_Backtest_V2_Analysis_Extract.xlsx")
