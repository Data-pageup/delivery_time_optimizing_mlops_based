from pathlib import Path

project_name = "food-delivery-mlops"

folders = [
    "data/raw",
    "data/processed",
    "src/data",
    "src/features",
    "src/models",
    "src/pipelines",
    "src/utils",
    "artifacts",
    "mlruns",
    "notebooks",
    "tests"
]

for folder in folders:
    Path(f"{project_name}/{folder}").mkdir(parents=True, exist_ok=True)

Path(f"{project_name}/app.py").touch()
Path(f"{project_name}/streamlit_app.py").touch()

print("Project structure created successfully!")