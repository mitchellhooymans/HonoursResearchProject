@echo off
REM Loop through all Jupyter Notebook files in the current directory
for %%f in (*.ipynb) do (
    echo Converting "%%f"...
    jupyter nbconvert --to script "%%f"
)

echo.
echo Done! All notebooks have been converted.
pause