#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL

rm -r ~/.cache
rm -r __pycache__
python main.py
git add .
git commit -m "Finish lab4 job"
git push
