#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL

python main.py
git add .
git commit -m "Finish job"
git push
