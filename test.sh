#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL

python test.py
git add .
git commit -m "Finish test job"
git push
