#!/bin/bash
#SBATCH --output=test.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-0:30:00

module purge
module load python/3.7.0

# Order of arguments: parameter_type, parameter, chem_property, run_label
# parameter type options: 'pop_size', 'selection_method', 'mutation_rate', or 'elitism_perc'
# parameter: the value to change the paraemter type to
# chem_property options: 'polar', 'opt_bg', or 'solv_eng'
# run_label options: 'A', 'B', 'C', 'D', or 'E'


python GA_main.py elitism_perc 0.8 solv_eng E