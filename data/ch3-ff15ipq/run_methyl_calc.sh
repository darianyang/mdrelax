#!/bin/bash
# run steps for methyl relax calc

# original args
#time python ../compute_tcfs.py --xtc sim1.xtc --tpr sim1.tpr --gmx gmx --out tcf --lblocks_bb 1000000 --e 1000000 --b 0
#time python ../specdens_mapping.py --quadric quadric_diffusion --pdbinertia pdbinertia --b 0 \
#--e 1000000 --lblocks_m 10000 --lblocks_bb 1000000 --stau 10000 --trajname sim --wD 145.858415 --ct_lim 2 --i tcf

# 10ns version
#time python ../compute_tcfs.py --xtc sim1.xtc --tpr sim1.tpr --gmx gmx --out tcf --lblocks_bb 10000 --e 10000 --b 0
#python ../specdens_mapping.py --quadric quadric_diffusion --pdbinertia pdbinertia --b 0 \
#--e 10000 --lblocks_m 10000 --lblocks_bb 10000 --stau 10000 --trajname sim --wD 145.858415 --ct_lim 2 --i tcf 

# using only the 10ns traj
python ../compute_tcfs.py --xtc sim1-10ns.xtc --tpr sim1.tpr --gmx gmx --out tcf --lblocks_bb 10000 --e 10000 --b 0
