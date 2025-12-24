#!/bin/bash

##############################
# User-defined experiment parameters
##############################

exp="bash_exp"        # experiment name used in file names
net="trans"          # workload
batch=64              # batch size
core="polar"          # core architecture

x=4                   # mesh X dimension
y=4                   # mesh Y dimension
stride=2              # stride (divisor of x)

noc_bw=64             # bandwidth
rel_dram_bw=2

cost=1                # cost code (-1 energy, 0 latency, 1 edp)

rounds=10              # number of rounds
gen_IR=0              # generate IR? (0/1)
tech=7

test_custom_only=1
run_init=1
run_ls=1
run_lp=1
run_set=1
run_custom=1
if [ "$test_custom_only" -eq 1 ]; then
    run_init=0
    run_ls=0
    run_lp=0
    run_set=0
fi

custom_tree_file="bash_exp_init_tree.txt"

########################################
# Derive metric name from cost code
########################################

if [ "$cost" -eq -1 ]; then
    metric="energy"
elif [ "$cost" -eq 0 ]; then
    metric="latency"
elif [ "$cost" -eq 1 ]; then
    metric="edp"
else
    metric="custom_cost_$cost"
fi


########################################
# Build output directory
########################################

OUTDIR="./out/${core}/${tech}nm/${rounds}_rounds/x_${x}_y_${y}_s_${stride}_noc_${noc_bw}/rel_dram_bw_${rel_dram_bw}/${net}/batch_${batch}/${metric}"
mkdir -p "$OUTDIR"

echo "Output directory: $OUTDIR"


########################################
# Run the experiment
########################################

echo "Running:"
echo "./build/stschedule --args $exp $net $batch $core $x $y $stride $noc_bw $rel_dram_bw $cost $rounds $tech $run_init $run_ls $run_lp $run_set $run_custom $custom_tree_file $gen_IR"

./build/stschedule --args \
    "$exp" "$net" "$batch" "$core" \
    "$x" "$y" "$stride" "$noc_bw" "$rel_dram_bw" \
    "$cost" "$rounds" "$tech" \
    "$run_init" "$run_ls" "$run_lp" "$run_set" "$run_custom" \
    "$custom_tree_file" "$gen_IR"
    
#"$custom_tree_file" #"$gen_IR"


########################################
# Collect output files
########################################

echo "Copying bash_exp_*.txt files to $OUTDIR"
cp ${exp}_*.txt "$OUTDIR"/

echo "Done."
