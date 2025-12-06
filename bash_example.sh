#!/bin/bash

##############################
# User-defined experiment parameters
##############################

exp="bash_exp"        # experiment name used in file names
net="vgg"          # workload
batch=64              # batch size
core="polar"          # core architecture

x=4                   # mesh X dimension
y=4                   # mesh Y dimension
stride=2              # stride (divisor of x)
bw=24                 # bandwidth
cost=1                # cost code (-1 energy, 0 latency, 1 edp)

round=10              # number of rounds
gen_IR=0              # generate IR? (0/1)


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

OUTDIR="./out/${core}/x_${x}_y_${y}_s_${stride}_bw_${bw}/${net}/batch_${batch}/${metric}"
mkdir -p "$OUTDIR"

echo "Output directory: $OUTDIR"


########################################
# Run the experiment
########################################

echo "Running:"
echo "./build/stschedule --args $exp $net $batch $core $x $y $stride $bw $cost $round $gen_IR"

./build/stschedule --args \
    "$exp" "$net" "$batch" "$core" \
    "$x" "$y" "$stride" "$bw" \
    "$cost" "$round" "$gen_IR"


########################################
# Collect output files
########################################

echo "Copying bash_exp_*.txt files to $OUTDIR"
cp ${exp}_*.txt "$OUTDIR"/

echo "Done."
