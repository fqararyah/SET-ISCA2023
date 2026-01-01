#!/bin/bash

##############################
# Fixed experiment parameters
##############################

exp="bash_exp"          # experiment name used in file names
core="polar"            # core architecture

x=8                     # mesh X dimension
y=8                     # mesh Y dimension
stride=2                # must divide x

cost=-1                  # -1 energy, 0 latency, 1 edp
rounds=100               # number of rounds
gen_IR=0                # generate IR? (0/1)
tech=7

test_custom_only=0
run_init=1
run_ls=1
run_lp=1
run_set=1
run_custom=0
custom_tree_file=""
if [ "$test_custom_only" -eq 1 ]; then
    run_init=0
    run_ls=0
    run_lp=0
    run_set=0
    custom_tree_file="" #add here ...
fi

##############################
# Networks to sweep
##############################

nets=(
"gpt_prefill"
"gpt_decode"
"bert"
"vit"
"deit"
"resnetv2"
"convnext"
)

# nets=(
# "resnet"
# "resnet101"
# "ires"
# "goog"
# "dense"
# "darknet"
# "vgg"
# "trans"
# "trans_cell"
# "pnas"
# "bert"
# "gpt_prefill"
# "gpt_decode"
# "resnetv2"
# "vit"
# "gpt_decode"
# )

##############################
# Batch sizes to sweep
##############################

batches=(1 8 64)

##############################
# NoC + DRAM sweep sets
##############################

noc_bws=(32 64 128)
rel_dram_bws=(1 2 4)

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
# Start parameter sweep
########################################

for net in "${nets[@]}"; do
    for batch in "${batches[@]}"; do
        for noc_bw in "${noc_bws[@]}"; do
            for rel_dram_bw in "${rel_dram_bws[@]}"; do

                echo ""
                echo "===================================================="
                echo " NET=$net | batch=$batch | NOC=$noc_bw | DRAM=$rel_dram_bw"
                echo "===================================================="

                ########################################
                # Output directory
                ########################################

                OUTDIR="./out/${core}/${tech}nm/${rounds}_rounds/x_${x}_y_${y}_s_${stride}_noc_${noc_bw}/rel_dram_bw_${rel_dram_bw}/${net}/batch_${batch}/${metric}"
                mkdir -p "$OUTDIR"

                echo "Output directory: $OUTDIR"

                ########################################
                # Run experiment
                ########################################

                echo "Running:"
                echo "./build/stschedule --args $exp $net $batch $core $x $y $stride $noc_bw $rel_dram_bw $cost $rounds $tech $run_init $run_ls $run_lp $run_set $run_custom $custom_tree_file $gen_IR"

                ./build/stschedule --args \
                    "$exp" "$net" "$batch" "$core" \
                    "$x" "$y" "$stride" "$noc_bw" "$rel_dram_bw" \
                    "$cost" "$rounds" "$tech" \
                    "$run_init" "$run_ls" "$run_lp" "$run_set" "$run_custom" \
                    "$custom_tree_file" "$gen_IR"

                ########################################
                # Copy output
                ########################################

                echo "Copying ${exp}_*.txt → $OUTDIR"
                cp ${exp}_*.txt "$OUTDIR"/

            done
        done
    done
done

echo "===================================================="
echo " SWEEP COMPLETE!"
echo "===================================================="
