#!/bin/bash
set -e
source synthetic/large/settings.sh

echo "MATLAB DIR: ${MATLAB_DIR}"
echo "MATLAB executable: ${MATLAB}"
require_program ${MATLAB}

read_depth=$1
trial=$2
noise_level=$3


require_variable "read_depth" $read_depth
require_variable "trial" $trial
require_variable "noise_level" $noise_level


convert_to_windows() {
	linux_path=$1
	echo "D:/${linux_path#/mnt/d/}"
}


create_config() {
	cfg_path=$1
	seed=$2
	out_path=$(convert_to_windows $3)
	metadata_file=$(convert_to_windows $4)
	counts_file=$(convert_to_windows $5)
	biomass_file=$(convert_to_windows $6)

	cat <<- EOFDOC > $cfg_path
[General]
run_inference = 1
run_simulations = 0
run_linear_stability = 0
run_post_processing = 0

# seed can be empty or a number
seed = ${2}

# Algorithm Options:
# BAL  -- Bayesian adaptive lasso
# BVS  -- Bayesian value select
# MLRR  -- maximum likelihood ridge regression
# MLCRR  -- maximum likelihood constrained ridge regression
algorithm = BVS
output_dir = ${out_path}
# Inference output filename is <algorithm's name>.mat,
# for example MLCRR.mat or BVS.mat. the predictions and post processing
# assume that this is the case.

# metadata in format specified in readme
metadata_file = ${metadata_file}
# biom converted to text, qiime or mothur out
counts_file = ${counts_file}
# biomass file in format specififed in readme
biomass_file = ${biomass_file}

[Parallel]
# Parallelization is of limited use, but can speeed up a small number of
# lengthy functions. particularly, the replicates in Ridge Regression, and
# the linear stability analysis.
cores = ${NUM_CORES}

[Preprocessing]
minMedCount = 10   # minimum median of counts (across all subjects and time-points)
numReplicates = 3   # number of replicates for biomass data
useSplines = 1

[Ridge Regression]
# set normalize counts to false (=0) if the counts data are already normalized
# for the biomass.
normalize_counts = 1
scaling_factor = 1  # if you want to scale the data for a conversion

differentiation = 1  # 1-3, 1=forward, 2=backward, 3=central

# number of groups for k fold cross validation
# if you have 100 samples, and k = 20, there are 20 groups, each with 5 samples
# 19 groups will be used for training the model, 1 for validating
k = 30

# the next three parameters define the space to look for regularization
# parameters. logspace(min, max, N): creates logarithmically spaced vector
# from min to max with N total points
# default (-3, 2, 15) spans 0.001 to 100 in 15 total points.
min = -3
max = 2
N = 15


# By default (=1), the k fold validation will break up each subject's
# data into groups, and the groups span subjects (ie some of both
# subject 1 and subject 2's data will be part of group A.) set mix
# trajectories to false if you have many subjects with fewer time points,
# true if there are fewer subjects but more timepoints
# Generally default (=1) will be acceptable.
mix_trajectories = 1

# parallelized
replicates = 3  # number of different shuffles for the cross fold validation

[Bayesian Lasso]
numIters = 10000   # number of MCMC iterations
numBurnin = 2000   # number of burnin iterations
data_std_init = 10   # initial value for data std
lambda_interact_init = 1.00E+13   # initial value for lambda_interact
gpB_lambda_interact = 1.00E+21   # second gamma prior hyperparameter for lambda_interact
gpA_lambda_interact = 1.00E-09   # first gamma prior hyperparameter for lambda_interact

[Bayesian Select]
numIters = 25000   # number of MCMC iterations
numBurnin = 2500   # number of burnin iterations
data_std_init = 1.00E+05   # initial value for data std
interact_beta_a = 90     #  L * (L-1) strong/weak priors. 0.5 agnostic
interact_beta_b = 0.5     #  0.5/0.1
# following unused for numPerturb = 0
# perturb_beta_a = 0.5
# perturb_beta_b = 0.5

[bayesian spline biomass]
numIters = 10000   # number of MCMC iterations
numBurnin = 2000   # number of burnin iterations
smoothnessOrder = 1   # smoothness order (1st or 2nd degree)
tauScale = 100   # variance on spline coefficients
init_lambda = 1.00E-03   # initial value for lambda parameters
gpA = 1.00E-09   # first gamma parameter for lambda prior
gpB = 1.00E+07   # second gamma paramter for lambda prior

[bayesian spline counts]
numIters = 10000   # number of MCMC iterations
numBurnin = 2000   # number of burnin iterations
smoothnessOrder = 1   # smoothness order (1st or 2nd degree)
tauScale = 100   # variance on spline coefficients
lambda_omega_init = 1.00E-03   # initial value for lambda_omega parameters
gpA_omega = 1.00E-09   # first gamma parameter for lambda_omega prior
gpB_omega = 1.00E+07   # second gamma paramter for lambda_omega prior
eps_a1_init = 1.00E-03   # NBD dispersion parameter a1 initialization
tune_eps_a1_factor = 1.00E+02   # tuning parameter for NBD dispersion parameter a1 (large value = smaller move)
tune_eps_a0_factor = 1.00E+01   # tuning parameter for NBD dispersion parameter a0 (large value = smaller move)
numInitEstimate = 1.50E+02   # number of iterations to do normal estimate during burnin
v_prop = 2.50E-01   # variance parameter for initial normal estimates during burnin

[Simulation]
start_time = 0
end_time = 30
time_step = 0.1
thin_rate = 1
assume_stiff = 1

[linear stability]
# parallelized
sample_step = 100

[Post Processing]
write_parameters = 1
write_cytoscape = 0
write_trajectories = 0
write_stability_analysis = 0

# 0 for don't perform. 1 for perform
# keystone analysis is a subset of stability analysis,
# these are ignored if write_stability_analysis = 0
perform_keystone_analysis = 0
keystone_cutoff = 0.75
EOFDOC
}


dataset=${DATASET_DIR}/data/trial_${trial}/reads_${read_depth}/noise_${noise_level}/subjset.pkl
trial_output_dir=${OUTPUT_DIR}/reads_${read_depth}/trial_${trial}/${noise_level}_noise
inference_out_dir=${trial_output_dir}/mdsine1
mkdir -p $inference_out_dir


seed=0
mdsine_cfg=$inference_out_dir/mdsine.cfg


if [ -f $inference_out_dir/BVS.mat ]; then
	echo "Found output ${inference_out_dir}/BVS.mat. Skipping."
	exit 0
fi


# ======= Run inference
echo "[*] Running mdsine1 inference (reads=${read_depth}, trial=${trial}, noise level=${noise_level})"

echo "[*] Generating configuration..."
metadata=${inference_out_dir}/metadata.txt
counts=${inference_out_dir}/counts.txt
biomass=${inference_out_dir}/biomass.txt
create_config $mdsine_cfg $seed $inference_out_dir $metadata $counts $biomass

echo "[*] Formatting synthetic inputs..."
python synthetic/helpers/create_mdsine1_inputs.py \
-i ${dataset} \
-o ${inference_out_dir} \
-m metadata.txt \
-c counts.txt \
-b biomass.txt \
-t 11

echo "[*] Running matlab implementation..."
cd $MDSINE1_DIR
true_cfg=$(convert_to_windows ${mdsine_cfg})
#${MATLAB} -nosplash -nodesktop -wait < mdsine.m -r "mdsine ${true_cfg} ; quit"
${MATLAB} -nosplash -nodesktop -wait < mdsine.m -r "try, mdsine ${true_cfg}; catch ME, quit; end; quit"
cd -
