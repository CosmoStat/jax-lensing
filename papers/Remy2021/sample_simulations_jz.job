#!/bin/bash
#SBATCH -A xdy@gpu
#SBATCH --job-name=validation_ODE   # nom du job
#SBATCH --ntasks=1                    # nombre de tâche MPI
#SBATCH --ntasks-per-node=1           # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:1                  # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=10            # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread          # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block    # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00               # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=validation_hmc.out # nom du fichier de sortie
#SBATCH --error=validation_hmc.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -C v100-32g

set -x

cd $WORK/repo/jl/scripts

module purge
module load tensorflow-gpu/py3/2.5.0

for i in {1..10}
do
  srun python ./sample_hmc.py --convergence=../data/ktng/ktng_kappa360v2.fits\
                        --mask=../data/COSMOS/cosmos_full_mask_0.29arcmin360copy.fits\
                        --model_weights=/gpfswork/rech/xdy/commun/Remy2021/score_sn1.0_std0.2/model-final.pckl\
                        --batch_size=10\
                        --initial_step_size=0.013\
                        --min_steps_per_temp=10\
                        --initial_temperature=1.\
                        --gaussian_only=False\
                        --reduced_shear=False\
                        --gaussian_path=../data/ktng/ktng_PS_theory.npy\
                        --gaussian_prior=True\
			--output_file=$i\
                        --output_folder=validation/annealed_hmc\
                        --cosmos_noise_realisation=False\
                        --no_cluster=True\
                        --COSMOS=False
done
