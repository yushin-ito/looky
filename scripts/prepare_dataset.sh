#! /bin/bash

#PBS -q gen_S
#PBS -A LOOKY
#PBS -l elapstim_req=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -b 1
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=4.1.8/gcc11.4.0-cuda12.8.1

module load openmpi/$NQSV_MPI_VER

cd $PBS_O_WORKDIR

mpirun ${NQSV_MPIOPTS} -np 1 -npernode 1 \
  --merge-stderr-to-stdout \
  -output-filename logs/ \
  bash -c '
  export HF_HOME="/work/LOOKY/yushin/.cache/huggingface"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export HF_HUB_CACHE="$HF_HOME/hub"

  source .venv/bin/activate

  python scripts/prepare_dataset.py
'