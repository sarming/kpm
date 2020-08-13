#!/bin/bash

{
  read head
  pbs_tmrsh $head ./start_ray.sh &
  sleep 5
  while read worker; do
    pbs_tmrsh $worker ./start_ray.sh $head &
  done
} <$PBS_NODEFILE

PORT=6379
export RAY_ADDRESS=$head:$PORT

. /opt/hlrs/non-spack/system/profile.d/modules.sh
module load python/3.8

python kpm_benchmark.py
