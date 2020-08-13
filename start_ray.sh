#!/bin/bash
. /opt/hlrs/non-spack/system/profile.d/modules.sh
module load python/3.8

RAY=./.local/bin/ray
PORT=6379

if [ $# -eq 0 ]; then
  $RAY start --block --head --redis-port=$PORT
else

  $RAY start --block --address=$1:$PORT
fi
