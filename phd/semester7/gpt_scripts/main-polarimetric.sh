#!/bin/bash
#
# We are looping over the image files and call the polarimetric decompositions SLURM scripts
# In this script we are waiting to last file generated to copy on siimon5 external disk drive

# First we are doing this in a sequentially manner

# Variable configurations
gpt=~/tools/esa-snap10.0/bin/gpt
base_path=~/data/cimat/dataset-sentinel
source=SLC
dest=POL
temp=temp1
set -x
set -e
for fname in $(ls -1 $base_path/$source); do
  name=${fname%%.*}
  echo "Processing $name in bash script"
  # Unzip product and make temp directory
  if [ ! -d $base_path/$source/$name.SAFE ]; then
    unzip -qo $base_path/$source/$fname -d $base_path/$source/
  fi
  mkdir -p $base_path/$temp/$name
  # Apply orbit file (we need internet access)
  if [ ! -f $base_path/$temp/$name/pol_01.dim ]; then
    $gpt scripts/polarimetric_01.xml -SsourceProduct=$base_path/$source/$name.SAFE -t $base_path/$temp/$name/pol_01.dim
  fi
  # Run slurm polarimetric decompositions script (on C0 cpu nodes)
  sbatch run-polarimetric.slurm $fname
  # Move result to siimon5 (we are implementing an active waiting using sleep)
  while [ ! -f $base_path/$dest/$name/pol_10.dim ]; do
    # Sleep
    echo "$base_path/$dest/$name result not created, waiting 30m..."
    sleep 30m
  done
  echo "$base_path/$dest/$name created, proceeding to move to siimon5"
  # Once that result is created we move it to siimon5
  scp -P 2235 $base_path/$dest/$name/pol_10.dim manuelsuarez@siimon5.cimat.mx:/home/mariocanul/image_storage/ssh_sharedir/POL/$name.dim
  scp -r -P 2235 $base_path/$dest/$name/pol_10.data manuelsuarez@siimon5.cimat.mx:/home/mariocanul/image_storage/ssh_sharedir/POL/$name.data
  # Remove temporary files 01-10
  #
  for step in 01 02 03 04 05 06 07 08 09 10
  do
    if [[ $step == "02" || $step == "03" || $step == "04" ]]; then
      for n in 1 2 3
      do
        if test -d $base_path/$temp/$name/pol_${step}_iw${n}.data; then
          rm -rf $base_path/$temp/$name/pol_${step}_iw${n}.data
        fi
        if test -f $base_path/$temp/$name/pol_${step}_iw${n}.dim; then
          rm $base_path/$temp/$name/pol_${step}_iw${n}.dim
        fi
      done
    else
      if test -d $base_path/$temp/$name/pol_${step}.data; then
        rm -rf $base_path/$temp/$name/pol_${step}.data
      fi
      if test -f $base_path/$temp/$name/pol_${step}.dim; then
        rm $base_path/$temp/$name/pol_${step}.dim
      fi
    fi
  done
  # Remove unziped product
  if test -d $base_path/$source/$name.SAFE; then
    rm -rf $base_path/$source/$name.SAFE
  fi
  # Proceed with next file
done
