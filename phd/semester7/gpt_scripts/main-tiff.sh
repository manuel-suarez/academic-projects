#!/bin/bash
#
# We are looping over the image files and call the polarimetric decompositions SLURM scripts
# In this script we are waiting to last file generated to copy on siimon5 external disk drive

# First we are doing this in a sequentially manner

# Variable configurations
gpt=~/tools/esa-snap10.0/bin/gpt
dataset=cimat
base_path=~/data/cimat/dataset-$dataset
source=products
dest=tiff
temp=temp3
num=$1
set -x
set -e
mkdir -p $base_path/$source
mkdir -p $base_path/$temp
# Download list of files to process
scp -P 2235 manuelsuarez@siimon5.cimat.mx:~/data/cimat/dataset-${dataset}/${source}_list${num}.txt $base_path/${source}_list${num}.txt
for fname in $(cat $base_path/${source}_list${num}.txt); do
  name=${fname%%.*}
  ext=${fname##*.}
  echo "Processing $name in bash script"
  echo "Extension $ext"
  # Transfer product from siimon5 GRD product directory
  scp -P 2235 manuelsuarez@siimon5.cimat.mx:~/data/cimat/dataset-${dataset}/$source/$fname $base_path/$source/$fname
  # Unzip product and make temp directory
  if [ $ext == "zip" ]; then
    if [ ! -d $base_path/$source/$name.SAFE ]; then
      unzip -qo $base_path/$source/$fname -d $base_path/$source/
    fi
  fi
  mkdir -p $base_path/$temp/$name
  # Apply orbit file (we need internet access)
  if [ ! -f $base_path/$temp/$name/tiff_01.dim ]; then
    # If product is Envisat (N1 extension) apply operator directly on product
    if [ $ext == "N1" ]; then
      $gpt scripts/tiff_01.xml -SsourceProduct=$base_path/$source/$fname -t $base_path/$temp/$name/tiff_01.dim
    else
      $gpt scripts/tiff_01.xml -SsourceProduct=$base_path/$source/$name.SAFE -t $base_path/$temp/$name/tiff_01.dim
    fi
  fi
  # Run slurm tiff band extraction script (on C0 cpu nodes)
  sbatch run-tiff.slurm $base_path $source $dest $temp $fname
  # Move result to siimon5 (we are implementing an active waiting using sleep)
  while [ ! -f $base_path/$dest/${name}.tif ]; do
    # Sleep
    echo "$base_path/$dest/$name result not created, waiting 30m..."
    sleep 30m
  done
  echo "$base_path/$dest/$name created, proceeding to move to siimon5"
  sleep 5m
  # Once that result is created we move it to siimon5
  scp -P 2235 $base_path/$dest/${name}.tif manuelsuarez@siimon5.cimat.mx:~/data/cimat/dataset-${dataset}/$dest/${name}.tif
  #scp -r -P 2235 $base_path/$dest/$name/${name}_VH.tif manuelsuarez@siimon5.cimat.mx:~/data/cimat/dataset-noaa/sentinel1/TIFF/${name}_VH.tif
  # Remove temporary files 01-05
  #
  for step in 01 02 03 04 05
  do
    if test -d $base_path/$temp/$name/tiff_${step}.data; then
      rm -rf $base_path/$temp/$name/tiff_${step}.data
    fi
    if test -f $base_path/$temp/$name/tiff_${step}.dim; then
      rm $base_path/$temp/$name/tiff_${step}.dim
    fi
  done
  # Remove temp files
  if test -d $base_path/$temp/$name; then
    rm -rf $base_path/$temp/$name;
  fi
  if test -f $base_path/$temp/$name.tif; then
    rm $base_path/$temp/$name.tif
  fi
  # Remove tiff files
  if test -f $base_path/$dest/$name.tif; then
    rm $base_path/$dest/$name.tif
  fi
  # Remove unziped product
  if test -d $base_path/$source/$name.SAFE; then
    rm -rf $base_path/$source/$name.SAFE
  fi
  # Remove product file
  if test -f $base_path/$source/$fname; then
    rm $base_path/$source/$fname
  fi
  # Proceed with next file
done
