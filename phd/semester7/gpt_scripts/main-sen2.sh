#!/bin/bash
#
# We are looping over the image files and call the polarimetric decompositions SLURM scripts
# In this script we are waiting to last file generated to copy on siimon5 external disk drive

# First we are doing this in a sequentially manner
num=$1

# Variable configurations
gpt=~/tools/esa-snap10.0/bin/gpt
dataset=noaa
datadir=sentinel2
base_path=~/data/cimat/dataset-$dataset
source=L2A
dest=tiff
temp=temp1
set -x
set -e
mkdir -p $base_path/$source
mkdir -p $base_path/$temp
mkdir -p $base_path/$dest
# Download list of files to process
scp -P 2235 manuelsuarez@siimon5.cimat.mx:~/data/cimat/dataset-${dataset}/$datadir/${source}_list${num}.txt $base_path/$datadir/${source}_list${num}.txt
for fname in $(cat $base_path/$datadir/${source}_list${num}.txt); do
  name=${fname%%.*}
  ext=${fname##*.}
  echo "Processing $name in bash script"
  echo "Extension $ext"
  # If final windfield zip file exists, continue
  if [ -f $base_path/$datadir/$dest/${name}/B12.tif ]; then
    echo "$base_path/$datadir/$dest/${name}/B12 band tiff file exists...., continue next"
    continue
  fi
  echo "Processing sentinel-2 bands extraction script"
  # Transfer product from siimon5 GRD product directory
  if [ ! -f $base_path/$datadir/$source/$fname ]; then
    scp -P 2235 manuelsuarez@siimon5.cimat.mx:~/data/cimat/dataset-${dataset}/$datadir/$source/$fname $base_path/$datadir/$source/$fname
  fi
  # Unzip product and make temp directory
  if [ $ext == "zip" ]; then
    if [ ! -d $base_path/$datadir/$source/$name.SAFE ]; then
      unzip -qo $base_path/$datadir/$source/$fname -d $base_path/$datadir/$source/
    fi
  fi
  mkdir -p $base_path/$datadir/$temp/$name
  mkdir -p $base_path/$datadir/$dest/$name
  # Run SLURM tiff band extraction script (on C0 cpu nodes)
  if [ ! -f $base_path/$datadir/$dest/$name/${name}_B12.tif ]; then
    sbatch run-sen2.slurm $base_path/$datadir $source $dest $temp $fname
  fi
  # bash make_sen2.sh $base_path/$datadir $source $dest $temp $fname
  # Move result to siimon5 (we are implementing an active waiting using sleep)
  while [ ! -f $base_path/$datadir/$dest/$name/${name}_B12.tif ]; do
    # Sleep
    echo "$base_path/$datadir/$dest/$name/${name}_B12.tif result not created, waiting 10m..."
    sleep 10m
  done
  echo "$base_path/$datadir/$dest/$name bands created, proceeding to move to siimon5"
  sleep 1m
  # Once that result is created we move it to siimon5
  ssh -p 2235 manuelsuarez@siimon5.cimat.mx "mkdir -p ~/data/cimat/dataset-${dataset}/$datadir/$dest/$name"
  for band in B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12; do
    scp -P 2235 $base_path/$datadir/$dest/$name/${name}_${band}.tif manuelsuarez@siimon5.cimat.mx:~/data/cimat/dataset-${dataset}/$datadir/$dest/$name/${name}_${band}.tif
  done
  # Delete temp files and directories
  for band in B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12; do
    if test -d $base_path/$datadir/$temp/$name/sen2_03_${band}.data; then
      rm -rf $base_path/$datadir/$temp/$name/sen2_03_${band}.data
    fi
    if test -f $base_path/$datadir/$temp/$name/sen2_03_${band}.dim; then
      rm $base_path/$datadir/$temp/$name/sen2_03_${band}.dim
    fi
  done
  if [ -f $base_path/$datadir/$temp/$name/sen2_02.dim]; then
    rm -rf $base_path/$datadir/$temp/$name/sen2_02.data;
    rm $base_path/$datadir/$temp/$name/sen2_02.dim;
  fi
  if [ -f $base_path/$datadir/$temp/$name/sen2_01.dim]; then
    rm -rf $base_path/$datadir/$temp/$name/sen2_01.data;
    rm $base_path/$datadir/$temp/$name/sen2_01.dim;
  fi
  # Remove temp files
  if test -d $base_path/$datadir/$temp/$name; then
    rm -rf $base_path/$datadir/$temp/$name;
  fi
  if test -d $base_path/$datadir/$dest/$name; then
    rm -rf $base_path/$datadir/$dest/$name
  fi
  # Remove unziped product
  if test -d $base_path/$datadir/$source/$name.SAFE; then
    rm -rf $base_path/$datadir/$source/$name.SAFE
  fi
  # Remove product file (on SIIMON5 not remove)
  if test -f $base_path/$datadir/$source/$fname; then
    rm $base_path/$datadir/$source/$fname
  fi
  # Proceed with next file
done
