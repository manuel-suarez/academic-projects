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
dest=texture
temp=temp2
num=$1
set -x
set -e
mkdir -p $base_path/$source
mkdir -p $base_path/$dest
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
  if [ ! -f $base_path/$temp/$name/pol_01.dim ]; then
    # If product is Envisat (N1 extension) apply operator directly on product
    if [ $ext == "N1" ]; then
      $gpt scripts/glm-gen_01.xml -SsourceProduct=$base_path/$source/$fname -t $base_path/$temp/$name/glm_01.dim
    else
      $gpt scripts/glm-gen_01.xml -SsourceProduct=$base_path/$source/$name.SAFE -t $base_path/$temp/$name/glm_01.dim
    fi
  fi
  # Run slurm texture features script (on C0 cpu nodes)
  sbatch run-glcm.slurm $fname
  # Move result to siimon5 (we are implementing an active waiting using sleep)
  # Once that result is created we move it to siimon5
  for feature in Contrast Dissimilarity Homogeneity ASM Energy MAX Entropy GLCMMean GLCMVariance GLCMCorrelation; do
    # There are ten features to be generated, we are waiting for the last
    while [ ! -f $base_path/$dest/$name/$feature.tif ]; do
      # Sleep
      if [ $feature == "Contrast" ]; then
        echo "$base_path/$dest/$name/$feature not created, waiting 20m..."
        sleep 20m
      else
        echo "$base_path/$dest/$name/$feature not created, waiting 3m..."
        sleep 3m
      fi
    done
    echo "$base_path/$dest/$name/$feature created, proceeding to move to siimon5"
    sleep 2m
    scp -P 2235 $base_path/$dest/$name/$feature.tif manuelsuarez@siimon5.cimat.mx:~/data/cimat/dataset-${dataset}/$dest/$feature/$name.tif
  done
  # Remove temporary files 01-06
  #
  for step in 01 02 03 04 05 06
  do
    if test -d $base_path/$temp/$name/glm_${step}.data; then
      rm -rf $base_path/$temp/$name/glm_${step}.data
    fi
    if test -f $base_path/$temp/$name/glm_${step}.dim; then
      rm $base_path/$temp/$name/glm_${step}.dim
    fi
  done
  # Remove generated features
  for feature in Contrast Dissimilarity Homogeneity ASM Energy MAX Entropy GLCMMean GLCMVariance GLCMCorrelation; do
    if test -f $base_path/$dest/$name/$feature.tif; then
      rm $base_path/$dest/$name/$feature.tif
    fi
  done
  # Remove temp and dest name directories
  if test -d $base_path/$dest/$name; then
    rm -rf $base_path/$dest/$name
  fi
  if test -d $base_path/$temp/$name; then
    rm -rf $base_path/$temp/$name
  fi
  # Remove unziped product
  if test -d $base_path/$source/$name.SAFE; then
    rm -rf $base_path/$source/$name.SAFE
  fi
  # Remove original product
  if test -f $base_path/$source/$fname; then
    rm $base_path/$source/$fname
  fi
  # Proceed with next file
done
