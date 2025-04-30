#!/bin/bash
gpt=~/tools/esa-snap10.0/bin/gpt
base_path=$1
source=$2
dest=$3
temp=$4
fname=$5

name=${fname%%.*}
echo "Processing $name in slurm script"

# Apply orbit file
set -x
set -e
if [ ! -d $base_path/$source/$name.SAFE ]; then
  unzip -qo $base_path/$source/$fname -d $base_path/$source/
fi
mkdir -p $base_path/$temp/$name
mkdir -p $base_path/$dest/$name
if ! test -f $base_path/$temp/$name/pol_01.dim; then
  $gpt scripts/polarimetric_01.xml -SsourceProduct=$base_path/$source/$name.SAFE -t $base_path/$temp/$name/pol_01.dim
fi
# TOPSAR Split subswath
for n in 1 2 3
do
  if ! test -f $base_path/$temp/$name/pol_02_iw${n}.dim; then
    $gpt scripts/polarimetric_02.xml -SsourceProduct=$base_path/$temp/$name/pol_01.dim -Psubswath=IW${n} -t $base_path/$temp/$name/pol_02_iw${n}.dim
  fi
done
# Calibrate splits
for n in 1 2 3 
do
  if ! test -f $base_path/$temp/$name/pol_03_iw${n}.dim; then
    $gpt scripts/polarimetric_03.xml -SsourceProduct=$base_path/$temp/$name/pol_02_iw${n}.dim -t $base_path/$temp/$name/pol_03_iw${n}.dim
  fi
done
# TOPSAR Deburst splits
for n in 1 2 3 
do
  if ! test -f $base_path/$temp/$name/pol_04_iw${n}.dim; then
    $gpt scripts/polarimetric_04.xml -SsourceProduct=$base_path/$temp/$name/pol_03_iw${n}.dim -t $base_path/$temp/$name/pol_04_iw${n}.dim
  fi
done
# TOPSAR Merge
if ! test -f $base_path/$temp/$name/pol_05.dim; then
  $gpt scripts/polarimetric_05.xml -Ssource1=$base_path/$temp/$name/pol_04_iw1.dim -Ssource2=$base_path/$temp/$name/pol_04_iw2.dim -Ssource3=$base_path/$temp/$name/pol_04_iw3.dim -t $base_path/$temp/$name/pol_05.dim
fi
# Polarimetric Matrix 
if ! test -f $base_path/$temp/$name/pol_06.dim; then
  $gpt scripts/polarimetric_06.xml -SsourceProduct=$base_path/$temp/$name/pol_05.dim -t $base_path/$temp/$name/pol_06.dim
fi
# Multilook
if ! test -f $base_path/$temp/$name/pol_07.dim; then
  $gpt scripts/polarimetric_07.xml -SsourceProduct=$base_path/$temp/$name/pol_06.dim -t $base_path/$temp/$name/pol_07.dim
fi
# Speckle filter (Lee)
if ! test -f $base_path/$temp/$name/pol_08.dim; then
  $gpt scripts/polarimetric_08.xml -SsourceProduct=$base_path/$temp/$name/pol_07.dim -t $base_path/$temp/$name/pol_08.dim
fi
# Polarimetric decomposition
if ! test -f $base_path/$temp/$name/pol_09.dim; then
  $gpt scripts/polarimetric_09.xml -SsourceProduct=$base_path/$temp/$name/pol_08.dim -t $base_path/$temp/$name/pol_09.dim
fi
# Ellipsoid correction
if ! test -f $base_path/$temp/$name/pol_10.dim; then
  $gpt scripts/polarimetric_10.xml -SsourceProduct=$base_path/$temp/$name/pol_09.dim -t $base_path/$temp/$name/pol_10.dim
fi
# Land mask
#if ! test -f $base_path/$temp/$name/pol_11.dim; then
#  $gpt scripts/polarimetric_11.xml -SsourceProduct=$base_path/$temp/$name/pol_10.dim -t $base_path/$temp/$name/pol_11.dim
#fi
# $gpt scripts/sar_export_to_tif.xml -SsourceProduct=dataset-sentinel/temp/$NAME.dim -t dataset-sentinel/GRD_tif/$NAME.tif
# We need to wait until pol_10.dim has been created and move to destination folder to allow main script to delete temporary products
mv $base_path/$temp/$name/pol_10.data $base_path/$dest/$name/pol_10.data
mv $base_path/$temp/$name/pol_10.dim $base_path/$dest/$name/pol_10.dim
