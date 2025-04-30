#!/bin/bash
#
#SBATCH --partition=C0
#SBATCH --job-name=PolarimetricDecompositions
#SBATCH --time=0
#SBATCH --mem=0

base_path=$1
prod=SLC
fname=$3
temp=$4
gpt=~/tools/esa-snap10.0/bin/gpt

set -x #echo on
set -e #stop on error
name=${fname%%.*}
echo $name
mkdir -p $base_path/$temp/$name
# Apply orbit file
if ! test -f $base_path/$temp/$name/pol_01.dim; then
  srun $gpt scripts/polarimetric_01.xml -SsourceProduct=$base_path/$prod/$name.SAFE -t $base_path/$temp/$name/pol_01.dim
fi
# TOPSAR Split subswath
for n in 1 2 3 
do
  if ! test -f $base_path/$temp/$name/pol_02_iw${n}.dim; then
    srun $gpt scripts/polarimetric_02.xml -SsourceProduct=$base_path/$temp/$name/pol_01.dim -Psubswath=IW${n} -t $base_path/$temp/$name/pol_02_iw${n}.dim
  fi
done
# Calibrate splits
for n in 1 2 3 
do
  if ! test -f $base_path/$temp/$name/pol_03_iw${n}.dim; then
    srun $gpt scripts/polarimetric_03.xml -SsourceProduct=$base_path/$temp/$name/pol_02_iw${n}.dim -t $base_path/$temp/$name/pol_03_iw${n}.dim
  fi
done
# TOPSAR Deburst splits
for n in 1 2 3 
do
  if ! test -f $base_path/$temp/$name/pol_04_iw${n}.dim; then
    srun $gpt scripts/polarimetric_04.xml -SsourceProduct=$base_path/$temp/$name/pol_03_iw${n}.dim -t $base_path/$temp/$name/pol_04_iw${n}.dim
  fi
done
# TOPSAR Merge
if ! test -f $base_path/$temp/$name/pol_05.dim; then
  srun $gpt scripts/polarimetric_05.xml -Ssource1=$base_path/$temp/$name/pol_04_iw1.dim -Ssource2=$base_path/$temp/$name/pol_04_iw2.dim -Ssource3=$base_path/$temp/$name/pol_04_iw3.dim -t $base_path/$temp/$name/pol_05.dim
fi
# Polarimetric Matrix 
if ! test -f $base_path/$temp/$name/pol_06.dim; then
  srun $gpt scripts/polarimetric_06.xml -SsourceProduct=$base_path/$temp/$name/pol_05.dim -t $base_path/$temp/$name/pol_06.dim
fi
# Multilook
if ! test -f $base_path/$temp/$name/pol_07.dim; then
  srun $gpt scripts/polarimetric_07.xml -SsourceProduct=$base_path/$temp/$name/pol_06.dim -t $base_path/$temp/$name/pol_07.dim
fi
# Speckle filter (Lee)
if ! test -f $base_path/$temp/$name/pol_08.dim; then
  srun $gpt scripts/polarimetric_08.xml -SsourceProduct=$base_path/$temp/$name/pol_07.dim -t $base_path/$temp/$name/pol_08.dim
fi
# Polarimetric decomposition
if ! test -f $base_path/$temp/$name/pol_09.dim; then
  srun $gpt scripts/polarimetric_09.xml -SsourceProduct=$base_path/$temp/$name/pol_08.dim -t $base_path/$temp/$name/pol_09.dim
fi
# Ellipsoid correction
if ! test -f $base_path/$temp/$name/pol_10.dim; then
  srun $gpt scripts/polarimetric_10.xml -SsourceProduct=$base_path/$temp/$name/pol_09.dim -t $base_path/$temp/$name/pol_10.dim
fi
# Land mask
if ! test -f $base_path/$temp/$name/pol_11.dim; then
  srun $gpt scripts/polarimetric_11.xml -SsourceProduct=$base_path/$temp/$name/pol_10.dim -t $base_path/$temp/$name/pol_11.dim
fi
# $gpt scripts/sar_export_to_tif.xml -SsourceProduct=$base_path/temp/$NAME.dim -t $base_path/GRD_tif/$NAME.tif
echo "Done!"
