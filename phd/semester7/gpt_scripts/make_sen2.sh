#!/bin/bash
gpt=~/tools/esa-snap10.0/bin/gpt
base_path=$1
source=$2
dest=$3
temp=$4
fname=$5

name=${fname%%.*}
ext=${fname##*.}
echo "Processing $name in bash script"
echo "Extension $ext"

set -x
set -e

# Unzip product and make temp directory
if [ $ext == "zip" ]; then
  if [ ! -d $base_path/$datadir/$source/$name.SAFE ]; then
    unzip -qo $base_path/$datadir/$source/$fname -d $base_path/$datadir/$source/
  fi
fi
mkdir -p $base_path/$datadir/$temp/$name
# Subset (B1-B12 bands)
if ! test -f $base_path/$datadir/$temp/$name/sen2_01.dim; then
  $gpt scripts/sen2_01.xml -SsourceProduct=$base_path/$datadir/$source/$fname -Pbands=B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12 -t $base_path/$datadir/$temp/$name/sen2_01.dim
fi
# Resampling
if ! test -f $base_path/$datadir/$temp/$name/sen2_02.dim; then
  $gpt scripts/sen2_02.xml -SsourceProduct=$base_path/$datadir/$temp/$name/sen2_01.dim -t $base_path/$datadir/$temp/$name/sen2_02.dim
fi
# Subset (per band)
for band in B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12; do
  if [ ! -f $base_path/$datadir/$temp/$name/sen2_03_${band}.dim ]; then
    $gpt scripts/sen2_03.xml -SsourceProduct=$base_path/$datadir/$temp/$name/sen2_02.dim -Pband=${band} -t $base_path/$datadir/$temp/$name/sen2_03_${band}.dim
  fi
  # Write
  if [ ! -f $base_path/$datadir/$temp/$name/${name}_${band}.tif ]; then
    $gpt scripts/sen2_04.xml -SsourceProduct=$base_path/$datadir/$temp/$name/sen2_03_${band}.dim -PoutputFile=$base_path/$datadir/$temp/$name/${name}_${band}.tif
  fi
  # Move to destination
  mv $base_path/$datadir/$temp/$name/${name}_${band}.tif $base_path/$datadir/$dest/$name/${name}_${band}.tif
done
