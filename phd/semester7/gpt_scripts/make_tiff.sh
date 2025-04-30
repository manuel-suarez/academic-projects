#!/bin/bash
gpt=~/tools/esa-snap10.0/bin/gpt
base_path=$1
source=$2
dest=$3
temp=$4
fname=$5

name=${fname%%.*}
echo $name
ext=${fname##*.}
echo $ext

set -x
set -e

# Unzip if is a sentinel product (zip extension)
if [ $ext == "zip" ]; then
  if [ ! -d $base_path/$source/$name.SAFE ]; then
    unzip -qo $base_path/$source/$fname -d $base_path/$source/
  fi
fi
mkdir -p $base_path/$temp/$name
# Apply orbit file
if ! test -f $base_path/$temp/$name/tiff_01.dim; then
  # If product is ENVISAT ASAR (N1 extension) apply operator on product
  if [ $ext == "N1" ]; then
    $gpt scripts/tiff_01.xml -SsourceProduct=$base_path/$source/$fname -t $base_path/$temp/$name/tiff_01.dim
  else
    $gpt scripts/tiff_01.xml -SsourceProduct=$base_path/$source/$name.SAFE -t $base_path/$temp/$name/tiff_01.dim
  fi
fi
# Calibration
if ! test -f $base_path/$temp/$name/tiff_02.dim; then
  $gpt scripts/tiff_02.xml -SsourceProduct=$base_path/$temp/$name/tiff_01.dim -t $base_path/$temp/$name/tiff_02.dim
fi
# Multilook
if ! test -f $base_path/$temp/$name/tiff_03.dim; then
  # If product is ENVISAT the looks parameter is 1, if sentinel, 2
  if [ $ext == "N1" ]; then
    $gpt scripts/tiff_03.xml -SsourceProduct=$base_path/$temp/$name/tiff_02.dim -Plooks=1 -t $base_path/$temp/$name/tiff_03.dim
  else
    $gpt scripts/tiff_03.xml -SsourceProduct=$base_path/$temp/$name/tiff_02.dim -Plooks=2 -t $base_path/$temp/$name/tiff_03.dim
  fi
fi
# Ellipsoid correction
if ! test -f $base_path/$temp/$name/tiff_04.dim; then
  $gpt scripts/tiff_04.xml -SsourceProduct=$base_path/$temp/$name/tiff_03.dim -t $base_path/$temp/$name/tiff_04.dim
fi
# Linear conversion of VV band
if ! test -f $base_path/$temp/$name/tiff_05.dim; then
  $gpt scripts/tiff_05.xml -SsourceProduct=$base_path/$temp/$name/tiff_04.dim -PsourceBand=Sigma0_VV -t $base_path/$temp/$name/tiff_05.dim
fi
# Write to output file
if ! test -f $base_path/$temp/$name.tif; then
  $gpt scripts/tiff_06.xml -SsourceProduct=$base_path/$temp/$name/tiff_05.dim -PoutputFile=$base_path/$temp/$name.tif
fi
# Linear conversion of VH band
#if ! test -f $base_path/$temp/$name/${name}_VH.tif; then
#  $gpt scripts/tiff_05.xml -SsourceProduct=$base_path/$temp/$name/tiff_04.dim -PsourceBand=Sigma0_VH -t $base_path/$temp/$name/${name}$feature.tif
# $gpt scripts/sar_export_to_tif.xml -SsourceProduct=dataset-sentinel/temp/$NAME.dim -t dataset-sentinel/GRD_tif/$NAME.tif
mv $base_path/$temp/$name.tif $base_path/$dest/$name.tif
#mv $base_path/$temp/$name/${name}_VH.tif $base_path/$dest/$name/${name}_VH.tif
