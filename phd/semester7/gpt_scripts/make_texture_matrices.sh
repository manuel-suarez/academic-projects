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

# Unzip if is a sentinel file (zip extension)
if [ $ext == "zip"]; then
  if [ ! -d $base_path/$source/$name.SAFE ]; then
    unzip -qo $base_path/$source/$fname -d $base_path/$source/
  fi
fi
mkdir -p $base_path/$temp/$name
mkdir -p $base_path/$dest/$name
# Apply orbit file
if ! test -f $base_path/$temp/$name/glm_01.dim; then
  # If product is Envisat (N1 extension) apply operator on filename directly
  if [ $ext == "N1" ]; then
    $gpt scripts/glm-gen_01.xml -SsourceProduct=$base_path/$source/$fname -t $base_path/$temp/$name/glm_01.dim
  else
    $gpt scripts/glm-gen_01.xml -SsourceProduct=$base_path/$source/$name.SAFE -t $base_path/$temp/$name/glm_01.dim
  fi
fi
# Calibration
if ! test -f $base_path/$temp/$name/glm_02.dim; then
  $gpt scripts/glm-gen_02.xml -SsourceProduct=$base_path/$temp/$name/glm_01.dim -t $base_path/$temp/$name/glm_02.dim
fi
# Speckle filter (Lee)
if ! test -f $base_path/$temp/$name/glm_03.dim; then
  $gpt scripts/glm-gen_03.xml -SsourceProduct=$base_path/$temp/$name/glm_02.dim -t $base_path/$temp/$name/glm_03.dim
fi
# Ellipsoid correction
if ! test -f $base_path/$temp/$name/glm_04.dim; then
  $gpt scripts/glm-gen_04.xml -SsourceProduct=$base_path/$temp/$name/glm_03.dim -t $base_path/$temp/$name/glm_04.dim
fi
# Linear conversion
if ! test -f $base_path/$temp/$name/glm_05.dim; then
  $gpt scripts/glm-gen_05.xml -SsourceProduct=$base_path/$temp/$name/glm_04.dim -t $base_path/$temp/$name/glm_05.dim
fi
# GLCM matrix generation
if ! test -f $base_path/$temp/$name/glm_06.dim; then
  $gpt scripts/glm-gen_06.xml -SsourceProduct=$base_path/$temp/$name/glm_05.dim -t $base_path/$temp/$name/glm_06.dim
fi
# Export individual bands (texture features)
for feature in Contrast Dissimilarity Homogeneity ASM Energy MAX Entropy GLCMMean GLCMVariance GLCMCorrelation; do
  if ! test -f $base_path/$temp/$name/$feature.tif; then
    # Extract individual feature band
    $gpt scripts/glm-gen_07.xml -SsourceProduct=$base_path/$temp/$name/glm_06.dim -PsourceBand=Sigma0_VV_db_${feature} -PoutputFile=$base_path/$temp/$name/$feature.tif
    # Move to destination folder
    mv $base_path/$temp/$name/$feature.tif $base_path/$dest/$name/$feature.tif
  fi
done
# $gpt scripts/sar_export_to_tif.xml -SsourceProduct=dataset-sentinel/temp/$NAME.dim -t dataset-sentinel/GRD_tif/$NAME.tif
