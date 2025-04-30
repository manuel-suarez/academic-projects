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

# In this second script we are applying the posterior steps to wind step estimation
#
# Terrain correction
if ! test -f $base_path/$temp/$name/wind_07.dim; then
  $gpt scripts/wind_07.xml -SsourceProduct=$base_path/$temp/$name/wind_06.dim -t $base_path/$temp/$name/wind_07.dim
fi
