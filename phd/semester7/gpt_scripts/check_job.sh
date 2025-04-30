#!/bin/bash
#
job_id=$1
#echo $job_status
if [[ $(squeue -h -j $job_id) != "" ]]; then 
  status=$(squeue -h -o "%T" -j $job_id)
  echo "still running $job_id $status" 
else 
  echo "has finished" 
fi

