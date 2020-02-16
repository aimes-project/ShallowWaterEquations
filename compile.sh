#!/bin/bash
export SIMPLE_BACKUP_SUFFIX=ibf
for conf in serial.conf OMP.conf blocking.conf
do
  echo 'Generating code for ('$conf') target'
  rm -r $conf'_target' 2> /dev/null
  python tool.zip -dsl GGDML -sp $conf -ni src
  cd $conf'_target'
  indent -kr -nut -l1000  *.c *.h 2> /dev/null
  rm *.cibf *.hibf 2> /dev/null
  cd ..
done
