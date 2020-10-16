##!/bin/bash

# If a prosgres process is running, ssh to node and kill process
export uname=$USER
export appname='postgres \-D'

# Check 6 login nodes
for i in {1..6}
do
  hname=thetalogin$i
  if [[ "$HOSTNAME" = $hname  ]]
  then
    hostname; ps aux|grep $uname|grep "$appname"
  else
    ssh $hname "hostname; ps aux|grep $uname|grep '$appname'"
  fi
done

# Check 3 MOM nodes
for i in {1..3}
do
  hname=thetamom$i
  if [[ "$HOSTNAME" = $hname  ]]
  then
    hostname; ps aux|grep $uname|grep '$appname'
  else
    ssh $hname "hostname; ps aux|grep $uname|grep '$appname'"
  fi
done
