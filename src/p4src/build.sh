#!/bin/bash
NAME=domain_spotter
rm -f bf_drivers.log*
rm -f *.log
rm -rf $SDE_INSTALL/$NAME.tofino/
rm -rf $NAME.tofino/
bf-p4c $NAME.p4 --create-graphs --display-power-budget --log-hashes -g -Xp4c=\"--disable-parse-depth-limit\"
cp -R $NAME.tofino $SDE_INSTALL/
