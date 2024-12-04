#!/bin/bash
clear
echo "Recompiling to integrate new changes..."
rm -r build
mkdir build
cd build/
echo "Changed to 'build' dir to start compilation..."
cmake ../src -DCMAKE_BUILD_TYPE=RELEASE
make -j 4
echo "Compilation done!"
 
