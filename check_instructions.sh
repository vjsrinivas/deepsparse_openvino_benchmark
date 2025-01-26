#!/bin/bash

echo -e "Feature\tSupported"
echo -e "-------\t--------"

# Check for AVX2 support
if grep -q 'avx2' /proc/cpuinfo; then
  echo -e "AVX2\tYes"
else
  echo -e "AVX2\tNo"
fi

# Check for AVX-512 support
if grep -q 'avx512' /proc/cpuinfo; then
  echo -e "AVX-512\tYes"
else
  echo -e "AVX-512\tNo"
fi

# Check for VNNI support
if grep -q 'vnni' /proc/cpuinfo; then
  echo -e "VNNI\tYes"
else
  echo -e "VNNI\tNo"
fi
