#!/bin/bash
# Looks for Conda in common Linux installation locations

readonly CONDA_PATHS=("/opt" "$HOME")
readonly CONDA_NAMES=("/ana" "/mini")
readonly CONDA_BINS=("/bin/conda" "/condabin/conda")

condabin="$(which conda 2>/dev/null)"                       # Check 'which conda'
if test -f "$condabin" ; then                               # This does not always return a path, so make sure location exists
    echo $(readlink -f "$(dirname "$condabin")/..")         # Echo Conda folder location
    exit 0;                                                 # Exit
fi

for path in "${CONDA_PATHS[@]}"; do                         # Search common locations
    for name in "${CONDA_NAMES[@]}" ; do
        foldername="$path${name}conda3"
        for bin in "${CONDA_BINS[@]}" ; do
            condabin="$foldername$vers$bin";
            if test -f "$condabin" ; then                   # Check location exists
            echo $(readlink -f "$(dirname "$condabin")/..") # Echo Conda folder location
            exit 0;                                         # Exit
            fi
        done
    done
done

echo ""                                                 # Output empty line if not found
