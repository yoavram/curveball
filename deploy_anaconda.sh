#!/bin/bash
# this script uses the ANACONDA_TOKEN env var. 
# to create a token:
# >>> anaconda login
# >>> anaconda auth -c -n cruveball-travis --max-age 307584000 --url https://anaconda.org/yoavram/curveball --scopes "api:write api:read"
set -e

echo "Converting conda package..."
conda convert --platform all $HOME/miniconda3/conda-bld/linux-64/curveball-*.tar.bz2 --output-dir conda-bld/

echo "Deploying to Anaconda.org..."
anaconda -t $ANACONDA_TOKEN upload conda-bld/win-32/curveball-*.tar.bz2
anaconda -t $ANACONDA_TOKEN upload conda-bld/win-64/curveball-*.tar.bz2
anaconda -t $ANACONDA_TOKEN upload conda-bld/linux-32/curveball-*.tar.bz2
anaconda -t $ANACONDA_TOKEN upload conda-bld/linux-64/curveball-*.tar.bz2
anaconda -t $ANACONDA_TOKEN upload conda-bld/osx-64/curveball-*.tar.bz2

echo "Successfully deployed to Anaconda.org."
exit 0
