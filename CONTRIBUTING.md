# Contributing

### Getting the source code

You need to clone the repository from GitHub.
Using SSH (recommended):

```
git clone git@github.com:yoavram/curveball.git
cd curveball
```

or HTTPS:

```
git clone https://github.com/yoavram/curveball.git
cd curveball
```

Than, create a new environment with all the dependencies

```
conda create -n curveball_dev -c https://conda.anaconda.org/t/yo-766bbd1c-8edd-4b45-abea-85cf58129278/yoavram curveball
activate curveball_dev
curveball --version
```

If there are no errors and the last command resulted in the current stable version of Curveball, 
you can install Curveball in development work so that any changes you make will affect your working environments:

```
python setup.py develop
```

### Editing the source code

- the root folder has different files that are used to test, package, and deploy Curveball.
- the source code is inside the `curveball` folder
- the tests are in the `tests` folder 
- data files are in the `data` folder
- examples are in the `examples` folder

### Editing the documentation

All documentation, including docstrings, is written in `reStructuredText  <http://sphinx-doc.org/rest.html>`_,
which is just plain text with some markup to indicate text structure.
The documentation files are in the `docs` folder: 
`.rst` files are the source files of the documentation, 
`conf.py` is a configuration file,
the `_static` folder mostly has images and some other static assets.
To build the documentation you need to install Sphinx and numpydoc:

```
pip install sphinx numpydoc sphinx-rtd-theme
```

Docstrings and documentation follow the `numpy/scipy standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

#### Building the documentation

You can build different formats of the docs, but the main format is the HTML website.
If you have *make* install, then the command:

```
make doc
```

will generate the documentation and open it in your browser.

If you don't have *make*, just run this command (which is what *make* does, anyway):

```
cd docs 
make html && open _build/html/index.html
```
### Review changes

When you are finished changing the source code (or docs), check that everything is committed:

```
git status
```

Review difference from the origin master branch:

```
git diff origin/master
```

and the commit log:

```
git log
```

Consider rebasing commits (Only those that were not pushed! The `master` branch is protected from force push):

```
git rebase -i HEAD~#
```

Replace `#` with the number of commits you want to rebase (read the instructions in the editor that opens up).

If all changes are committed and you are sure you are interested in all the commits, you can continue.

### Run tests locally

If you changed the source code, you need to run the tests.
First, install the tests dependencies:

```
pip install nose pillow
```

Then run the tests:

```
nosetests -v tests
```

If you only changed the documentation, just make sure it is built and looks good (see above).

### Build Curveball

Build the package using conda. 
You first need to install `conda-build`:

```
conda install conda-build
```

The command

```
conda build conda-recipe -c yoavram
```

will create a new `curveball-<curveball-version>-<python-version>-<build-number>.tar.bz2` or similar package file in the `CONDAPATH/conda-bld/<operating-system>` folder, where `CONDAPATH` is the path to your Anaconda/Miniconda install (for example, on my PC it's at ` d:\Anaconda3\conda-bld\win-64\curveball-v0.1.10b-py34_7.tar.bz2`; on Travis-CI Ubuntu machine it's at `~/miniconda/conda-bld/linux-64/curveball-v0.1.10b-py34_2.tar.bz2`). 
The build will fail if any dependencies are missing.

### Test the build

Now create a new environment using `conda` and try to install Curveball from the package you just built (replace `PYTHON_VERSION` with the Python version you are using, either 2.7 or 3.4):

```
conda create -n test python=PYTHON_VERSION
activate test
conda install --use-local curveball
```

Verify the installation:

```
curveball --version
```

If everything worked out you can remove the environment:
```
deactivate
conda env remove -n test
```

### Releasing a new version

The release process includes several stages:

- review changes
- test locally or using continuous integration
- adding the changes to the change log
- declaring a new version
- building and deploying the package to a package index
- building and deploying the documentation

A more elaborate scheme can be found [here](https://khmer.readthedocs.org/en/v1.1/release.html).

#### Adding changes to the change log

Before declaring a new version you should describe the changes made since the last version.
Call `git log vV.. --oneline --decorate`, replacing `V` with the tag of the last version, 
to see all the commit logs since the last version; then edit `CHANGELOG.md`.

#### Declaring a new version

This is done by tagging a commit with a new version number which must conform to PEP440 (v#.#.#).
No change in the code itself is required as everything is handled by [versioneer](https://github.com/warner/python-versioneer).
**NOTE**: make sure the last commit didn't have `[skip ci]` in the message!.

```
git tag v#.#.#
git push
git push --tags
```

#### Automatic deployment

Pushing the tags will cause [Travis-CI](https://magnum.travis-ci.com/yoavram/curveball)
to pull the code, install dependencies, and test the code. 
If the tests succeed, Travis-CI will **build the package and deploy it to Anaconda.org**,
but only on tagged commits from the `master` branch.

Travis-CI will also **build the documentation and deploy it to [netlify](https://curveball.netlify.com)**.
