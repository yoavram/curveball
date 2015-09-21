# Curveball: growth curves analysis

[![Latest Version](https://img.shields.io/pypi/v/curveball.svg)](https://pypi.python.org/pypi/curveball/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/curveball.svg)](https://pypi.python.org/pypi/curveball/)
[![License](https://img.shields.io/pypi/l/curveball.svg)](https://pypi.python.org/pypi/curveball/)

[![wercker status](https://app.wercker.com/status/ccdaa657b94c79e9a1fe194353613c13/m/master "wercker status")](https://app.wercker.com/project/bykey/ccdaa657b94c79e9a1fe194353613c13)

[![logo](/docs/_static/logo_200px.png?raw=true)](http://www.freepik.com/free-vector/ball-of-wool_762106.htm)

Analyzing microbial growth curves using ecological and evolutionary models

## Installation

Curveball can be installed with:
```
pip install git+https://github.com/yoavram/curveball.git#egg=curveball
```

You might need to install the `lxml` dependencies. On _Ubuntu_ you'll need to [setup `libxml` on the global environment](http://stackoverflow.com/a/15761014/1063612) before running `pip`::
```
sudo apt-get install libxml2-dev libxslt1-dev
```

	To test the package on Windows inside a virtualenv you might need to [set the Tcl environment variables](https://github.com/pypa/virtualenv/issues/93):
```
set TCL_LIBRARY=c:\python27\tcl\tcl8.5
set TK_LIBRARY=c:\python27\tcl\tk8.5
```

where the numbers could change depending on versions and paths.
You can set this in the `venv\scripts\activate.bat` to occur automatically.

## Contribute
**curveball** is an open-source software. Everyone is welcome to contribute! Please Use issues and pull requests in the official [Github repository](https://github.com/yoavram/curveball)


