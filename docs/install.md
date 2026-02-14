# Install

**Requirements**

- Python 3.9+
- A C++11 compiler (Clang, GCC, or MSVC)
- CMake 3.20+
- Ninja (recommended)

**Install from PyPI**

```bash
pip install autoneuronet
```

**Optional extras**

```bash
python -m pip install autoneuronet[demo]
python -m pip install autoneuronet[dev]
```

**Build locally**

First, clone the repo from [Github](https://github.com/RishabSA/AutoNeuroNet):

```bash
git clone https://github.com/RishabSA/AutoNeuroNet.git
```

To build the necessary C++ dependencies for use in Python run:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
```

The C++ dependencies can then be accessed from the build/ folder and imported in any Python files.

For building the python package locally, run the following commands:

```bash
python -m pip uninstall -y autoneuronet # (if a preexisting package already exists)

python -m pip install -U pip build
python -m pip install -U build twine
python -m build
python -m pip install dist/*.whl
```

To build with twine to upload to PyPi, run the following commands:

```bash
python -m twine check dist/*
```

Upload to TestPyPi or PyPi:

```bash
python -m twine upload -r testpypi dist/* # Upload to TestPyPi
python -m twine upload dist/* # Upload to PyPi
```

**Build wheels for macOS, Windows, and Linux via Github Actions**

1. Commit all changes, but don't push yet
2. Push a tag for the version: v0.1.N

```bash
git tag -a v0.1.N -m "v0.1.N"
git push origin main v0.1.N
```

The workflow in `.github/workflows/wheels.yml` builds wheels for Linux, macOS, and Windows. Wheels are then published to PyPI via an API token.

**Verify the install**

```bash
python - <<'PY'
import autoneuronet as ann
x = ann.Var(2.0)
y = x * x
print(y)
PY
```
