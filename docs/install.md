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

**Build wheels for macOS, Windows, and Linux**

Local (current OS only):

```bash
python -m pip install -U cibuildwheel
python -m cibuildwheel --output-dir dist
```

All OSes via GitHub Actions:

1. Push a tag like `v0.1.3` (or run the workflow manually).
2. The workflow in `.github/workflows/wheels.yml` builds wheels for Linux, macOS, and Windows.
3. Wheels are published to PyPI via Trusted Publishing. If you prefer API tokens, replace the publish step with `twine upload`.

```bash
git tag -a v0.1.3 -m "v0.1.3"
git push origin v0.1.3
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

**Verify the install**

```bash
python - <<'PY'
import autoneuronet as ann
x = ann.Var(2.0)
y = x * x
print(y)
PY
```
