# AutoNeuroNet - Automatic Differentiation and Neural Networks

## Setup

**AutoNeuroNet** is a C++ implementation of automatic differentiation, with custom matrices and a full neural network architecture and training pipeline. It comes with Python bindings through PyBind11, allowing for quick and easy development of networks through Python, backed with C++.

To build the necessary C++ dependencies for use in Python run:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
```

The C++ dependencies can then be accessed from the build/ folder and imported in any Python files.

For building the python package, run the following commands:

```
python -m pip uninstall -y autoneuronet # (if a preexisting package already exists)

python -m pip install -U pip build
python -m build
python -m pip install dist/*.whl
```

To build with twine to upload to PyPi, run the following commands:

```
python -m pip install -U build twine
python -m build
python -m twine check dist/*
```

Upload to TestPyPi first to test:

```
python -m twine upload -r testpypi dist/*
```

Upload to PyPi first to test:

```
python -m twine upload dist/*
```

Resources I used as reference:

- [What's Automatic Differentiation? - HuggingFace](https://huggingface.co/blog/andmholm/what-is-automatic-differentiation)
- [Differentiate Automatically](https://comp6248.ecs.soton.ac.uk/handouts/autograd-handouts.pdf)
- [Andrej Karpathy's MicroGrad](https://github.com/karpathy/micrograd)
