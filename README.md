# Collision Stress Test
Basic collision stress scene.

## Getting Started (GHC Linux)
Note: GHC has cmake, python preinstalled and in PATH. 

### Install dependencies
1. Run installer helper `python3 scripts/build.py install`

### Build and run
1. Build executable `python3 scripts/build.py build --type release`
    - Use `debug` for debuggable version (i.e. can use lldb.)
2. Execute runnable with `./build/bin/collisionStress`

## Getting Started (Mac/Windows)

1. install `python` (>=3.8)
2. [conan ](https://conan.io/downloads) 
    - ex. `pip install conan`
3. [cmake](https://cmake.org/download/) 
    - Ubuntu: `sudo apt install -y cmake`
4. Run installer helper `python ./scripts/build.py install`

### Build and run
1. Configure scene in `demos/collisionStress.cpp`. E.g. run headless (no window/rendering) or with window 
2. Build executable `python src/build.py build --type release`
    - Use `debug` for debuggable version. 
3. Execute runnable in `./build/bin/collisionStress`

## Notes
Ensure no spaces in directory names. 