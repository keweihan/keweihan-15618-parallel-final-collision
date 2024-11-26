#!/bin/bash

# ------------------INSTALL CONAN ------------------- #
pip install conan

CONFIG_FILE="$HOME/.bashrc"  
LINE_TO_ADD='export PATH=$HOME/.local/bin:$PATH'

# Check if the line is already in the file
if ! grep -Fxq "$LINE_TO_ADD" "$CONFIG_FILE"; then
  echo "$LINE_TO_ADD" >> "$CONFIG_FILE"
  echo "Added line to $CONFIG_FILE"
else
  echo "Line $LINE_TO_ADD already exists in $CONFIG_FILE"
fi

export PATH=$HOME/.local/bin:$PATH

conan profile detect
conan install --requires=imgui/1.90.6-docking -g CMakeDeps -g CMakeToolchain --build=missing --output-folder=build/Release/generators -s build_type=Release
conan install --requires=imgui/1.90.6-docking -g CMakeDeps -g CMakeToolchain --build=missing --output-folder=build/Debug/generators -s build_type=Debug


# ------------------ADD CUDA TO PATH (From asst2 instructions)------------------- #
LINE_TO_ADD='export PATH=/usr/local/cuda-11.7/bin:${PATH}'

# Check if the line is already in the file
if ! grep -Fxq "$LINE_TO_ADD" "$CONFIG_FILE"; then
  echo "$LINE_TO_ADD" >> "$CONFIG_FILE"
  echo "Added line to $CONFIG_FILE"
else
  echo "Line $LINE_TO_ADD already exists in $CONFIG_FILE"
fi

export PATH=/usr/local/cuda-11.7/bin:${PATH}


# ------------------INSTALL ------------------- #
LINE_TO_ADD="source $HOME/.bashrc"

if ! grep -Fxq "$LINE_TO_ADD" "$HOME/.bash_profile"; then
  echo "$LINE_TO_ADD" >> "$CONFIG_FILE"
  echo "Added line to $CONFIG_FILE"
else
  echo "Line $LINE_TO_ADD already exists in $CONFIG_FILE"
fi