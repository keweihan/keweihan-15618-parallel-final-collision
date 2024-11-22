import argparse
import os
import subprocess
import socket

def is_ghc_cluster() -> bool:
    hostname = socket.gethostname()
    return hostname.startswith("ghc")

def run_command(command):
    """Run a shell command and print the output in real time."""
    try:
        print(f"Executing command: {command}")
        result = subprocess.run(command, shell=True, check=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing: {command}")
        print(e)
        exit(1)


def install_dependencies():
    """Install dependencies with Conan"""
    if is_ghc_cluster():
        run_command("bash -c 'source scripts/ghc_setup.sh'")
        return
    
    run_command(
        "conan install . --output-folder=. --build=missing -s build_type=Release"
    )
    run_command(
        "conan install . --output-folder=. --build=missing -s build_type=Debug"
    )


def build_release():
    """Configure CMake build and build for release"""
    build_dir = os.path.join(os.getcwd(), "build")
    os.makedirs(build_dir, exist_ok=True)
    is_windows = os.name == "nt"

    if is_windows:
        # Configure build system with CMake
        run_command(
            f"cmake -B build -DCMAKE_TOOLCHAIN_FILE='build/generators/conan_toolchain.cmake'"
        )
        # Build with CMake
        run_command(f"cmake --build {build_dir} --config Release")
    else:
        run_command(
            f"cmake -B build/Release -DCMAKE_TOOLCHAIN_FILE='build/Release/generators/conan_toolchain.cmake' -DCMAKE_BUILD_TYPE=Release"
        )
        run_command(f"cmake --build {build_dir}/Release")


def build_debug():
    """Configure CMake build and build for debug"""
    build_dir = os.path.join(os.getcwd(), "build")
    os.makedirs(build_dir, exist_ok=True)
    is_windows = os.name == "nt"

    if is_windows:
        # Configure build system with CMake
        run_command(
            f"cmake ./build -DCMAKE_TOOLCHAIN_FILE='conan_toolchain.cmake'"
        )
        # Build with CMake
        run_command(f"cmake --build {build_dir} --config Debug")
    else:
        run_command(
            f"cmake -B build/Debug -DCMAKE_TOOLCHAIN_FILE='build/Debug/generators/conan_toolchain.cmake' -DCMAKE_BUILD_TYPE=Debug"
        )
        run_command(f"cmake --build {build_dir}/Debug")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and install dependencies."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Install command
    subparsers.add_parser("install", help="Install dependencies")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build the project")
    build_parser.add_argument(
        "--type",
        choices=["release", "debug"],
        required=True,
        help="Type of build: release or debug",
    )

    return parser.parse_args()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and install dependencies."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Install command
    subparsers.add_parser("install", help="Install dependencies")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build the project")
    build_parser.add_argument(
        "--type",
        choices=["release", "debug"],
        required=True,
        help="Type of build: release or debug",
    )

    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    if args.command == "install":
        install_dependencies()
    elif args.command == "build":
        if args.type == "release":
            build_release()
        elif args.type == "debug":
            build_debug()
    else:
        parser.print_help()
        exit(1)
