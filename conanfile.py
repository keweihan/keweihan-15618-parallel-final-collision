from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain, cmake_layout


class SimpleECSConan(ConanFile):
    name = "SimpleECS"
    version = "1.0"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"
    default_options = {
        "boost/*:shared": True,
    }
        
    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_CXX_STANDARD"] = "17"    
        tc.generate()
            
    def requirements(self):
        # Specify all required dependencies
        self.requires("sdl/2.28.3", override=True)
        self.requires("sdl_ttf/2.20.1")
        self.requires("sdl_mixer/2.8.0")
        self.requires("sdl_image/2.6.3")
        self.requires("imgui/1.90.6-docking")
        self.requires("b2/5.2.0")
        self.requires("gtest/1.14.0")
        self.requires("boost/1.86.0")

    def deploy(self):
        self.copy("*.dll", src="bin", dst="bin")
       
    def layout(self):
        cmake_layout(self)