from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, source_dir: str = "cpp") -> None:
        super().__init__(name, sources=[])
        self.source_dir = str(Path(source_dir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        config = "Release"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}",
            f"-DPython3_EXECUTABLE={self.get_executable()}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            "-DTURBOBUS_BUILD_PYTHON=ON",
            f"-DCMAKE_BUILD_TYPE={config}",
        ]
        build_args = ["--config", config]

        subprocess.check_call(["cmake", ext.source_dir, *cmake_args], cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", *build_args], cwd=build_temp)

    def get_executable(self) -> str:
        return os.environ.get("PYTHON", sys.executable)


setup(
    ext_modules=[CMakeExtension("turbobus._turbobus")],
    cmdclass={"build_ext": CMakeBuild},
)
