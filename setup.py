"""Build and bundle the native command-line programs during installation."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).parent.resolve()
PROGRAMS = {'camus-native': 'main.cpp', 'camus-count': 'count.cpp'}
LICENSES = ('PROJECT-LICENSE', 'NATIVE-LICENSE', 'SORT-LICENSE')


def build_assets(package):
    binary_dir = package / 'bin'
    binary_dir.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    flags = ['-O3', '-std=c++17', '-fopenmp', '-ffp-contract=off', '-fno-fast-math']
    for name, source in PROGRAMS.items():
        try:
            subprocess.run([*compiler, *flags, str(ROOT / 'native' / source),
                            '-o', str(binary_dir / name)], check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError('Installing from source requires a C++17 compiler with OpenMP support (GCC 11+).') from error
        (binary_dir / name).chmod(0o755)
    model_dir = package / 'models'
    model_dir.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'models/all-frequency.json', model_dir / 'all-frequency.json')
    license_dir = package / 'licenses'
    license_dir.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'LICENSE', license_dir / LICENSES[0])
    shutil.copyfile(ROOT / 'native/LICENSE', license_dir / LICENSES[1])
    shutil.copyfile(ROOT / 'native/SORT-LICENSE', license_dir / LICENSES[2])


class BuildPackage(build_py):
    def run(self):
        super().run()
        package = ROOT / 'camus_nn' if self.editable_mode else Path(self.build_lib) / 'camus_nn'
        build_assets(package)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        package = ROOT / 'camus_nn' if self.editable_mode else Path(self.build_lib) / 'camus_nn'
        return outputs + [str(package / 'bin' / name) for name in PROGRAMS] + [
            str(package / 'models/all-frequency.json'),
            *[str(package / 'licenses' / name) for name in LICENSES]]


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform = super().get_tag()
        return 'py3', 'none', platform


setup(distclass=NativeDistribution, cmdclass={'build_py': BuildPackage, 'bdist_wheel': PlatformWheel})
