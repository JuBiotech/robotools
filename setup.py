# Copyright 2021 Forschungszentrum Jülich GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import pathlib
import re

import setuptools

__packagename__ = "robotools"


def package_files(directory):
    assert pathlib.Path(directory).exists()
    fp_typed = pathlib.Path(__packagename__, "py.typed")
    fp_typed.touch()
    paths = [str(fp_typed.absolute())]
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(str(pathlib.Path("..", path, filename)))
    return paths


def get_version():
    VERSIONFILE = pathlib.Path(__packagename__, "__init__.py")
    initfile_lines = open(VERSIONFILE, "rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


__version__ = get_version()

setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),  # this must be the same as the name above
    version=__version__,
    description="Pythonic in-silico liquid handling and creation of Tecan FreedomEVO worklists.",
    url="https://github.com/jubiotech/robotools",
    download_url="https://github.com/jubiotech/robotools/tarball/%s" % __version__,
    author="Michael Osthege",
    author_email="m.osthege@fz-juelich.de",
    license="GNU Affero General Public License v3",
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=[
        "numpy",
        "pandas",
    ],
    package_data={
        "robotools": package_files(str(pathlib.Path(pathlib.Path(__file__).parent, "robotools").absolute()))
    },
    include_package_data=True,
)
