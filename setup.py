import setuptools
import os
import pathlib

__packagename__ = 'robotools'


def package_files(directory):
    assert os.path.exists(directory)
    fp_typed = pathlib.Path(__packagename__, 'py.typed')
    fp_typed.touch()
    paths = [str(fp_typed)]
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


def get_version():
    import os, re
    VERSIONFILE = os.path.join(__packagename__, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))

__version__ = get_version()

setuptools.setup(name = __packagename__,
        packages = setuptools.find_packages(), # this must be the same as the name above
        version=__version__,
        description='Standard workflows and convenience methods for Janus and FreedomEVO robots.',
        url='https://jugit.fz-juelich.de/IBG-1/biopro/robotools',
        download_url = 'https://jugit.fz-juelich.de/IBG-1/biopro/robotools/tarball/%s' % __version__,
        author='DigInBio Contributors',
        author_email='m.osthege@fz-juelich.de',
        copyright='(c) 2019 Forschungszentrum Juelich GmbH',
        license='(c) 2019 Forschungszentrum Juelich GmbH',
        classifiers= [
            'Programming Language :: Python',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Intended Audience :: Developers'
        ],
        install_requires=[
            'numpy',
            'pandas',
        ],
        package_data={
            'robotools': package_files(str(pathlib.Path(pathlib.Path(__file__).parent, 'robotools').absolute()))
        },
        include_package_data=True
)
