import platform
import os
import sys
import shutil
from setuptools import setup, Extension 
from distutils.command.clean import clean as Clean
import numpy

# Version number
version = '0.2.24'


def readme():
    with open('README.md') as f:
       return f.read()

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

class CleanCommand(Clean):
    description = "Remove build directories, and compiled files (including .pyc)"

    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                if (   filename.endswith('.so')
                    or filename.endswith('.pyd')
                    #or filename.find("wrap_qfc.cpp") != -1 # remove automatically generated source file
                    #or filename.endswith('.dll')
                    or filename.endswith('.pyc')
                                ):
                    tmp_fn = os.path.join(dirpath, filename)
                    print "removing", tmp_fn
                    os.unlink(tmp_fn)

# set up macro
if platform.system() == "Darwin":
    macros = [("__APPLE__", "1")]
elif "win" in platform.system().lower():
    macros = [("_WIN32", "1")]
else:
    macros = [("_UNIX", "1")]

#see http://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
if use_cython:
    ext_modules = [Extension(name="fastlmmhpc.util.stats.quadform.qfc_src.wrap_qfc",
                             language="c++",
                             sources=["fastlmmhpc/util/stats/quadform/qfc_src/wrap_qfc.pyx", "fastlmmhpc/util/stats/quadform/qfc_src/QFC.cpp"],
                             include_dirs=[numpy.get_include()],
                             define_macros=macros)]
    cmdclass = {'build_ext': build_ext, 'clean': CleanCommand}
else:
    ext_modules = [Extension(name="fastlmmhpc.util.stats.quadform.qfc_src.wrap_qfc",
                             language="c++",
                             sources=["fastlmmhpc/util/stats/quadform/qfc_src/wrap_qfc.cpp", "fastlmmhpc/util/stats/quadform/qfc_src/QFC.cpp"],
                             include_dirs=[numpy.get_include()],
                             define_macros=macros)]
    cmdclass = {}

#python setup.py sdist bdist_wininst upload
setup(
    name='fastlmmhpc',
    version=version,
    description='Fast GWAS',
    long_description=readme(),
    keywords='gwas bioinformatics LMMs MLMs',
    url="http://research.microsoft.com/en-us/um/redmond/projects/mscompbio/fastlmmhpc/",
    author='MSR',
    author_email='martineh@uji.es',
    license='Apache 2.0',
    packages=[
        "fastlmmhpc/association/tests",
        "fastlmmhpc/association",
        "fastlmmhpc/external/util",
        "fastlmmhpc/external",
        "fastlmmhpc/feature_selection",
        "fastlmmhpc/inference",
        "fastlmmhpc/pyplink/altset_list", #old snpreader
        "fastlmmhpc/pyplink/snpreader", #old snpreader
        "fastlmmhpc/pyplink/snpset", #old snpreader
        "fastlmmhpc/pyplink", #old snpreader
        "fastlmmhpc/util/runner",
        "fastlmmhpc/util/stats/quadform",
        "fastlmmhpc/util/stats/quadform/qfc_src",
        "fastlmmhpc/util/standardizer",
        "fastlmmhpc/util/stats",
        "fastlmmhpc/util",
        "fastlmmhpc",
    ],
    package_data={"fastlmmhpc/association" : [
                       "Fastlmmhpc_autoselect/FastlmmhpcC.exe",
                       "Fastlmmhpc_autoselect/libiomp5md.dll",
                       "Fastlmmhpc_autoselect/fastlmmhpcc",
                       "Fastlmmhpc_autoselect/FastlmmhpcC.Manual.pdf"],
                  "fastlmmhpc/feature_selection" : [
                       "examples/bronze.txt",
                       "examples/ScanISP.Toydata.config.py",
                       "examples/ScanLMM.Toydata.config.py",
                       "examples/ScanOSP.Toydata.config.py",
                       "examples/toydata.5chrom.bed",
                       "examples/toydata.5chrom.bim",
                       "examples/toydata.5chrom.fam",
                       "examples/toydata.bed",
                       "examples/toydata.bim",
                       "examples/toydata.cov",
                       "examples/toydata.dat",
                       "examples/toydata.fam",
                       "examples/toydata.iidmajor.hdf5",
                       "examples/toydata.map",
                       "examples/toydata.phe",
                       "examples/toydata.shufflePlus.phe",
                       "examples/toydata.sim",
                       "examples/toydata.snpmajor.hdf5",
                       "examples/toydataTest.phe",
                       "examples/toydataTrain.phe"
                       ]
                 },
    install_requires = ['scipy>=0.16.0', 'numpy>=1.9.3', 'pandas>=0.16.2','matplotlib>=1.4.3', 'scikit-learn>=0.16.1', 'pysnptools>=0.3.8', 'dill'],
    cmdclass = cmdclass,
    ext_modules = ext_modules,
  )
