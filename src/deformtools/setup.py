from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension(
        name="deformtools.haversine",
        sources=["deformtools/haversine.pyx"],
        # include_dirs=['/some/path/to/include/'], # not needed for fftw unless it is installed in an unusual place
        # include_dirs=[gsw.get_include()]
        # library_dirs=['/some/path/to/include/'], # not needed for fftw unless it is installed in an unusual place
    ),
]

setup(
   name='deformtools',
   version='1.0',
   description='A module containing tools to analyse velocity gradients from drifters.',
   author='Sebastian Essink',
   author_email='sebastianessink@gmail.com',
   packages=['deformtools'],  #same as name
   # install_requires=['pandas', 'numpy'], #external packages as dependencies,
   ext_modules= cythonize(extensions),
   zip_safe=False
)
