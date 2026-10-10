from distutils.core import setup

from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
ext_modules = [
    Extension(
        "merge_superpxiel",  # 这里定义了生成的文件名
        sources=["merge_superpxiel.pyx"],
        language="c++",
    )
]

setup(
    name="my_package",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()],
    options={'build_ext': {'build_lib': '../'}}
)
