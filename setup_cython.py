from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import skeletonization.skeleton.io_tools

path = skeletonization.skeleton.io_tools.module_relative_path('skeletonization/skeleton/thinning.pyx')
ext_modules = [
    Extension('skeletonization.skeleton.thinning',
              [path],
              )]
setup(
    name='skeletonization.skeleton.thinning',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
