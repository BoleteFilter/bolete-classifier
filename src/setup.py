from setuptools import setup
from Cython.Build import cythonize

# directives = ({"linetrace": False, "language_level": 3},)
setup(
    name="Performance app",
    ext_modules=cythonize(["performance.pyx", "lookalikes.pyx"], language_level=3),
    zip_safe=False,
    # compiler_directives={"language_level": "3"},
)
