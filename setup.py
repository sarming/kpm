from Cython.Build import cythonize
from setuptools import setup

if __name__ == "__main__":
    setup(
        ext_modules=cythonize(
            "kpm/chebyshev.py", compiler_directives={'language_level': "3"}, annotate=True
        )
    )
