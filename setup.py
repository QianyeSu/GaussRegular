from setuptools import Extension, setup
import os
import numpy as np

# Default to fast-math (can be disabled with GAUSSREGULAR_FAST_MATH=0).
fast_math = os.environ.get("GAUSSREGULAR_FAST_MATH", "1") == "1"

if os.name == "nt":
    compile_args = ["/O2", "/GL", "/Oi", "/Ot"]
    if fast_math:
        compile_args.append("/fp:fast")
    else:
        compile_args.append("/fp:precise")
    link_args = ["/LTCG"]
else:
    compile_args = ["-O3", "-fno-math-errno"]
    if fast_math:
        compile_args.append("-ffast-math")
    link_args = []

ext_modules = [
    Extension(
        "gaussregular._core",
        sources=["src/gaussregular/_core.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(ext_modules=ext_modules)
