from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:
    raise SystemExit("请先安装 Cython：python -m pip install Cython") from exc


extensions = [
    Extension(
        "uva_model._tokenizer_accel",
        ["uva_model/_tokenizer_accel.pyx"],
    )
]


setup(
    name="unified-variational-attention",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False,
)
