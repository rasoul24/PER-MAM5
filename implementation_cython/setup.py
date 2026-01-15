from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# On définit les extensions explicitement pour plus de contrôle
extensions = [
    Extension("medial_axis", ["medial_axis.pyx"]),
    Extension("description_image", ["description_image.pyx"]),
    Extension("bounding_box", ["bounding_box.pyx"]),
]

setup(
    name="MonProjetCython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"}
    ),
    include_dirs=[np.get_include()],
)