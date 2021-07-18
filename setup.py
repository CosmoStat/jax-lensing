from setuptools import setup

setup(name='jax-lensing',
      version='0.1',
      description='A JAX package for gravitational lensing',
      url='https://github.com/CosmoStat/jax-lensing',
      author='Benjamin Remy',
      author_email='benjamin.remy@cea.fr',
      license='MIT',
      packages=['jax_lensing'],
      install_requires=["astropy", 
                        "jax>=0.2.17", 
                        "optax>=0.0.9"
                        "dm-haiku>=0.0.4", 
                        "tensorflow-probability>=0.13.0",
                        "tensorflow-datasets>=4.3.0"],
      zip_safe=False)
