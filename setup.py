from setuptools import setup

# import ``__version__`` from code base
exec(open('MPWeakSupervision/version.py').read())

setup(
    name='MPWeakSupervision',
    version=__version__,
    description='Weak Supervision For Morphological Profiling',
    author="Matthew O'Meara",
    author_email="maom@umich.edu",
    packages=['MPWeakSupervision',],
    install_requires=[
        'numpy>=1.12.1',
        'pyarrow>=3.0.0',
        'pandas>=1.2.3',
        'torch>=1.13.0',
        'scikit-learn>=0.24.1'],
    tests_require=['pytest'],
    url='http://github.com/maomlab/MPWeakSupervision',
    keywords='machine learning morphological profiling',
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9'],
)
