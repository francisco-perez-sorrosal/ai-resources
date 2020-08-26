from setuptools import setup

setup(
    name='deep_learning_library',
    version='0.1',
    description='My deep learning library',
    url='http://github.com/francisco-perez-sorrosal/ai-resources',
    author='Satanas',
    author_email='fperezsorrosal@gmail.com',
    license='Apache 2.0',
    packages=['deep_lib'],
    zip_safe=False,
    install_requires=[],
    entry_points='''
        [console_scripts]
    ''')
