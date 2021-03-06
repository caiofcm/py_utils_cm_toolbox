import setuptools
from utils_cm_toolbox.version import Version


setuptools.setup(name='utils_cm_toolbox',
                 version=Version('1.0.0').number,
                 description='Python Package Boilerplate',
                 long_description=open('README.md').read().strip(),
                 author='Package Author',
                 author_email='caiocuritiba@gmail.com',
                 url='http://path-to-my-packagename',
				 packages=["utils_cm_toolbox"],
                 #py_modules=['packagename'],
                 install_requires=[],
                 license='MIT License',
                 zip_safe=False,
                 keywords='boilerplate package',
                 classifiers=['Packages', 'Boilerplate'])
