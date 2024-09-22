from setuptools import setup, find_packages

setup(
    name='rlstartdemo',
    version='0.1.0',
    author='Yinghan Sun',
    author_email='yinghansun2@gmail.com',
    description='A short description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # url='https://github.com/yourusername/your_project_name',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'requests',
    ],
    entry_points={
        'console_scripts': [
            'your_command=your_package.module:main_function',
        ],
    },
)