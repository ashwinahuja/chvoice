from setuptools import setup, find_packages

setup(
  name = 'speech-enhancement',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'L46 Assignment, Speech Enhancement',
  author = 'Harry Songhurst and Ashwin Ahuja',
  author_email = 'harrysonghurst@gmail.com',
  url = 'https://github.com/indrasweb/speech-enhancement',
  keywords = ['speech enhancement', 'model compression'],
  install_requires=[
      'librosa>=0.8.0'
  ],
  classifiers=[
      'Development Status :: Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.7.9',
  ],
)