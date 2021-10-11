from setuptools import setup
import os

# populate list of all paths in `./trails/web`
web = []
for root, subdirs, files in os.walk(os.path.join('trails', 'web')):
  for file in (files or []):
    web.append(
      os.path.join(
        root.replace('trails/', '')
            .replace('trails\\', ''),
        file
      )
    )

setup(
  name='trails',
  version='0.0.1',
  packages=['trails'],
  package_data={
    'trails': web,
  },
  keywords=['webgl',
            'three.js',
            'tensorflow',
            'machine-learning'],
  description='Visualize massive image collections with WebGL',
  url='https://github.com/yaledhlab/trails',
  author='Douglas Duhaime',
  author_email='douglas.duhaime@gmail.com',
  license='MIT',
  install_requires=[
    'glob2==0.7',
  ],
  entry_points={
    'console_scripts': [
      'trails=trails:parse',
    ],
  },
)