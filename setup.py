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
    'colorthief==0.2.1',
    'glob2==0.7',
    'imageio==2.9.0',
    'numba==0.54.0',
    'numpy~=1.19.2',
    'opencv-python==4.5.3.56',
    'Pillow==8.3.2',
    'scikit-learn==0.24.2',
    'tensorflow==2.6.0',
    'tqdm==4.46.1',
    'umap-learn==0.5.1',
    'yale-dhlab-keras-preprocessing>=1.1.1',
  ],
  entry_points={
    'console_scripts': [
      'trails=trails:parse',
    ],
  },
)