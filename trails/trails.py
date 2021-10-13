from tensorflow.keras.preprocessing.image import save_img, img_to_array, array_to_img
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.applications import InceptionV3
from sklearn.preprocessing import minmax_scale
from distutils.dir_util import copy_tree
from sklearn.decomposition import NMF
from pathlib import Path
from umap import UMAP
import mimetypes
import argparse
import glob2
import json
import os

'''
TODO: Add --filter flag to remove objects without text/image properties
'''

config = {
  'inputs': None,
  'text': None,
  'limit': None,
  'sort': None,
  'n_neighbors': 4,
  'min_dist': 0.1,
  'vectorize': 'text',
  'output_folder': 'output',
}

def parse():
  '''Read command line args and begin data processing'''
  description = 'Create the data required to create a Trails viewer'
  parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--inputs', '-i', type=str, default=config['inputs'], help='path to a glob of files to process', required=True)
  parser.add_argument('--text', '-c', type=str, default=config['text'], help='field in metadata that contains body text (for text objects)', required=True)
  parser.add_argument('--limit', '-l', type=int, default=config['limit'], help='the maximum number of observations to analyze', required=False)
  parser.add_argument('--sort', '-s', type=str, default=config['sort'], help='the field to use when sorting objects', required=False)
  parser.add_argument('--vectorize', '-v', type=str, default=config['vectorize'], help='whether to vectorize text or images', required=False)
  config.update(vars(parser.parse_args()))
  validate_config(**config)
  process(**config)

def validate_config(**kwargs):
  assert kwargs['vectorize'] in ['text', 'image']

def process(**kwargs):
  # get a list of objects, one per datum to be represented
  kwargs['objects'] = get_objects(**kwargs)
  # get a list of vectors, one per datum to be represented
  kwargs['vectors'] = get_vectors(**kwargs)
  # project the vectors to 2D
  kwargs['positions'] = get_positions(**kwargs)
  # project the vectors to 1D or extract image colors
  kwargs['colors'] = get_colors(**kwargs)
  # write the outputs
  write_outputs(**kwargs)
  # fin
  print(' * done!')

##
# Get Objects
##

def get_objects(**kwargs):
  '''
  Format inputs into a dictionary stream

  Supported filetypes:
    * application/json
    x image/jpeg
    x image/gif
    x image/png
    x text/csv
    x text/plain
  '''
  objects = []
  for path in glob2.glob(kwargs['inputs']):
    mimetype = get_mimetype(path)
    # handle the case of multi-object mimetypes (csv, json)
    if mimetype == 'application/json':
      for o in get_json_objects(path, **kwargs):
        objects.append(o)
    else:
      print('WARNING: Only JSON objects are currently supported')
  # optionally sort the objects
  if kwargs.get('sort'):
    default = '' if isinstance(kwargs['sort'], str) else 0
    objects = sorted(objects, key=lambda i: i.get(kwargs['sort'], default), reverse=True)
  # optionally limit the objects
  if kwargs.get('limit'):
    objects = objects[:kwargs['limit']]
  return objects

def get_json_objects(path, **kwargs):
  with open(path) as f:
    j = json.load(f)
    # handle the case that j is an array of objects
    if isinstance(j, list):
      for i in j:
        yield i
    # handle the case that j is a single object
    elif isinstance(j, dict):
      yield j

def get_mimetype(path, full=True):
  '''Given a filepath, return the mimetype'''
  mime = mimetypes.MimeTypes().guess_type(path)[0]
  if mime and not full: mime = mime.split('/')[0]
  return mime

class Image(dict):
  '''
  A subclass of dict that permits lazy loading of images
  Usage im = Image({'path': 'url_str'}); im['image'] = keras img
  '''
  def __missing__(self, key):
    return load_img(self['path'])

##
# Vectorize Objects
##

def get_vectors(**kwargs):
  # TODO: create vector cache
  if kwargs['vectorize'] == 'image':
    base = InceptionV3(include_top=True, weights='imagenet',)
    model = Model(inputs=base.input, outputs=base.get_layer('avg_pool').output)
    vecs = []
    with tqdm(total=len(kwargs['objects'])) as progress_bar:
      for i in kwargs['objects']:
        im = i['image']
        im = preprocess_input( img_to_array( im.resize((299,299)) ) )
        progress_bar.update(1)
        vecs.append(model.predict(np.expand_dims(im, 0)).squeeze())
    return vecs
  elif kwargs['vectorize'] == 'text':
    text = [i.get(kwargs['text'], '') for i in kwargs['objects']]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    return NMF(n_components=100, max_iter=100, verbose=1, init='nndsvd').fit_transform(X)

##
# UMAP
##

def get_positions(**kwargs):
  return minmax_scale(run_umap(n_components=2, **kwargs), feature_range=(-1,1))

def get_colors(**kwargs):
  return minmax_scale(run_umap(n_components=1, **kwargs))

def run_umap(**kwargs):
  return UMAP(n_neighbors=kwargs['n_neighbors'],
              min_dist=kwargs['min_dist'],
              n_components=kwargs['n_components'],
              verbose=True).fit_transform(kwargs['vectors'])

##
# Write Outputs
##

def write_outputs(**kwargs):
  # make directories
  Path(os.path.join(kwargs['output_folder'], 'data')).mkdir(parents=True, exist_ok=True)
  # copy web assets
  src = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'web')
  dest = os.path.join(os.getcwd(), kwargs['output_folder'])
  copy_tree(src, dest)
  # write data
  write_json(os.path.join(kwargs['output_folder'], 'data', 'positions.json'), kwargs['positions'].tolist())
  write_json(os.path.join(kwargs['output_folder'], 'data', 'colors.json'), kwargs['colors'].tolist())
  write_json(os.path.join(kwargs['output_folder'], 'data', 'objects.json'), kwargs['objects'])

def write_json(path, obj):
  with open(path, 'w') as out:
    json.dump(obj, out)

if __name__ == '__main__':
  parse()