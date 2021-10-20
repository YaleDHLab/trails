from tensorflow.keras.preprocessing.image import save_img, img_to_array, array_to_img
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import InceptionV3
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import NMF, PCA
from tensorflow.keras.models import Model
from distutils.dir_util import copy_tree
from colorthief import ColorThief
from multiprocessing import Pool
from functools import partial
from gzip import GzipFile
from pathlib import Path
from tqdm import tqdm
from umap import UMAP
import numpy as np
import mimetypes
import argparse
import shutil
import glob2
import uuid
import json
import csv
import os

'''
TODO:
  Add --filter flag to remove objects without text/image properties
  Add --template arg that selects from available templates (text, image, text+image)
  Add --vectors flag with one vector per object
  Set the vectorize argument to image if all inputs are images
  Write config.json with kwargs for ui conditionalization
  Conditionalize the default template selected
  Multiprocess image color extraction
  Given just text files, create objects with filename attribute
  Lazy load objects in DOM rather than loading all in objects.json
'''

config = {
  'inputs': None,
  'text': None,
  'limit': None,
  'sort': None,
  'metadata': None,
  'n_neighbors': 5,
  'min_dist': 0.1,
  'vectorize': None,
  'output_folder': 'output',
  'color_quality': 10,
  'plot_id': str(uuid.uuid1()),
}

def parse():
  '''Read command line args and begin data processing'''
  description = 'Create the data required to create a Trails viewer'
  parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--inputs', '-i', type=str, default=config['inputs'], help='path to a glob of files to process', required=True)
  parser.add_argument('--text', '-c', type=str, default=config['text'], help='field in metadata that contains body text (for text objects)', required=False)
  parser.add_argument('--limit', '-l', type=int, default=config['limit'], help='the maximum number of observations to analyze', required=False)
  parser.add_argument('--sort', '-s', type=str, default=config['sort'], help='the field to use when sorting objects', required=False)
  parser.add_argument('--vectorize', '-v', type=str, default=config['vectorize'], help='whether to vectorize text or images', required=False)
  parser.add_argument('--metadata', '-m', type=str, default=config['metadata'], help='metadata JSON for image inputs', required=False)
  config.update(vars(parser.parse_args()))
  validate_config(**config)
  process(**config)

def validate_config(**kwargs):
  # check vectorization strategy
  assert kwargs.get('vectorize') in ['text', 'image', None]
  # check metadata
  metadata_mimetype = None if not kwargs['metadata'] else get_mimetype(kwargs['metadata'])
  if metadata_mimetype: assert metadata_mimetype in ['application/json', 'text/csv']
  # check color_quality
  assert isinstance(kwargs['color_quality'], int) and kwargs['color_quality'] >= 1

def process(**kwargs):
  # create output directories
  print(' * preparing directories')
  create_directories(**kwargs)
  # get a metadata map, one k/v pair per datum to be represented
  print(' * collecting metadata')
  kwargs['metadata'] = get_metadata(**kwargs)
  # get a list of objects, one per datum to be represented
  print(' * collecting objects')
  kwargs['objects'] = get_objects(**kwargs)
  # determine the vectorization type
  print(' * determining field to vectorize')
  kwargs['vectorize'] = get_vectorize(**kwargs)
  # get a list of vectors, one per datum to be represented
  print(' * collecting vectors using {} data'.format(kwargs['vectorize']))
  kwargs['vectors'] = get_vectors(**kwargs)
  # project the vectors to 2D
  print(' * collecting positions')
  kwargs['positions'] = get_positions(**kwargs)
  # project the vectors to 1D or extract image colors
  print(' * collecting colors')
  kwargs['colors'] = get_colors(**kwargs)
  # write the outputs
  print(' * writing outputs')
  write_outputs(**kwargs)
  # fin
  print(' * done!')

def create_directories(**kwargs):
  Path(os.path.join('cache', 'image-vectors')).mkdir(parents=True, exist_ok=True)
  Path(os.path.join('cache', 'image-colors')).mkdir(parents=True, exist_ok=True)
  Path(os.path.join(kwargs['output_folder'], 'data')).mkdir(parents=True, exist_ok=True)

##
# Get Metadata
##

def get_metadata(**kwargs):
  if not kwargs['metadata']: return {}
  metadata = {}
  mimetype = get_mimetype(kwargs['metadata'])
  if mimetype == 'text/csv':
    with open(kwargs['metadata']) as f:
      reader = csv.reader(f)
      headers = next(reader)
      # the first row of CSVs must contain headers, and the first column must be filename
      assert headers[0].lower() == 'filename'
      for row in reader:
        row = {headers[idx]: i for idx, i in enumerate(row)}
        metadata[ row['filename'] ] = row
  elif mimetype == 'application/json':
    with open(kwargs['metadata']) as f:
      for i in json.load(f):
        assert 'filename' in i
        metadata[ i['filename'] ] = i
  return metadata

##
# Get Objects
##

def get_objects(**kwargs):
  '''
  Format inputs into a dictionary stream

  Supported mimetypes:
    * text/plain
    * application/json
    * image/jpeg
    * image/png
    x image/gif
    x text/csv
  '''
  objects = []
  unparsed = []
  paths = glob2.glob(kwargs['inputs'])
  with tqdm(total=len(paths)) as progress_bar:
    for path in paths:
      mimetype = get_mimetype(path)
      mimetype_base = mimetype.split('/')[0] if mimetype else ''
      # JSON inputs
      if mimetype == 'application/json':
        for o in get_json_objects(path, **kwargs):
          objects.append(o)
      # plaintext inputs
      elif mimetype == 'text/plain':
        o = get_plaintext_object(path, **kwargs)
        objects.append(o)
      # image inputs
      elif mimetype_base == 'image':
        o = get_image_object(path, **kwargs)
        if o: objects.append(o)
      else:
        unparsed.append(path)
      progress_bar.update(1)
      # halt if we get enough
      if kwargs.get('limit') and len(objects) >= kwargs['limit']:
        break
  # throw an error if no objects are present
  if not objects:
    raise Exception('No inputs were found! Please check the value provided to --input')
  # warn user about unparsed objects
  if unparsed:
    print('WARNING: Only JSON and images are currently supported. The following were not processed:')
    print(' '.join(unparsed))
  # optionally sort the objects
  if kwargs.get('sort'):
    default = '' if isinstance(kwargs['sort'], str) else 0
    objects = sorted(objects, key=lambda i: i.get(kwargs['sort'], default), reverse=True)
  # optionally limit the number of objects
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

def get_plaintext_object(path, **kwargs):
  with open(path) as f:
    return Plaintext(text=f.read(), metadata=dict(kwargs['metadata'].get(path, {}), **{
      'filename': os.path.basename(path),
    }))

def get_image_object(path, **kwargs):
  try:
    im = Image(path=path, metadata=kwargs['metadata'].get(os.path.basename(path), {}))
    # make sure images load
    return im if im['image'] else None
  except:
    pass

def get_mimetype(path, full=True):
  '''Given a filepath, return the mimetype'''
  mime = mimetypes.MimeTypes().guess_type(path)[0]
  if mime and not full: mime = mime.split('/')[0]
  return mime

class Image(dict):
  '''
  A subclass of dict that permits lazy loading of images
  Usage im = Image(path='url_str'); im['image'] = keras img
  '''
  def __missing__(self, key):
    return load_img(self['path'])

class Plaintext(dict):
  pass

##
# Vectorize Objects
##

def get_vectorize(**kwargs):
  if kwargs.get('vectorize'): return kwargs['vectorize']
  if all([isinstance(i, Image) for i in kwargs['objects']]): return 'image'
  return 'text'

def get_vectors(**kwargs):
  if kwargs['vectorize'] == 'image':
    base = InceptionV3(include_top=True, weights='imagenet',)
    model = Model(inputs=base.input, outputs=base.get_layer('avg_pool').output)
    vecs = []
    with tqdm(total=len(kwargs['objects'])) as progress_bar:
      for i in kwargs['objects']:
        cache_path = os.path.join('cache', 'image-vectors', i['path'].replace('/', '-'))
        if os.path.exists(cache_path + '.npy'):
          vecs.append(np.load(cache_path + '.npy'))
          progress_bar.update(1)
          continue
        try:
          # this next line uses the __missing__ handler to lazily load the image into RAM
          im = preprocess_input( img_to_array( i['image'].resize((299,299)) ) )
          vec = model.predict(np.expand_dims(im, 0)).squeeze()
          vecs.append(vec)
          progress_bar.update(1)
          np.save(cache_path, vec)
        except Exception as exc:
          print('WARNING: Image', i, 'could not be vectorized')
    return vecs
  elif kwargs['vectorize'] == 'text':
    text = [i[kwargs['text'] if kwargs['text'] else 'text'] for i in kwargs['objects']]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    return NMF(n_components=100, max_iter=50, verbose=1, init='nndsvd').fit_transform(X)

##
# UMAP
##

def get_positions(**kwargs):
  return minmax_scale(run_umap(n_components=2, **kwargs), feature_range=(-1,1))

def run_umap(**kwargs):
  vecs = PCA(n_components=50).fit_transform(kwargs['vectors'])
  return UMAP(n_neighbors=kwargs['n_neighbors'],
              min_dist=kwargs['min_dist'],
              n_components=kwargs['n_components'],
              verbose=True).fit_transform(vecs)

##
# Colors
##

def get_colors(**kwargs):
  # for text data, return 1D umap
  if kwargs['vectorize'] == 'text':
    return minmax_scale(run_umap(n_components=1, **kwargs))
  # for image data, return the dominant color for each image
  elif kwargs['vectorize'] == 'image':
    return get_all_image_colors(**kwargs)

def get_all_image_colors(**kwargs):
  colors = [None for _ in kwargs['objects']]
  with tqdm(total=len(kwargs['objects'])) as progress_bar:
    pool = Pool()
    l = [[idx, i, kwargs['color_quality']] for idx, i in enumerate(kwargs['objects'])]
    for i in pool.imap(get_image_colors, l):
      idx, color = i
      colors[idx] = color
      progress_bar.update(1)
    pool.close()
    pool.join()
  return np.array(colors)

def get_image_colors(args):
  idx, obj, quality = args
  cache_path = os.path.join('cache', 'image-colors', obj['path'].replace('/', '-'))
  if os.path.exists(cache_path + '.npy'):
    color = np.load(cache_path + '.npy')
  else:
    color = ColorThief(obj['path']).get_color(quality=quality)
    np.save(cache_path + '.npy', color)
  return [idx, color]

##
# Write Outputs
##

def write_outputs(**kwargs):
  # copy web assets
  src = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'web')
  dest = os.path.join(os.getcwd(), kwargs['output_folder'])
  copy_tree(src, dest)
  # copy media before mutating objects below
  copy_media(**kwargs)
  # write data
  kwargs['objects'] = format_objects(**kwargs)
  positions_path = os.path.join(kwargs['output_folder'], 'data', 'positions.json')
  colors_path = os.path.join(kwargs['output_folder'], 'data', 'colors.json')
  objects_path = os.path.join(kwargs['output_folder'], 'data', 'objects.json')
  write_json(positions_path, round_floats(kwargs['positions'].tolist()), gzip=True)
  write_json(colors_path, kwargs['colors'].squeeze().tolist(), gzip=True)
  write_json(objects_path, kwargs['objects'], gzip=True)

def format_objects(**kwargs):
  # format user-provided objects for writing
  if kwargs['vectorize'] == 'text':
    return kwargs['objects']
  elif kwargs['vectorize'] == 'image':
    for i in kwargs['objects']:
      i['path'] = os.path.basename(i['path'])
      if not i['metadata']: del i['metadata']
    return kwargs['objects']

def round_floats(obj, digits=4):
  '''Return 2D array obj with rounded float precision'''
  return [[round(float(j), digits) for j in i] for i in obj]

def write_json(path, obj, gzip=False):
  if gzip:
    if not path.endswith('.gz'): path += '.gz'
    with GzipFile(path, 'w') as out:
      out.write(json.dumps(obj).encode('utf-8'))
  else:
    with open(path, 'w') as out:
      json.dump(obj, out)

def copy_media(thumb_size=100, **kwargs):
  if kwargs['vectorize'] == 'image':
    print(' * copying media')
    thumbs_dir = os.path.join(kwargs.get('output_folder'), 'data', 'thumbs')
    originals_dir = os.path.join(kwargs.get('output_folder'), 'data', 'originals')
    Path(thumbs_dir).mkdir(parents=True, exist_ok=True)
    Path(originals_dir).mkdir(parents=True, exist_ok=True)
    with tqdm(total=len(kwargs['objects'])) as progress_bar:
      for i in kwargs['objects']:
        name = os.path.basename(i['path'])
        # process originals
        shutil.copy(i['path'], os.path.join(originals_dir, name))
        # process thumbs
        im = i['image']
        w, h = im.size
        width = thumb_size
        height = int(h * (thumb_size / w))
        save_img(os.path.join(thumbs_dir, name), im.resize((width, height)))
        progress_bar.update(1)

if __name__ == '__main__':
  parse()