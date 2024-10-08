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
from urllib.parse import unquote
from bs4 import BeautifulSoup
from functools import partial
from gzip import GzipFile
from pathlib import Path
from tqdm import tqdm
from umap import UMAP
import numpy as np
import mimetypes
import argparse
import codecs
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
  Write config.json with kwargs for ui conditionalization
  Multiprocess image color extraction
'''

config = {
  'inputs': None,
  'text': None,
  'label': None,
  'vector': None,
  'limit': None,
  'metadata': None,
  'sort': None,
  'position_x': None,
  'position_y': None,
  'n_neighbors': 5,
  'min_dist': 0.1,
  'vectorize': 'auto',
  'output_folder': 'output',
  'color_quality': 10,
  'max_iter': 50,
  'encoding': 'utf8',
  'xml_base_tag': None,
  'plot_id': str(uuid.uuid1()),
}

def parse():
  '''Read command line args and begin data processing'''
  description = 'Create the data required to create a Trails viewer'
  parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--inputs', '-i', type=str, default=config['inputs'], help='path to a glob of files to process', required=True)
  parser.add_argument('--text', type=str, default=config['text'], help='attribute or field that contains "body" text', required=False)
  parser.add_argument('--label', type=str, default=config['label'], help='attribute or field that contains "label" text', required=False)
  parser.add_argument('--vector', type=list, default=config['vector'], help='attribute or field that contains object vector', required=False)
  parser.add_argument('--limit', '-l', type=int, default=config['limit'], help='the maximum number of observations to analyze', required=False)
  parser.add_argument('--metadata', '-m', type=str, default=config['metadata'], help='CSV or JSON metadata for objects (joined on filename)', required=False)
  parser.add_argument('--sort', '-s', type=str, default=config['sort'], help='the metadata field to use when sorting objects', required=False)
  parser.add_argument('--position_x', '-x', type=str, default=config['position_x'], help='the metadata field that designates the x position to use for points (also requires --position_y)', required=False)
  parser.add_argument('--position_y', '-y', type=str, default=config['position_y'], help='the metadata field that designates the y position to use for points (also requires --position_x)', required=False)
  parser.add_argument('--encoding', type=str, default=config['encoding'], help='the encoding to use when parsing text files', required=False)
  parser.add_argument('--max_iter', '-mi', type=int, default=config['max_iter'], help='the max number of NMF iterations when vectorizing text', required=False)
  parser.add_argument('--vectorize', '-v', type=str, default=config['vectorize'], help='whether to vectorize text or images', required=False)
  parser.add_argument('--xml_base_tag', type=str, default=config['xml_base_tag'], help='the XML tag that contains text to parse', required=False)
  parser.add_argument('--output_folder', '-o', type=str, default=config['output_folder'], help='folder in which outputs will be written', required=False)
  config.update(preprocess_config(**vars(parser.parse_args())))
  validate_config(**config)
  process(**config)

def preprocess_config(**kwargs):
  # perform inferences on configuration from initial parameters
  if kwargs.get('vectorize') == 'auto' and kwargs.get('text'):
    kwargs['vectorize'] = 'text'
  return kwargs

def validate_config(**kwargs):
  # check vectorization strategy
  assert kwargs.get('vectorize') in ['text', 'image', 'auto']
  # check metadata
  metadata_mimetype = None if not kwargs['metadata'] else get_mimetype(kwargs['metadata'])
  if metadata_mimetype: assert metadata_mimetype in ['application/json', 'text/csv']
  # check color_quality
  assert isinstance(kwargs['color_quality'], int) and kwargs['color_quality'] >= 1
  # check both position_x and position_y are present if either is present
  if kwargs.get('position_x'): assert kwargs.get('position_y')
  if kwargs.get('position_y'): assert kwargs.get('position_x')

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
  kwargs['vectorize'] = get_vectorize_strategy(**kwargs)
  # get a list of vectors, one per datum to be represented
  print(' * collecting vectors using {} data'.format(kwargs['vectorize']))
  kwargs['objects'], kwargs['vectors'] = get_vectors(**kwargs)
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
        row = standardize_metadata_row(row, **kwargs)
        metadata[ format_filename(row['filename']) ] = row
  elif mimetype == 'application/json':
    with open(kwargs['metadata']) as f:
      for i in json.load(f):
        assert 'filename' in i
        metadata[ format_filename(i['filename']) ] = standardize_metadata_row(i, **kwargs)
  return metadata

def standardize_metadata_row(d, **kwargs):
  standardized = {}
  for k, v in d.items():
    # the primary key must always be added, even if it's used for text or label
    if k == 'filename':
      v = format_filename(v)
      standardized[k] = v
    for w in ['label', 'text']:
      if k == kwargs.get(w):
        k = w
    standardized[k] = v
  return standardized

def format_filename(s):
  return '_'.join(
      unquote(s).split()
    ) \
    .replace('/', '_') \
    .replace('\\', '_')

##
# Get Objects
##

def get_objects(**kwargs):
  '''
  Format inputs into a dictionary stream

  Supported mimetypes:
    * text/plain
    * text/xml
    * text/html
    * application/json
    * image/jpeg
    * image/png
    x image/gif
    x text/csv
  '''
  objects = []
  unparsed = []
  paths = glob2.glob(kwargs['inputs'])
  total = min(len(paths), kwargs['limit']) if kwargs.get('limit') else len(paths)
  with tqdm(total=total) as progress_bar:
    for path in paths:
      mimetype = get_mimetype(path)
      mimetype_base = mimetype.split('/')[0] if mimetype else ''
      # JSON inputs
      if mimetype == 'application/json':
        for o in get_json_objects(path, **kwargs):
          objects.append(standardize_metadata_row(o, **kwargs))
      # plaintext inputs
      elif mimetype == 'text/plain':
        o = get_plaintext_object(path, **kwargs)
        objects.append(o)
      elif mimetype in ['text/xml', 'text/html']:
        o = get_xml_object(path, **kwargs)
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
  # optionally add position data if the user provided it
  if kwargs.get('position_x') and kwargs.get('position_y'):
    filtered = []
    for i in objects:
      if i.get(kwargs['position_x']) and i.get(kwargs['position_y']):
        i['position'] = [
          float(i[kwargs['position_x']]),
          float(i[kwargs['position_y']]),
        ]
        filtered.append(i)
    objects = filtered
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
  bn = os.path.basename(path)
  meta = kwargs['metadata'].get(format_filename(bn), {})
  with codecs.open(path, 'r', kwargs['encoding']) as f:
    text = Plaintext(text=f.read(), label=bn)
    text.update(meta)
    return text

def get_xml_object(path, **kwargs):
  bn = os.path.basename(path)
  meta = kwargs['metadata'].get(format_filename(bn), {})
  with codecs.open(path, 'r', kwargs['encoding']) as f:
    soup = BeautifulSoup(f, 'html.parser')
    if kwargs.get('xml_base_tag'):
      soup = soup.find(kwargs['xml_base_tag'])
    text = ' '.join(soup.get_text().split())
    text = Plaintext(text=text, label=bn)
    text.update(meta)
    return text

def get_image_object(path, **kwargs):
  bn = os.path.basename(path)
  formatted_bn = format_filename(bn)
  meta = kwargs['metadata'].get(formatted_bn, {})
  im = Image(path=path, label=formatted_bn, image=formatted_bn)
  im.update(meta)
  # make sure images load
  try:
    im['__load_image__']
    return im
  except:
    return None

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

def get_vectorize_strategy(**kwargs):
  if kwargs.get('position_x') and kwargs.get('position_y'):
    return 'null'
  if kwargs.get('vectorize') != 'auto':
    return kwargs['vectorize']
  elif len([i for i in kwargs['objects'] if isinstance(i, Image)]) >= 0.5 * len(kwargs['objects']):
    return 'image'
  elif len([i.get(kwargs.get('vector', '')) for i in kwargs['objects']])  >= 0.5 * len(kwargs['objects']):
    return 'vector'
  return 'text'

def get_vectors(**kwargs):
  # if user provided both position_x and position_y, use positions as vectors
  if kwargs.get('position_x') and kwargs.get('position_y'):
    return [kwargs['objects'], [i['position'] for i in kwargs['objects']]]
  vecs = []
  objects = []
  if kwargs['vectorize'] == 'vector':
    with tqdm(total=len(kwargs['objects'])) as progress_bar:
      for i in kwargs['objects']:
        # object has vector
        if i.get(kwargs.get('vector', '')):
          vecs.append(i[kwargs['vector']])
          objects.append(i)
          progress_bar.update(1)
          continue
    return [objects, vecs]
  elif kwargs['vectorize'] == 'image':
    base = InceptionV3(include_top=True, weights='imagenet',)
    model = Model(inputs=base.input, outputs=base.get_layer('avg_pool').output)
    with tqdm(total=len(kwargs['objects'])) as progress_bar:
      for i in kwargs['objects']:
        vec = None
        try:
          # read vector off object directly
          vec = i.get(kwargs.get('vector', ''))
          # else check cache
          cache_path = os.path.join('cache', 'image-vectors', i['path'].replace('/', '-'))
          if vec is None and os.path.exists(cache_path + '.npy'):
            vec = np.load(cache_path + '.npy')
          # else create vector
          if vec is None:
            im = preprocess_input( img_to_array( i['__load_image__'].resize((299,299)) ) )
            vec = model.predict(np.expand_dims(im, 0)).squeeze()
            np.save(cache_path, vec)
          # add the vector and the object to the list of items to return
          vecs.append(vec)
          objects.append(i)
          progress_bar.update(1)
        except Exception as exc:
          print('WARNING: Image', i, 'could not be vectorized', exc)
    return [objects, vecs]
  elif kwargs['vectorize'] == 'text':
    text = []
    for i in kwargs['objects']:
      t = i.get('text', '')
      if isinstance(t, list):
        t = ' '.join(t)
      text.append(t.lower())
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    vecs = NMF(n_components=min(len(text), 100), max_iter=kwargs['max_iter'], verbose=1, init='nndsvd').fit_transform(X)
    return [kwargs['objects'], vecs]

##
# UMAP
##

def get_positions(**kwargs):
  # if the user provided point positions, use them
  if kwargs.get('position_x') and kwargs.get('position_y'):
    return scale([i['position'] for i in kwargs['objects']])
  # else get vector positions
  return scale(run_umap(n_components=2, **kwargs))

def run_umap(**kwargs):
  vecs = PCA(n_components=min(len(kwargs['vectors']), 50)).fit_transform(kwargs['vectors'])
  return UMAP(n_neighbors=kwargs['n_neighbors'],
              min_dist=kwargs['min_dist'],
              n_components=kwargs['n_components'],
              verbose=True).fit_transform(vecs)

##
# Colors
##

def get_colors(**kwargs):
  # if user provided custom positions, return null colors
  if kwargs.get('position_x') and kwargs.get('position_y'):
    return np.array([[255,255,255] for i in kwargs['objects']])
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
    try:
      color = ColorThief(obj['path']).get_color(quality=quality)
    except:
      print(' * Could not extract color for', obj['path'])
      color = [255,255,255]
    np.save(cache_path + '.npy', color)
  return [idx, color]

##
# Write Outputs
##

def write_outputs(**kwargs):
  # copy web assets
  src = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'web')
  dest = os.path.join(os.getcwd(), kwargs['output_folder'])
  print(' * copying web assets')
  copy_tree(src, dest)
  # copy media before mutating objects below
  copy_media(**kwargs)
  # merge metadata into object
  kwargs['objects'] = [ {**i, **i.get('metadata', {})} for i in kwargs['objects'] ]
  # write outputs
  positions_path = os.path.join(kwargs['output_folder'], 'data', 'positions.json')
  colors_path = os.path.join(kwargs['output_folder'], 'data', 'colors.json')
  objects_path = os.path.join(kwargs['output_folder'], 'data', 'objects.json')
  print(' * writing positions')
  write_json(positions_path, round_floats(kwargs['positions']), gzip=True)
  print(' * writing colors')
  write_json(colors_path, round_floats(kwargs['colors'].squeeze(), digits=2), gzip=True)
  print(' * writing objects')
  write_objects(**kwargs)

def scale(a):
  '''Scale a 2D numpy array while preserving proportions'''
  if not isinstance(a, np.ndarray): a = np.array(a)
  biggest = np.max(a)
  smallest = np.min(a)
  for i in range(2):
    #smallest = np.min(a[:, i])
    a[:, i] -= smallest
    a[:, i] /= (biggest-smallest)
  # scale -1 : 1
  a -= 0.5
  a *= 2
  return a

def round_floats(a, digits=4):
  '''Return 1D or 2D array a with rounded float precision'''
  if not isinstance(a, np.ndarray): a = np.array(a)
  if len(a.shape) == 2: return [[round(float(j), digits) for j in i] for i in a]
  elif len(a.shape) == 1: return [round(float(j), digits) for j in a]
  return a

def write_json(path, obj, gzip=False):
  if gzip:
    if not path.endswith('.gz'): path += '.gz'
    with GzipFile(path, 'w') as out:
      out.write(json.dumps(obj).encode('utf-8'))
  else:
    with open(path, 'w') as out:
      json.dump(obj, out)

def write_objects(**kwargs):
  objects_dir = os.path.join(kwargs['output_folder'], 'data', 'objects')
  Path(objects_dir).mkdir(parents=True, exist_ok=True)
  with tqdm(total=len(kwargs['objects'])) as progress_bar:
    for idx, i in enumerate(kwargs['objects']):
      path = os.path.join(objects_dir, '{}.json'.format(idx))
      write_json(path, i, gzip=False)
      progress_bar.update(1)

def copy_media(thumb_size=100, **kwargs):
  if kwargs['vectorize'] == 'image':
    print(' * copying media')
    thumbs_dir = os.path.join(kwargs.get('output_folder'), 'data', 'thumbs')
    originals_dir = os.path.join(kwargs.get('output_folder'), 'data', 'originals')
    Path(thumbs_dir).mkdir(parents=True, exist_ok=True)
    Path(originals_dir).mkdir(parents=True, exist_ok=True)
    with tqdm(total=len(kwargs['objects'])) as progress_bar:
      for i in kwargs['objects']:
        name = format_filename(os.path.basename(i['path']))
        # process originals
        shutil.copy(i['path'], os.path.join(originals_dir, name))
        # process thumbs
        im = i['__load_image__']
        w, h = im.size
        width = thumb_size
        height = int(h * (thumb_size / w))
        save_img(os.path.join(thumbs_dir, name), im.resize((width, height)))
        progress_bar.update(1)

if __name__ == '__main__':
  parse()