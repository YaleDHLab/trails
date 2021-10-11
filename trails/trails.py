import mimetypes
import argparse
import glob2
import json
import os

config = {
  'inputs': None,
  'title': None,
  'text': None,
  'limit': None,
  'sort': None,
}

def parse():
  '''Read command line args and begin data processing'''
  description = 'Create the data required to create a Trails viewer'
  parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--inputs', '-i', type=str, default=config['inputs'], help='path to a glob of files to process', required=True)
  parser.add_argument('--title', '-t', type=str, default=config['title'], help='field in metadata that contains object title', required=True)
  parser.add_argument('--text', '-c', type=str, default=config['text'], help='field in metadata that contains body text (for text objects)', required=True)
  parser.add_argument('--limit', '-l', type=int, default=config['limit'], help='the maximum number of observations to analyze', required=False)
  parser.add_argument('--sort', '-s', type=str, default=config['sort'], help='the field to use when sorting objects', required=False)
  config.update(vars(parser.parse_args()))
  process(**config)

def process(**kwargs):
  objects = list(get_objects(**kwargs))
  if kwargs.get('sort'): objects = sorted(objects, key=lambda i: i[kwargs['sort']])


def get_objects(**kwargs):
  '''
  Format inputs into a dictionary stream, each with the following attributes:

  title: title attribute || filename
  body: text content or image array

  Supported filetypes:
    * application/json
    x image/jpeg
    x image/gif
    x image/png
    x text/csv
    x text/plain
  '''
  for path in glob2.glob(kwargs['inputs']):
    mimetype = get_mimetype(path)
    # handle the case of multi-object mimetypes (csv, json)
    if mimetype == 'application/json':
      return get_json_objects(path, **kwargs)
    else:
      print('WARNING: Only JSON objects are currently supported')

def get_json_objects(path, **kwargs):
  with open(path) as f:
    j = json.load(f)
    # handle the case that j is an array of objects
    if isinstance(j, list):
      for i in j:
        yield get_json_object(i, **kwargs)
    # handle the case that j is a single object
    elif isinstance(j, dict):
      yield get_json_object(j, **kwargs)

def get_json_object(d, **kwargs):
  used_keys = [kwargs[i] for i in ['title', 'text']]
  unused_keys = [i for i in d.keys() if i not in used_keys]
  return {
    'title': d[kwargs['title']],
    'text':  d[kwargs['text']],
    'meta': {i: d[i] for i in unused_keys},
  }

def get_mimetype(path, full=True):
  '''Given a filepath, return the mimetype'''
  mime = mimetypes.MimeTypes().guess_type(path)[0]
  if mime and not full: mime = mime.split('/')[0]
  return mime

if __name__ == '__main__':
  parse()