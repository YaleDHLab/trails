# Trails

> Trails builds interactive plots of massive datasets. You can use it to visualize large text, image, audio, video or other collections!

![App preview](/trails/web/assets/images/preview.png?raw=true)

# Basic Usage

```
# install the app
pip install git+https://github.com/YaleDHLab/trails.git

# download a sample JSON dataset
wget https://lab-data-collections.s3.amazonaws.com/wiki-people.json

# process the wiki-people.json file using the "abstract" field for vectorization
trails --input "wiki-people.json" --label "name" --text "abstract"
```

# Examples

[Wikipedia People](https://lab-apps.s3-us-west-2.amazonaws.com/sketches/trails/index.html)<br/>
[Samuel Johnson's Dictionary](https://lab-apps.s3-us-west-2.amazonaws.com/sketches/trails/johnson/index.html)<br/>
[Oslo Photographic Collection](https://lab-apps.s3-us-west-2.amazonaws.com/trails/oslo/index.html)<br/>
[Harvard Art Museum Collection](https://lab-apps.s3-us-west-2.amazonaws.com/trails/image-data/index.html)<br/>

# Data Processing

Trails can process text files, image files, JSON files, or other filetypes that have already been vectorized.

**Text Inputs**

To process a text collection with Trails, provide the paths to your text files:

```bash
trails --inputs "texts/*.txt"
```

**Image Inputs**

To process an image collection with Trails, provide the paths to your image files:

```bash
trails --inputs "images/*.jpg"
```

**JSON Inputs**

To process a collection of JSON files with Trails, provide the path to your JSON file(s), then indicate the fields that should be used for each item's `label` and `text` fields:

```bash
trails --input "wiki-people.json" --label "name" --text "abstract"
```

**Custom Vectors**

If each object in your collection already has been vectorized, format your inputs as JSON, include the vectors in those JSON files, and specify the field that contains the vector when evoking Trails:

```bash
trails --input "birdsong.json" --vector "vec"
```

**Custom Positions**

If each object in your collection already has a 2D position, just add an x column and a y column to your metadata and specify those columns when evoking Trails:

```bash
trails --input "birdsong.json" -x "longitude" -y "latitude"
```

**Adding Metadata**

If you have metadata associated with your objects (e.g. you have a collection of text files _and_ a JSON file with associated metadata), make sure your metadata has `filename` as its first column (in case of CSV metadata) or has `filename` as an attribute (in case of JSON metadata). Then you can provide your metadata to the data pipeline as follows:

```bash
trails --inputs "images/*.jpg" --metadata "image_metadata.json"
```

**Iterating Quickly**

To iterate quickly, use the `--limit` flag to only process a small subset of your collection:

```bash
trails --inputs "images/*.jpg" --limit 100
```

**Multiple Plots**

If you want to create multiple plots in the same directory, use the `--output_folder` flag to specify the directory in which the current outputs will be written

```bash
trails --inputs "images/*.jpg" --output_folder "catplot"
```

# How It Works

Trails uses three pieces of data to create interactive displays:

1) `Objects`: Objects are the individual items in your dataset (e.g. a text file, or an image). Each point in the scatterplot corresponds to one object. When a user hovers or clicks on a point, we display the corresponding object. After Trails runs, each object is represented by a single JSON file in `./output/data/objects/`. Those files are named `0.json` through `n-1.json`, where `n` is the number of objects to be displayed. When it's time to display an object, we populate `./output/preview.html` with the data from the object's JSON file. If a user clicks the object preview, we populate `./output/tooltip.html` with the data from the object's JSON file.

2) `Positions`: The position of each object is contained in `./output/data/positions.json.gz`. The ith object's position is contained at index position i in this file.

3) `Colors`: The color of each point in the scatterplot is contained in `./output/data/colors.json.gz`. The ith object's color is contained at index position i in this file.

# Customizing the UI

To customize the Trails UI, there are three files you may want to modify:

1) `./output/custom.css`: Your custom CSS can go in this file, and these styles will overwrite the default styles.

2) `./output/preview.html`: To change the way objects look when being previewed, change the HTML template in this file. This HTML is a [Lodash template](https://lodash.com/docs/4.17.15#template).

3) `./output/preview.html`: To change the way an object looks when a user clicks on the corresponding preview, change the HTML template in this file. This HTML is a [Lodash template](https://lodash.com/docs/4.17.15#template).