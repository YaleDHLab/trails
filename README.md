# Trails

> Trails builds interactive plots of massive datasets.

![App preview](/trails/web/assets/images/preview.png?raw=true)

# Basic Usage

```
# install the app
pip install git+https://github.com/YaleDHLab/trails.git

# download a sample dataset of wikipedia people
wget https://lab-data-collections.s3.amazonaws.com/wiki-people.json

# process the wiki-people.json file using the "abstract" field for vectorization
trails --input "wiki-people.json" --text "abstract" --label "name"
```

# Customizing UI

TODO