# Trails

> Visualizing massive datasets with WebGL.

Trails aims to provide a flexible high-level tool capable of displaying massive visualizations using WebGL. 

![App preview](/trails/web/assets/images/preview.png?raw=true)

# Usage

```
# install the app
pip install https://github.com/yaledhlab/trails/archive/master.zip

# download sample inputs
wget https://lab-data-collections.s3.amazonaws.com/wiki-people.json

# run the app
trails --inputs datasets/wiki-people.json --text "abstract" --sort views --limit 100000
```

# Customizing UI

TODO: discuss PREVIEW_TEMPLATE & TOOLTIP_TEMPLATE in `output/index.html`