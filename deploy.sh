aws s3 sync . s3://lab-apps/sketches/trails/ --profile yale-admin --acl public-read --exclude ".git/*" --exclude "assets/data/thumbs/*"
