# Build documentation website
pip install mkdocs-material
mkdocs build

# Deploy to gh-pages
git checkout -B gh-pages
git add -f site/
git commit -m "Deploy documentation"
git push origin gh-pages --force