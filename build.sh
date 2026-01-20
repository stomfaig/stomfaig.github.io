#!/usr/bin/env bash
set -e

CONFIG=site.yml
INPUT=notes
OUTPUT=public

mkdir -p $OUTPUT

for file in $INPUT/*.md; do
  name=$(basename "$file" .md)
  pandoc "$file" \
    --template=templates/page.html \
    --mathjax \
    --highlight-style=pygments \
    -o "$OUTPUT/$name.html"
done

cp -r static "$OUTPUT/"

