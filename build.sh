#!/usr/bin/env bash
set -e

CONFIG=site.yml
INPUT=notes
OUTPUT=public

mkdir -p $OUTPUT

cp templates/style.css $OUTPUT/style.css

for file in $INPUT/*.md; do
  name=$(basename "$file" .md)
  pandoc "$file" \
    -f markdown+fenced_divs \
    --template=templates/page.html \
    --mathjax \
    --css=style.css \
    --standalone \
    -o "$OUTPUT/$name.html"
done

