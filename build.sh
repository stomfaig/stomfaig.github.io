#!/usr/bin/env bash
set -e

CONFIG=site.yml
INPUT=notes
OUTPUT=public

mkdir -p $OUTPUT

cp templates/style.css $OUTPUT/style.css

tmpdir=$(mktemp -d)

for file in $INPUT/*.md; do
  name=$(basename "$file" .md)
  pandoc "$file" \
    -f markdown+fenced_divs \
    --lua-filter=templates/sidenotes.lua \
    --template=templates/page.html \
    --mathjax \
    --css=style.css \
    --standalone \
    -o "$OUTPUT/$name.html"

  title=$(awk '/^---/{p=!p;next} p && /^title:/{sub(/^title:[[:space:]]*/,""); print; exit}' "$file")
  description=$(awk '/^---/{p=!p;next} p && /^description:/{sub(/^description:[[:space:]]*/,""); print; exit}' "$file")
  date=$(awk '/^---/{p=!p;next} p && /^date:/{sub(/^date:[[:space:]]*/,""); print; exit}' "$file")

  if [ -n "$description" ]; then
    cat > "$tmpdir/${date}_${name}" <<EOF
      <li>
        <a href="${name}.html">${title}</a>
        <p class="article-meta">${date}</p>
        <p class="article-blurb">${description}</p>
      </li>
EOF
  fi
done

articles_html=""
for entry in $(ls "$tmpdir" | sort -r); do
  articles_html="${articles_html}$(cat "$tmpdir/$entry")"$'\n'
done
rm -rf "$tmpdir"

python3 -c "
import sys
template = open('templates/index.html').read()
print(template.replace('ARTICLES_PLACEHOLDER', sys.stdin.read().rstrip()), end='')
" <<< "$articles_html" > "$OUTPUT/index.html"

