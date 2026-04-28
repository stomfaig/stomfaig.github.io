local counter = 0

local function convert_math(el)
  if el.mathtype == 'InlineMath' then
    return pandoc.RawInline('html', '\\(' .. el.text .. '\\)')
  else
    return pandoc.RawInline('html', '\\[' .. el.text .. '\\]')
  end
end

function Note(note)
  counter = counter + 1
  local n = tostring(counter)
  local blocks = {}
  for i, block in ipairs(note.content) do
    blocks[i] = block:walk({ Math = convert_math })
  end
  local html = pandoc.write(pandoc.Pandoc(blocks), 'html'):gsub('</?p[^>]*>', '')
  return pandoc.RawInline('html',
    '<sup class="sidenote-ref">' .. n .. '</sup>' ..
    '<span class="sidenote"><sup>' .. n .. '</sup>&thinsp;' .. html .. '</span>'
  )
end
