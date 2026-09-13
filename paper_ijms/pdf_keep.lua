-- Keep a table caption with its table in the reading PDF: before each caption
-- paragraph ("**Table N.** ...") followed by a table, ask for enough vertical
-- room for the caption plus the table's rows (capped), so short tables are
-- not split across pages.
local function rows_of(tbl)
  local n = 0
  for _, body in ipairs(tbl.bodies) do n = n + #body.body end
  return n
end

function Blocks(blocks)
  local out = pandoc.List()
  local i = 1
  while i <= #blocks do
    local b = blocks[i]
    local nxt = blocks[i + 1]
    if b.t == 'Para' and nxt and nxt.t == 'Table' and b.content[1]
       and b.content[1].t == 'Strong'
       and pandoc.utils.stringify(b.content[1]):match('^Table') then
      local lines = math.min(rows_of(nxt) * 2 + 8, 40)
      out:insert(pandoc.RawBlock('latex',
        string.format('\\needspace{%d\\baselineskip}', lines)))
    end
    out:insert(b)
    i = i + 1
  end
  return out
end
