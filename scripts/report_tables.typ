// Adapt MyST's tablex calls to native Typst tables. Native headers are tied to
// actual page breaks, avoiding tablex's stale pagination state in long reports.
#let cellx = table.cell
#let hlinex = table.hline
#let vlinex = table.vline
#let space-elem = [ ].func()
#let is-blank(c) = {
  if c == none or c == [] or c == [ ] { return true }
  if type(c) == str { return c.trim() == "" }
  if type(c) != content { return false }
  let f = c.func()
  if f == space-elem or f == parbreak { return true }
  if f == table.cell { return is-blank(c.body) }
  if c.has("children") { return c.children.all(is-blank) }
  false
}
#let tablex(columns: 1, header-rows: 1, repeat-header: true, ..args) = {
  let cells = args.pos()
  let ncols = if type(columns) == int { columns } else { columns.len() }
  let nhead = ncols * header-rows
  // A body row whose only content is its first cell is a group label (for
  // example "Mean process" in the parameter-range tables): span the full width.
  let body-cells = cells.slice(nhead)
  if ncols > 1 and calc.rem(body-cells.len(), ncols) == 0 {
    let rows = ()
    for r in body-cells.chunks(ncols) {
      if not is-blank(r.at(0)) and r.slice(1).all(is-blank) {
        rows.push(table.cell(colspan: ncols, r.at(0)))
      } else {
        rows += r
      }
    }
    body-cells = rows
  }
  show table.cell.where(y: 0): set text(weight: "bold")
  let body = table(columns: columns,
    stroke: 0.4pt + rgb("#b8b8b8"),
    fill: (x, y) => if y < header-rows { rgb("#eeeeee") } else { none },
    inset: (x: 4pt, y: 3pt), ..args.named(),
    table.header(repeat: repeat-header,
      table.hline(stroke: 0.8pt + rgb("#666666")),
      ..cells.slice(0, nhead),
      table.hline(stroke: 0.8pt + rgb("#666666")),
    ),
    ..body-cells,
  )
  // Keep compact results tables together; long tables repeat their header
  // only at the top of a continuation page.
  if cells.len() / ncols <= 18 {
    block(breakable: false, body)
  } else {
    body
  }
}
