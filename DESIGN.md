# Design direction for the plasmon dashboard

Written by the agent at the owner's request and revised with the owner's feedback after
the first version: more energy, live diagrams, motion that shows work. The owner can
replace any line. Every choice carries its reason.

## Reading

Operations dashboard for two audiences: an admin who watches a fleet, and an employee who
lends a PC and wants one answer ("is my machine training, and what?"). Technical buyers,
but also non-technical staff. References the owner named for the feel: Weights & Biases
(dense metric cards and charts), the OpenRouter dashboard (cards with percentages), the
PlanetScale cluster view (boxes connected by paths that move when data flows) and the
Apache Flink job graph. They are references for structure and liveliness, not a skin:
plasmon keeps its own palette and motif.

Dial: ENERGY 2 / RHYTHM 2 / MOTION 2.

- ENERGY 2: the owner asked for a dashboard that feels alive. Cards with one large number
  each, a diagram as the first thing on the overview, colour used for state. Not 3: the
  pages are still about reading numbers.
- RHYTHM 2: each page opens with a different shape (a diagram, a chart, a status line, a
  table) and continues with cards and tables on one grid.
- MOTION 2: motion carries information and nothing else. The inventory is closed:
  1. Marching ants on a path: data is moving on that link right now (a machine training,
     an update in flight). Static when nothing moves.
  2. Pulse ring on a status dot: a live heartbeat was received in the last interval.
  3. Rise-in on first paint: cards and rows appear in order so the eye lands on the
     focal point first. Never replayed on refresh.
  4. Progress bars and numbers change in place with a 200 ms ease; a changed cell gets a
     short background flash.
  5. Skeleton shimmer while a diagram waits for its first data.
  Everything stops under `prefers-reduced-motion: reduce`, and dashes stay static.

## Identity

- Name and motif: plasmon, a collective wave. One thin sine wave drawn as inline SVG is the
  logo mark and the loading indicator. Sparklines and diagram paths use the same stroke
  width, so every line reads as part of one system.
- Typeface: the system UI stack. Reason: no font download, so the dashboard loads on an
  air-gapped network at full speed, and it matches the machine's own UI, which fits a tool
  that reports on that machine. Tabular figures on every number.
- Palette: two neutrals (paper and ink, light and dark values) plus one accent, teal
  `#0f766e` light / `#5eead4` dark. The accent marks the focal element, focus rings,
  progress and the coordinator in diagrams. Contrast: teal on paper 5.3:1, light teal on
  dark 12.8:1.
- Status scale (semantic, always paired with the word):
  training green, idle blue, paused yellow, unavailable grey, offline red, error magenta.
  Light and dark values chosen for 4.5:1 or more against their background. In diagrams
  the same six colours drive the node border, the port dot and the path.
- Shape: radius 4 px on inputs and buttons, 8 px on cards and diagram nodes. Nothing
  pill-shaped except status chips, which are pills because they are read as labels.
- Elevation: cards have a hairline border and one soft shadow (0 1px 2px). Reason: a page
  now holds many small cards and the shadow separates them from the page without heavier
  borders. One level only.
- Layout: a fixed left rail with the pages the role can open, and a content column with a
  maximum width of 1320 px. Metric cards sit on an auto-fill grid of 180 px columns. Below
  720 px the rail becomes a horizontal row of links and cards stack.
- Themes: light and dark, following the system and switchable with a button that
  remembers the choice. Both are checked for contrast.

## Diagrams

- Structure follows the PlanetScale cluster view at the owner's request: compact HTML
  nodes (12 px text, 96 px minimum width) with a header strip, a body with an icon and two
  lines, and a monospaced footer (`CPU: 41%`); 8 px port dots; group boxes with two stacked
  translucent borders behind them and a status pill that overlaps the bottom border; 1 px
  wires drawn as SVG paths measured from the DOM, with marching ants on the wires that
  carry data. Colours, icons and copy are plasmon's.
- Every node has one fixed size (156 × 66 px; the coordinator 236 px wide) whatever its
  status or content: the subtitle line and the footer slots are always present, with "–"
  for a value that does not exist yet. A status change never moves a neighbour.
- Wires follow a trunk-and-bus scheme so that no two colours share a segment: one trunk
  from the coordinator down to a horizontal bus, one drop per group, and for a group in a
  later row a corridor that hugs the boxes above or passes through a gap between them.
  Inside a group the same scheme repeats at a smaller scale. The trunk and the bus turn
  green and move while any machine trains; a drop takes the colour of what it reaches.
- A machine belongs to a job's group while it trains it: the trainer reports `training`
  between rounds too ("round 7 done") until the job ends. Machines of the last two rounds
  that stopped reporting stay in the group as idle "between rounds"; the group goes back
  to a waiting node only when nobody took part recently. A job's waiting sentence appears after 20 s without
  an update, never between the rounds of a job that progresses.
- Network diagram (overview, fleet): the coordinator on top (header "Coordinator", the org
  name with a count of machines online, one small square per machine coloured by status,
  footer with the scheduler state and rounds per hour); below it one group per running
  job holding the machines that train it (pill: job name ・ round r/N), a group of
  available machines and a group of offline or error machines. Icons: Apple, Windows,
  Linux, generic PC, GPU chip for CUDA machines, a server for the coordinator.
- Job diagram (job page): data → trainers of the current round (a vertical group) →
  aggregation → weights, left to right. Each trainer's wire shows its state in the round:
  assigned (dashed grey), committed (yellow, moving), revealed (green, moving), accepted
  (green), rejected or expired (red dashed).
- `diagram.js` draws from the JSON API and reconciles nodes by id every three seconds,
  so animations do not restart. It renders a skeleton first and an error line when the
  API is unreachable. Wires are redrawn on resize.

## Focal points

- Overview: the network diagram, then the machines-online card.
- Jobs: the loss curve of the opened job; the job diagram shows the round in progress.
- Open jobs: the "your machines" column, where the join decision is made. The table
  puts pay and requirements before progress because that is the order a trainer
  compares jobs in.
- Job page, Trainers section: the pending requests come first with the Approve button,
  because they are the one thing waiting on the owner. Approved and rejected machines
  follow as a quieter table.
- My machine: the status line ("training mnist-home for you, round 12"), then the
  standing on each job (joined, waiting, not accepted) with the one action each allows.
- Fleet: the diagram, then the status totals.
- Server: the scheduler state and the ledger head.

Enrolment and hold standings use the status scale, always with the word: pending and held
are yellow (waiting on someone), approved and released green, rejected and voided red,
left grey. No new colour was added for the marketplace.

A control that cannot act is not drawn: while a funded job runs, the weights are not
downloadable, so the page shows the sentence that says when they will be, not a disabled
button.

## Copy

Short, factual labels in sentence case. No marketing words. Buttons name the action
("Download latest weights", "Cancel job", "Confirm code", "Switch theme").

## Terminal

- The CLI uses the terminal's own 16 colours, never fixed RGB, so the views read on light
  and dark themes and follow the user's palette. Cyan is the brand accent (the wave, the
  active tab, titles); green, blue, yellow, grey, red and magenta are the status scale, the
  same as the web dashboard.
- Start-up sequence (about 0.9 s, any key skips): scattered dots lock into a travelling
  wave, the wordmark resolves out of noise left to right (░ ▒ ▓ then the glyph), then the
  tagline and the version. It plays on the bare `plasmon`, on `plasmon dashboard` and once,
  inline, before `plasmon trainer start` hands over to the trainer log. Off when stdout is
  not a TTY, with `--plain`, or under `NO_COLOR`, `PLASMON_NO_ANIM`, `TERM=dumb`.
- Dashboard: one header line (mark, tabs, user and server), a rule, the body, one footer
  line (key hints left, refresh cadence right, the last error in red). Cards use rounded
  borders with a muted title; sections use a top rule with a bold title. Status is always
  a coloured dot plus the word. Load is a five-cell bar coloured green, yellow or red by
  level, followed by the number. A selected row is reversed as one block.
- The same screens serve `job watch`, `fleet --watch`, `fleet show --watch` and
  `server status --watch`, locked to one view.
