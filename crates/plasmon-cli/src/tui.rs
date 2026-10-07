//! Terminal views: the start-up sequence, the tabbed dashboard, the `--watch` screens and
//! the login spinner. One palette: the terminal's own 16 colours, so every view reads on a
//! light or a dark terminal. Cyan is the brand accent (the wave); green, blue, yellow, grey,
//! red and magenta are the status scale from DESIGN.md.

use crate::api::Api;
use crate::commands::{ago, num, s};
use anyhow::Result;
use crossterm::cursor::{Hide, MoveUp, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;
use ratatui::symbols::Marker;
use ratatui::widgets::{
    Axis, Block, BorderType, Borders, Cell, Chart, Clear, Dataset, Gauge, GraphType, Paragraph,
    Row, Sparkline, Table, TableState, Wrap,
};
use serde_json::Value;
use std::io::{stderr, stdout, IsTerminal, Write};
use std::time::{Duration, Instant};

pub const LOGO: [&str; 6] = [
    "██████╗ ██╗      █████╗ ███████╗███╗   ███╗ ██████╗ ███╗   ██╗",
    "██╔══██╗██║     ██╔══██╗██╔════╝████╗ ████║██╔═══██╗████╗  ██║",
    "██████╔╝██║     ███████║███████╗██╔████╔██║██║   ██║██╔██╗ ██║",
    "██╔═══╝ ██║     ██╔══██║╚════██║██║╚██╔╝██║██║   ██║██║╚██╗██║",
    "██║     ███████╗██║  ██║███████║██║ ╚═╝ ██║╚██████╔╝██║ ╚████║",
    "╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
];
pub const MARK: &str = "∿";
pub const TAGLINE: &str = "thousands of GPUs, one wave";
const VERSION: &str = env!("CARGO_PKG_VERSION");

pub const ACCENT: Color = Color::Cyan;
pub const MUTED: Color = Color::DarkGray;

pub fn animation_enabled() -> bool {
    stdout().is_terminal()
        && std::env::var_os("NO_COLOR").is_none()
        && std::env::var_os("PLASMON_NO_ANIM").is_none()
        && std::env::var("TERM").map(|t| t != "dumb").unwrap_or(true)
}

pub fn status_color(status: &str) -> Color {
    match status {
        "training" | "running" | "completed" | "accepted" | "revealed" | "closed" => Color::Green,
        "idle" | "open" => Color::Blue,
        "paused" | "committed" | "pending" => Color::Yellow,
        "unavailable" | "assigned" | "cancelled" => MUTED,
        "offline" | "failed" | "rejected" | "expired" | "out-of-credits" => Color::Red,
        "error" => Color::Magenta,
        _ => Color::Reset,
    }
}

fn bold() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}
fn muted() -> Style {
    Style::default().fg(MUTED)
}
fn accent() -> Style {
    Style::default().fg(ACCENT)
}
fn chip(status: &str) -> Span<'static> {
    Span::styled(
        format!("● {status}"),
        Style::default().fg(status_color(status)),
    )
}
/// `▰▰▰▱▱` style bar for a 0..100 value, coloured by load.
fn bar(pct: Option<f64>, width: usize) -> Span<'static> {
    match pct {
        None => Span::styled("·".repeat(width), muted()),
        Some(p) => {
            let p = p.clamp(0.0, 100.0);
            let filled = ((p / 100.0) * width as f64).round() as usize;
            let color = if p >= 90.0 {
                Color::Red
            } else if p >= 60.0 {
                Color::Yellow
            } else {
                Color::Green
            };
            Span::styled(
                format!(
                    "{}{}",
                    "▰".repeat(filled),
                    "▱".repeat(width.saturating_sub(filled))
                ),
                Style::default().fg(color),
            )
        }
    }
}
/// Progress, as distinct from load: accent while it runs, green once complete.
fn progress(frac: f64, width: usize) -> Span<'static> {
    let f = if frac.is_finite() {
        frac.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let filled = (f * width as f64).round() as usize;
    Span::styled(
        format!("{}{}", "▰".repeat(filled), "▱".repeat(width - filled)),
        Style::default().fg(if f >= 1.0 { Color::Green } else { ACCENT }),
    )
}
/// A text sparkline: eight block heights, scaled to the series' own range.
fn spark(values: &[f64], width: usize) -> String {
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() {
        return String::new();
    }
    let tail: Vec<f64> = values.iter().rev().take(width).rev().copied().collect();
    let (lo, hi) = tail
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    tail.iter()
        .map(|v| {
            let t = if hi > lo { (v - lo) / (hi - lo) } else { 0.5 };
            BLOCKS[((t * 7.0).round() as usize).min(7)]
        })
        .collect()
}
fn f3(v: &Value) -> String {
    v.as_f64()
        .map(|x| format!("{x:.3}"))
        .unwrap_or_else(|| "–".into())
}
fn pct_str(v: &Value) -> String {
    v.as_f64()
        .map(|x| format!("{:.1} %", x * 100.0))
        .unwrap_or_else(|| "–".into())
}
fn arr(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}
fn hash(x: usize, y: usize) -> f64 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h % 10_000) as f64 / 10_000.0
}

struct Term {
    terminal: Terminal<CrosstermBackend<std::io::Stdout>>,
}

impl Term {
    fn enter() -> Result<Self> {
        enable_raw_mode()?;
        stdout().execute(EnterAlternateScreen)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
        Ok(Self { terminal })
    }
}

impl Drop for Term {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = stdout().execute(LeaveAlternateScreen);
        let _ = stdout().execute(Show);
    }
}

// ----- the start-up sequence ---------------------------------------------------------------

/// One frame of the wordmark resolving out of noise, boxd-style: a left-to-right sweep with
/// per-cell jitter, each cell passing through ░ ▒ ▓ before it settles. `t` in 0..1.
fn logo_frame(t: f64) -> Vec<Line<'static>> {
    let width = LOGO[0].chars().count();
    LOGO.iter()
        .enumerate()
        .map(|(y, row)| {
            let mut spans: Vec<Span> = Vec::new();
            for (x, ch) in row.chars().enumerate() {
                if ch == ' ' {
                    spans.push(Span::raw(" "));
                    continue;
                }
                let start = 0.55 * x as f64 / width as f64 + 0.3 * hash(x, y);
                let p = ((t - start) / 0.15).clamp(0.0, 1.0);
                let (glyph, style) = if p <= 0.0 {
                    (' ', Style::default())
                } else if p < 0.35 {
                    ('░', accent())
                } else if p < 0.7 {
                    ('▒', accent())
                } else if p < 1.0 {
                    ('▓', accent())
                } else {
                    (ch, bold())
                };
                spans.push(Span::styled(glyph.to_string(), style));
            }
            Line::from(spans)
        })
        .collect()
}

/// The wave: scattered dots lock into a sine as `t` grows; the crest is bright, the trough dim.
fn wave_lines(width: usize, rows: usize, t: f64, phase: f64) -> Vec<Line<'static>> {
    let mut grid = vec![vec![(' ', MUTED); width]; rows];
    #[allow(clippy::needless_range_loop)]
    for x in 0..width {
        let wave = (x as f64 / 7.0 - phase).sin();
        let noise = (hash(x, 99) - 0.5) * (1.0 - t.min(1.0)) * rows as f64;
        let y_f = (0.5 - wave * 0.5) * (rows - 1) as f64 + noise;
        let y = y_f.round().clamp(0.0, (rows - 1) as f64) as usize;
        let color = if wave > 0.6 {
            Color::LightCyan
        } else if wave > -0.3 {
            ACCENT
        } else {
            MUTED
        };
        grid[y][x] = (if t > 0.5 { '●' } else { '·' }, color);
    }
    grid.into_iter()
        .map(|row| {
            Line::from(
                row.into_iter()
                    .map(|(c, color)| Span::styled(c.to_string(), Style::default().fg(color)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

/// About 0.9 s: the wave locks in while the wordmark resolves. Any key skips.
fn intro(term: &mut Term, server: &str) -> Result<()> {
    let start = Instant::now();
    let total = Duration::from_millis(900);
    let logo_w = LOGO[0].chars().count() as u16;
    while start.elapsed() < total {
        let t = start.elapsed().as_secs_f64() / total.as_secs_f64();
        term.terminal.draw(|f| {
            let area = f.area();
            let compact = area.width < logo_w + 4 || area.height < 16;
            let block_h: u16 = if compact { 6 } else { 14 };
            let top = area.height.saturating_sub(block_h) / 2;
            let wave_area = Rect::new(area.x, area.y + top, area.width, 4);
            f.render_widget(
                Paragraph::new(wave_lines(area.width as usize, 4, t * 2.0, t * 5.0)),
                wave_area,
            );
            let mut lines: Vec<Line> = if compact {
                let shown = ((t - 0.3).max(0.0) / 0.5 * 8.0) as usize;
                vec![Line::from(vec![
                    Span::styled(format!("{MARK} "), accent()),
                    Span::styled("plasmon".chars().take(shown).collect::<String>(), bold()),
                ])]
            } else {
                logo_frame(t)
            };
            lines.push(Line::from(""));
            if t > 0.78 {
                lines.push(Line::from(Span::styled(TAGLINE, muted())));
                lines.push(Line::from(Span::styled(
                    format!("v{VERSION} · {server}"),
                    muted(),
                )));
            }
            let text_area = Rect::new(
                area.x,
                area.y + top + 5,
                area.width,
                area.height.saturating_sub(top + 5),
            );
            f.render_widget(
                Paragraph::new(lines).alignment(Alignment::Center),
                text_area,
            );
        })?;
        if event::poll(Duration::from_millis(33))? {
            if let Event::Key(_) = event::read()? {
                break;
            }
        }
    }
    Ok(())
}

/// The same reveal in the scrollback, for commands that hand the terminal to another
/// process afterwards (`trainer start`). Leaves the wordmark and one status line behind.
pub fn intro_inline(what: &str) -> Result<()> {
    let mut out = stdout();
    let width = crossterm::terminal::size().map(|(w, _)| w).unwrap_or(80) as usize;
    let logo_w = LOGO[0].chars().count();
    if !animation_enabled() || width < logo_w + 2 {
        println!("{MARK} plasmon v{VERSION} · {what}");
        return Ok(());
    }
    out.execute(Hide)?;
    let frames = 16;
    let mut first = true;
    for i in 0..=frames {
        let t = i as f64 / frames as f64;
        if !first {
            out.execute(MoveUp(LOGO.len() as u16))?;
        }
        first = false;
        for line in logo_frame(t) {
            let mut text = String::new();
            for span in line.spans {
                let c = span.content.as_ref();
                if span.style.fg == Some(ACCENT) {
                    text.push_str(&format!("\x1b[36m{c}\x1b[0m"));
                } else if span.style.add_modifier.contains(Modifier::BOLD) {
                    text.push_str(&format!("\x1b[1m{c}\x1b[0m"));
                } else {
                    text.push_str(c);
                }
            }
            writeln!(out, "\x1b[2K{text}")?;
        }
        out.flush()?;
        std::thread::sleep(Duration::from_millis(40));
    }
    writeln!(out, "\x1b[2m{TAGLINE} · v{VERSION} · {what}\x1b[0m\n")?;
    out.execute(Show)?;
    Ok(())
}

/// A spinner on stderr while waiting (login). Falls back to dots when stderr is not a TTY.
pub struct Spinner {
    label: String,
    i: usize,
    tty: bool,
}

impl Spinner {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            i: 0,
            tty: stderr().is_terminal() && animation_enabled(),
        }
    }
    pub fn tick(&mut self) {
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        if self.tty {
            eprint!(
                "\r\x1b[2K\x1b[36m{}\x1b[0m {}",
                FRAMES[self.i % FRAMES.len()],
                self.label
            );
        } else {
            eprint!(".");
        }
        let _ = stderr().flush();
        self.i += 1;
    }
    pub fn done(&mut self, line: &str) {
        if self.tty {
            eprintln!("\r\x1b[2K\x1b[32m✓\x1b[0m {line}");
        } else {
            eprintln!("\n{line}");
        }
    }
}

/// An OSC 8 hyperlink when the terminal is interactive, the bare URL otherwise.
pub fn link(url: &str) -> String {
    if animation_enabled() {
        format!("\x1b]8;;{url}\x1b\\\x1b[4;36m{url}\x1b[0m\x1b]8;;\x1b\\")
    } else {
        url.to_string()
    }
}

// ----- data ---------------------------------------------------------------------------------

#[derive(Default)]
struct Data {
    me: Value,
    jobs: Value,
    open: Value,
    fleet: Value,
    summary: Value,
    server: Value,
    job: Value,
    updates: Value,
    machine: Value,
    logs: Value,
    /// Closed-round eval losses per job, for the overview sparklines. Completed jobs are
    /// fetched once; running jobs on every refresh.
    losses: std::collections::HashMap<String, Vec<f64>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Overview,
    Jobs,
    Fleet,
    Machines,
    Server,
}

impl Tab {
    const ALL: [Tab; 5] = [
        Tab::Overview,
        Tab::Jobs,
        Tab::Fleet,
        Tab::Machines,
        Tab::Server,
    ];
    fn label(self) -> &'static str {
        match self {
            Tab::Overview => "overview",
            Tab::Jobs => "jobs",
            Tab::Fleet => "fleet",
            Tab::Machines => "my machines",
            Tab::Server => "server",
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum View {
    Tab(Tab),
    Job(String),
    Machine(String),
}

pub struct Dash<'a> {
    api: &'a Api,
    interval: Duration,
    view: View,
    /// A `--watch` command locks the view: tabs and Enter do nothing, q leaves.
    locked: bool,
    data: Data,
    jobs_state: TableState,
    fleet_state: TableState,
    mine_state: TableState,
    last_fetch: Option<Instant>,
    error: Option<String>,
    help: bool,
}

impl<'a> Dash<'a> {
    pub fn new(api: &'a Api, interval: Duration, view: View, locked: bool) -> Self {
        Self {
            api,
            interval,
            view,
            locked,
            data: Data::default(),
            jobs_state: TableState::default().with_selected(0),
            fleet_state: TableState::default().with_selected(0),
            mine_state: TableState::default().with_selected(0),
            last_fetch: None,
            error: None,
            help: false,
        }
    }

    fn fetch(&mut self) {
        let api = self.api;
        let mut err: Option<String> = None;
        self.data.me = api.get("/v1/auth/me").unwrap_or(Value::Null);
        match api
            .get("/v1/jobs?all=true")
            .or_else(|_| api.get("/v1/jobs"))
        {
            Ok(v) => self.data.jobs = v,
            Err(e) => err = Some(e.to_string()),
        }
        match api.get("/v1/fleet") {
            Ok(v) => self.data.fleet = v,
            Err(e) => err = err.or(Some(e.to_string())),
        }
        if let Ok(v) = api.get("/v1/fleet/summary") {
            self.data.summary = v;
        }
        if matches!(self.view, View::Tab(Tab::Overview)) {
            let ids: Vec<(String, bool)> = arr(&self.data.jobs)
                .iter()
                .map(|j| (s(&j["id"]), j["status"] == "running"))
                .collect();
            for (id, running) in ids.into_iter().take(12) {
                if !running && self.data.losses.contains_key(&id) {
                    continue;
                }
                if let Ok(j) = api.get(&format!("/v1/jobs/{id}")) {
                    let l: Vec<f64> = arr(&j["rounds"])
                        .iter()
                        .filter(|r| r["status"] == "closed")
                        .filter_map(|r| r["eval_loss"].as_f64())
                        .collect();
                    self.data.losses.insert(id, l);
                }
            }
        }
        if matches!(self.view, View::Tab(Tab::Server) | View::Tab(Tab::Overview)) {
            self.data.server = api.get("/v1/server/status").unwrap_or(Value::Null);
        }
        if matches!(self.view, View::Tab(Tab::Jobs)) {
            self.data.open = api.get("/v1/jobs/open").unwrap_or(Value::Null);
        }
        match &self.view {
            View::Job(id) => {
                match api.get(&format!("/v1/jobs/{id}")) {
                    Ok(v) => self.data.job = v,
                    Err(e) => err = Some(e.to_string()),
                }
                if let Ok(v) = api.get(&format!("/v1/jobs/{id}/updates")) {
                    self.data.updates = v;
                }
            }
            View::Machine(node) => {
                match api.get(&format!("/v1/fleet/{node}")) {
                    Ok(v) => self.data.machine = v,
                    Err(e) => err = Some(e.to_string()),
                }
                if let Ok(v) = api.get(&format!("/v1/fleet/{node}/logs?limit=40")) {
                    self.data.logs = v;
                }
            }
            View::Tab(_) => {}
        }
        self.error = err;
        self.last_fetch = Some(Instant::now());
    }

    fn my_email(&self) -> String {
        s(&self.data.me["user"]["email"])
    }
    fn mine(&self) -> Vec<Value> {
        let me = self.my_email();
        arr(&self.data.fleet)
            .iter()
            .filter(|m| m["owner"].as_str() == Some(me.as_str()))
            .cloned()
            .collect()
    }

    /// Runs until q. Returns Ok when the user leaves.
    pub fn run(&mut self, animate: bool) -> Result<()> {
        let mut term = Term::enter()?;
        if animate {
            intro(&mut term, self.api.base())?;
        }
        loop {
            if self
                .last_fetch
                .map(|t| t.elapsed() >= self.interval)
                .unwrap_or(true)
            {
                self.fetch();
            }
            term.terminal.draw(|f| self.draw(f))?;
            if event::poll(Duration::from_millis(120))? {
                if let Event::Key(k) = event::read()? {
                    if k.kind != KeyEventKind::Press {
                        continue;
                    }
                    if self.handle_key(k.code, k.modifiers) {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// True when the user wants to leave.
    fn handle_key(&mut self, code: KeyCode, mods: KeyModifiers) -> bool {
        if self.help {
            self.help = false;
            return false;
        }
        match code {
            KeyCode::Char('q') => return true,
            KeyCode::Char('c') if mods.contains(KeyModifiers::CONTROL) => return true,
            KeyCode::Char('?') => self.help = true,
            KeyCode::Char('r') => self.last_fetch = None,
            KeyCode::Esc | KeyCode::Backspace => {
                if self.locked {
                    return true;
                }
                match self.view {
                    View::Job(_) => self.view = View::Tab(Tab::Jobs),
                    View::Machine(_) => self.view = View::Tab(Tab::Fleet),
                    View::Tab(_) => return true,
                }
            }
            KeyCode::Char(c) if !self.locked => {
                let tab = match c {
                    '1' | 'o' => Some(Tab::Overview),
                    '2' | 'j' => Some(Tab::Jobs),
                    '3' | 'f' => Some(Tab::Fleet),
                    '4' | 'm' => Some(Tab::Machines),
                    '5' | 's' => Some(Tab::Server),
                    _ => None,
                };
                if let Some(t) = tab {
                    self.view = View::Tab(t);
                    self.last_fetch = None;
                }
            }
            KeyCode::Tab | KeyCode::BackTab if !self.locked => {
                if let View::Tab(t) = self.view {
                    let i = Tab::ALL.iter().position(|x| *x == t).unwrap_or(0);
                    let n = Tab::ALL.len();
                    let next = if code == KeyCode::Tab {
                        (i + 1) % n
                    } else {
                        (i + n - 1) % n
                    };
                    self.view = View::Tab(Tab::ALL[next]);
                    self.last_fetch = None;
                }
            }
            KeyCode::Down | KeyCode::Up | KeyCode::PageDown | KeyCode::PageUp => {
                let step: i64 = match code {
                    KeyCode::Down => 1,
                    KeyCode::Up => -1,
                    KeyCode::PageDown => 10,
                    _ => -10,
                };
                let (state, len) = match self.view {
                    View::Tab(Tab::Jobs) => (&mut self.jobs_state, arr(&self.data.jobs).len()),
                    View::Tab(Tab::Fleet) => (&mut self.fleet_state, arr(&self.data.fleet).len()),
                    View::Tab(Tab::Machines) => {
                        let n = self.mine().len();
                        (&mut self.mine_state, n)
                    }
                    _ => return false,
                };
                if len > 0 {
                    let cur = state.selected().unwrap_or(0) as i64;
                    state.select(Some((cur + step).clamp(0, len as i64 - 1) as usize));
                }
            }
            KeyCode::Enter if !self.locked => {
                let next = match self.view {
                    View::Tab(Tab::Jobs) => self
                        .jobs_state
                        .selected()
                        .and_then(|i| arr(&self.data.jobs).get(i))
                        .map(|j| View::Job(s(&j["id"]))),
                    View::Tab(Tab::Fleet) => self
                        .fleet_state
                        .selected()
                        .and_then(|i| arr(&self.data.fleet).get(i))
                        .map(|m| View::Machine(s(&m["node_id"]))),
                    View::Tab(Tab::Machines) => self
                        .mine_state
                        .selected()
                        .and_then(|i| self.mine().get(i).cloned())
                        .map(|m| View::Machine(s(&m["node_id"]))),
                    _ => None,
                };
                if let Some(v) = next {
                    self.view = v;
                    self.last_fetch = None;
                }
            }
            _ => {}
        }
        false
    }

    // ----- drawing ------------------------------------------------------------------------

    pub fn draw(&mut self, f: &mut Frame) {
        let chunks = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(5),
            Constraint::Length(1),
        ])
        .split(f.area());
        self.draw_header(f, chunks[0]);
        f.render_widget(
            Block::default().borders(Borders::TOP).border_style(muted()),
            chunks[1],
        );
        match self.view.clone() {
            View::Tab(Tab::Overview) => self.draw_overview(f, chunks[2]),
            View::Tab(Tab::Jobs) => self.draw_jobs(f, chunks[2]),
            View::Tab(Tab::Fleet) => self.draw_fleet(f, chunks[2], false),
            View::Tab(Tab::Machines) => self.draw_fleet(f, chunks[2], true),
            View::Tab(Tab::Server) => self.draw_server(f, chunks[2]),
            View::Job(id) => self.draw_job(f, chunks[2], &id),
            View::Machine(node) => self.draw_machine(f, chunks[2], &node),
        }
        self.draw_footer(f, chunks[3]);
        if self.help {
            self.draw_help(f);
        }
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let mut spans = vec![
            Span::styled(format!(" {MARK} "), accent().add_modifier(Modifier::BOLD)),
            Span::styled("plasmon", bold()),
            Span::raw("   "),
        ];
        if !self.locked {
            for (i, t) in Tab::ALL.iter().enumerate() {
                let active = matches!(&self.view, View::Tab(x) if x == t)
                    || matches!(
                        (&self.view, t),
                        (View::Job(_), Tab::Jobs) | (View::Machine(_), Tab::Fleet)
                    );
                let style = if active {
                    accent().add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
                } else {
                    muted()
                };
                spans.push(Span::styled(format!("{} {}", i + 1, t.label()), style));
                spans.push(Span::raw("   "));
            }
        } else {
            let title = match &self.view {
                View::Job(id) => format!("job {id}"),
                View::Machine(node) => {
                    format!("machine {}", node.chars().take(8).collect::<String>())
                }
                View::Tab(t) => t.label().to_string(),
            };
            spans.push(Span::styled(title, accent().add_modifier(Modifier::BOLD)));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
        let who = format!(
            "{} ({}) · {} ",
            self.my_email(),
            s(&self.data.me["user"]["role"]),
            self.api.base()
        );
        f.render_widget(
            Paragraph::new(Span::styled(who, muted())).alignment(Alignment::Right),
            area,
        );
    }

    fn draw_footer(&self, f: &mut Frame, area: Rect) {
        let hints = if self.locked {
            " q quit   r refresh   ? help"
        } else {
            match self.view {
                View::Tab(Tab::Jobs) | View::Tab(Tab::Fleet) | View::Tab(Tab::Machines) => {
                    " ↑↓ select   enter open   1-5 tabs   r refresh   ? help   q quit"
                }
                View::Tab(_) => " 1-5 tabs   tab next   r refresh   ? help   q quit",
                _ => " esc back   r refresh   ? help   q quit",
            }
        };
        let left = match &self.error {
            Some(e) => Span::styled(format!(" ✖ {e}"), Style::default().fg(Color::Red)),
            None => Span::styled(hints, muted()),
        };
        f.render_widget(Paragraph::new(left), area);
        let age = self.last_fetch.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        let right = format!("↻ {} s · updated {age} s ago ", self.interval.as_secs());
        f.render_widget(
            Paragraph::new(Span::styled(right, muted())).alignment(Alignment::Right),
            area,
        );
    }

    fn draw_help(&self, f: &mut Frame) {
        let area = f.area();
        let w = 64.min(area.width.saturating_sub(4));
        let h = 15.min(area.height.saturating_sub(2));
        let rect = Rect::new(
            area.x + (area.width - w) / 2,
            area.y + (area.height - h) / 2,
            w,
            h,
        );
        f.render_widget(Clear, rect);
        let lines = vec![
            Line::from(vec![
                Span::styled("1-5  o j f m s", bold()),
                Span::raw("  switch tab"),
            ]),
            Line::from(vec![
                Span::styled("tab / shift-tab", bold()),
                Span::raw("  next, previous tab"),
            ]),
            Line::from(vec![
                Span::styled("↑ ↓  pgup pgdn", bold()),
                Span::raw("  select a row"),
            ]),
            Line::from(vec![
                Span::styled("enter", bold()),
                Span::raw("  open the job or the machine"),
            ]),
            Line::from(vec![
                Span::styled("esc", bold()),
                Span::raw("  back to the list"),
            ]),
            Line::from(vec![Span::styled("r", bold()), Span::raw("  refresh now")]),
            Line::from(vec![Span::styled("q", bold()), Span::raw("  quit")]),
            Line::from(""),
            Line::from(Span::styled(
                "Same data in the terminal, one command each:",
                muted(),
            )),
            Line::from(Span::styled(
                "plasmon job watch <id> · plasmon fleet --watch",
                muted(),
            )),
            Line::from(Span::styled(
                "plasmon fleet show <node> --watch · plasmon server status --watch",
                muted(),
            )),
        ];
        f.render_widget(
            Paragraph::new(lines).wrap(Wrap { trim: false }).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(accent())
                    .title(Span::styled(" keys ", bold()))
                    .padding(ratatui::widgets::Padding::horizontal(1)),
            ),
            rect,
        );
    }

    fn card(&self, f: &mut Frame, area: Rect, title: &str, value: Line, sub: Line) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(muted())
            .title(Span::styled(format!(" {title} "), muted()));
        let inner = block.inner(area);
        f.render_widget(block, area);
        let rows =
            Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).split(Rect::new(
                inner.x + 1,
                inner.y,
                inner.width.saturating_sub(2),
                inner.height,
            ));
        f.render_widget(Paragraph::new(value), rows[0]);
        if rows.len() > 1 {
            f.render_widget(Paragraph::new(sub), rows[1]);
        }
    }

    /// `████▓▓░░` with one coloured segment per status, in proportion to the counts.
    fn status_bar(&self, width: usize) -> Line<'static> {
        let by = &self.data.summary["by_status"];
        let order = [
            "training",
            "idle",
            "paused",
            "unavailable",
            "offline",
            "error",
        ];
        let total: u64 = order.iter().map(|k| by[k].as_u64().unwrap_or(0)).sum();
        if total == 0 {
            return Line::from(Span::styled("░".repeat(width), muted()));
        }
        let mut spans = Vec::new();
        let mut used = 0usize;
        for (i, k) in order.iter().enumerate() {
            let n = by[k].as_u64().unwrap_or(0);
            if n == 0 {
                continue;
            }
            let mut w = ((n as f64 / total as f64) * width as f64).round() as usize;
            if i == order.len() - 1 || used + w > width {
                w = width.saturating_sub(used);
            }
            w = w.max(1);
            used += w;
            spans.push(Span::styled(
                "█".repeat(w),
                Style::default().fg(status_color(k)),
            ));
        }
        Line::from(spans)
    }

    fn draw_overview(&mut self, f: &mut Frame, area: Rect) {
        let n_jobs = arr(&self.data.jobs).len() as u16;
        let jobs_h = (n_jobs + 3).clamp(4, area.height.saturating_sub(4) / 2);
        let rows = Layout::vertical([
            Constraint::Length(4),
            Constraint::Length(jobs_h),
            Constraint::Min(4),
        ])
        .split(area);
        let cards = Layout::horizontal([
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
        ])
        .split(rows[0]);
        let sm = &self.data.summary;
        let by = &sm["by_status"];
        let online = sm["online"].as_u64().unwrap_or(0);
        let machines = sm["machines"].as_u64().unwrap_or(0);
        let bar_w = cards[0].width.saturating_sub(4) as usize;
        self.card(
            f,
            cards[0],
            "machines online",
            Line::from(vec![
                Span::styled(format!("{online}"), bold().fg(ACCENT)),
                Span::styled(format!(" / {machines}"), muted()),
                Span::raw("  "),
                Span::styled(
                    format!(
                        "{} training · {} idle",
                        by["training"].as_u64().unwrap_or(0),
                        by["idle"].as_u64().unwrap_or(0)
                    ),
                    muted(),
                ),
            ]),
            self.status_bar(bar_w),
        );
        let running = arr(&self.data.jobs)
            .iter()
            .filter(|j| j["status"] == "running")
            .count();
        let completed = arr(&self.data.jobs)
            .iter()
            .filter(|j| j["status"] == "completed")
            .count();
        self.card(
            f,
            cards[1],
            "jobs running",
            Line::from(vec![
                Span::styled(format!("{running}"), bold().fg(ACCENT)),
                Span::styled(format!("   {completed} completed"), muted()),
            ]),
            Line::from(Span::styled(
                format!("{} rounds in the last hour", sm["rounds_last_hour"]),
                muted(),
            )),
        );
        let sched = &self.data.server["scheduler"]["running"];
        let (sched_txt, sched_color) = match sched.as_bool() {
            Some(true) => ("running", Color::Green),
            Some(false) => ("stopped", Color::Red),
            None => ("operator role shows it", MUTED),
        };
        self.card(
            f,
            cards[2],
            "scheduler",
            Line::from(Span::styled(sched_txt, bold().fg(sched_color))),
            Line::from(Span::styled(
                match self.data.server["uptime_s"].as_f64() {
                    Some(u) => format!(
                        "up {} min · {} ledger entries",
                        (u / 60.0) as u64,
                        self.data.server["ledger"]["entries"]
                    ),
                    None => format!("mode {}", s(&self.data.me["mode"])),
                },
                muted(),
            )),
        );
        let errors = by["error"].as_u64().unwrap_or(0);
        let offline = by["offline"].as_u64().unwrap_or(0);
        let (att_val, att_color) = if errors + offline == 0 {
            ("all clear".to_string(), Color::Green)
        } else {
            (format!("{}", errors + offline), Color::Red)
        };
        self.card(
            f,
            cards[3],
            "attention",
            Line::from(Span::styled(att_val, bold().fg(att_color))),
            Line::from(vec![
                Span::styled(
                    format!("error {errors}"),
                    Style::default().fg(status_color("error")),
                ),
                Span::styled(" · ", muted()),
                Span::styled(
                    format!("offline {offline}"),
                    Style::default().fg(status_color("offline")),
                ),
            ]),
        );

        // active jobs: name, state, progress, loss, accuracy, sparkline
        let jobs: Vec<&Value> = arr(&self.data.jobs)
            .iter()
            .filter(|j| j["status"] == "running")
            .chain(
                arr(&self.data.jobs)
                    .iter()
                    .filter(|j| j["status"] != "running")
                    .take(5),
            )
            .collect();
        let trows: Vec<Row> = jobs
            .iter()
            .map(|j| {
                let round = j["round"].as_f64().unwrap_or(0.0);
                let total = j["total_rounds"].as_f64().unwrap_or(1.0).max(1.0);
                let losses: Vec<f64> = self
                    .data
                    .losses
                    .get(&s(&j["id"]))
                    .cloned()
                    .unwrap_or_default();
                Row::new(vec![
                    Cell::from(Span::styled(s(&j["name"]), bold())),
                    Cell::from(chip(j["status"].as_str().unwrap_or(""))),
                    Cell::from(Line::from(vec![
                        progress(round / total, 10),
                        Span::styled(format!(" {}/{}", j["round"], j["total_rounds"]), muted()),
                    ])),
                    Cell::from(f3(&j["eval_loss"])),
                    Cell::from(pct_str(&j["eval_acc"])),
                    Cell::from(Span::styled(spark(&losses, 20), accent())),
                    Cell::from(Span::styled(s(&j["owner"]), muted())),
                ])
            })
            .collect();
        let empty = trows.is_empty();
        let table = Table::new(
            trows,
            [
                Constraint::Length(20),
                Constraint::Length(12),
                Constraint::Length(20),
                Constraint::Length(9),
                Constraint::Length(8),
                Constraint::Length(22),
                Constraint::Min(10),
            ],
        )
        .header(
            Row::new([
                "job",
                "state",
                "round",
                "eval loss",
                "acc",
                "loss, last rounds",
                "owner",
            ])
            .style(muted()),
        )
        .block(section(" jobs "));
        f.render_widget(table, rows[1]);
        if empty {
            f.render_widget(
                Paragraph::new(Span::styled(
                    "No jobs yet. Submit one: plasmon job submit examples/mnist/job.yaml",
                    muted(),
                )),
                Rect::new(
                    rows[1].x + 2,
                    rows[1].y + 2,
                    rows[1].width.saturating_sub(4),
                    1,
                ),
            );
        }

        // your machines
        let mine = self.mine();
        let mrows: Vec<Row> = mine.iter().map(|m| machine_row(m, false, false)).collect();
        let none = mrows.is_empty();
        let table = Table::new(mrows, machine_widths(false))
            .header(Row::new(machine_headers(false)).style(muted()))
            .block(section(" your machines "));
        f.render_widget(table, rows[2]);
        if none {
            f.render_widget(
                Paragraph::new(Span::styled(
                    "No machine of yours is enrolled. Run: plasmon trainer start",
                    muted(),
                )),
                Rect::new(
                    rows[2].x + 2,
                    rows[2].y + 2,
                    rows[2].width.saturating_sub(4),
                    1,
                ),
            );
        }
    }

    fn draw_jobs(&mut self, f: &mut Frame, area: Rect) {
        // The open-jobs table answers "what could my machine train?"; it shares the tab with the
        // jobs list because both are the same marketplace seen from the two sides.
        let n_open = arr(&self.data.open).len() as u16;
        let open_h = (n_open + 3).clamp(4, area.height.saturating_sub(6) / 2);
        let parts = Layout::vertical([Constraint::Min(6), Constraint::Length(open_h)]).split(area);
        self.draw_jobs_table(f, parts[0]);
        self.draw_open_jobs(f, parts[1]);
    }

    fn draw_open_jobs(&self, f: &mut Frame, area: Rect) {
        let rows: Vec<Row> = arr(&self.data.open)
            .iter()
            .map(|j| {
                let cells = crate::commands::open_job_row(j);
                Row::new(vec![
                    Cell::from(Span::styled(cells[1].clone(), bold())),
                    Cell::from(Span::styled(cells[2].clone(), muted())),
                    Cell::from(cells[3].clone()),
                    Cell::from(cells[4].clone()),
                    Cell::from(cells[5].clone()),
                    Cell::from(cells[6].clone()),
                    Cell::from(cells[7].clone()),
                    Cell::from(Span::styled(cells[8].clone(), accent())),
                ])
            })
            .collect();
        let empty = rows.is_empty();
        let table = Table::new(
            rows,
            [
                Constraint::Length(20),
                Constraint::Length(22),
                Constraint::Length(18),
                Constraint::Length(22),
                Constraint::Length(16),
                Constraint::Length(8),
                Constraint::Length(9),
                Constraint::Min(14),
            ],
        )
        .header(
            Row::new([
                "open job",
                "owner",
                "pays",
                "needs",
                "enrolment",
                "round",
                "trainers",
                "your machines",
            ])
            .style(muted()),
        )
        .block(section(" open jobs · plasmon trainer join <id> "));
        f.render_widget(table, area);
        if empty {
            f.render_widget(
                Paragraph::new(Span::styled(
                    "No job is running. One in automatic mode reaches your machines on its own.",
                    muted(),
                )),
                Rect::new(area.x + 2, area.y + 2, area.width.saturating_sub(4), 1),
            );
        }
    }

    fn draw_jobs_table(&mut self, f: &mut Frame, area: Rect) {
        let sel = self.jobs_state.selected();
        let rows: Vec<Row> = arr(&self.data.jobs)
            .iter()
            .enumerate()
            .map(|(i, j)| {
                let plain = sel == Some(i);
                let round = j["round"].as_f64().unwrap_or(0.0);
                let total = j["total_rounds"].as_f64().unwrap_or(1.0).max(1.0);
                let cells: Vec<Line> = vec![
                    Line::from(Span::styled(s(&j["name"]), bold())),
                    Line::from(Span::styled(s(&j["id"]), muted())),
                    Line::from(chip(j["status"].as_str().unwrap_or(""))),
                    Line::from(vec![
                        progress(round / total, 10),
                        Span::styled(format!(" {}/{}", j["round"], j["total_rounds"]), muted()),
                    ]),
                    Line::from(f3(&j["eval_loss"])),
                    Line::from(pct_str(&j["eval_acc"])),
                    Line::from(s(&j["spec"]["model"]["arch"])),
                    Line::from(Span::styled(s(&j["owner"]), muted())),
                ];
                Row::new(cells.into_iter().map(|c| Cell::from(plainify(c, plain))))
            })
            .collect();
        let empty = rows.is_empty();
        let table = Table::new(
            rows,
            [
                Constraint::Length(20),
                Constraint::Length(18),
                Constraint::Length(13),
                Constraint::Length(20),
                Constraint::Length(9),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Min(10),
            ],
        )
        .header(
            Row::new([
                "job",
                "id",
                "state",
                "round",
                "eval loss",
                "acc",
                "model",
                "owner",
            ])
            .style(muted()),
        )
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
        .highlight_symbol("▸ ")
        .block(section(" all jobs "));
        f.render_stateful_widget(table, area, &mut self.jobs_state);
        if empty {
            f.render_widget(
                Paragraph::new(Span::styled("No jobs yet.", muted())),
                Rect::new(area.x + 2, area.y + 2, area.width.saturating_sub(4), 1),
            );
        }
    }

    fn draw_fleet(&mut self, f: &mut Frame, area: Rect, mine_only: bool) {
        let list: Vec<Value> = if mine_only {
            self.mine()
        } else {
            arr(&self.data.fleet).to_vec()
        };
        let by = &self.data.summary["by_status"];
        let title = if mine_only {
            " my machines ".to_string()
        } else {
            format!(
                " fleet · {} online · {} training · {} idle · {} paused · {} offline · {} error ",
                self.data.summary["online"],
                by["training"].as_u64().unwrap_or(0),
                by["idle"].as_u64().unwrap_or(0),
                by["paused"].as_u64().unwrap_or(0),
                by["offline"].as_u64().unwrap_or(0),
                by["error"].as_u64().unwrap_or(0)
            )
        };
        let sel = if mine_only {
            self.mine_state.selected()
        } else {
            self.fleet_state.selected()
        };
        let rows: Vec<Row> = list
            .iter()
            .enumerate()
            .map(|(i, m)| machine_row(m, true, sel == Some(i)))
            .collect();
        let empty = rows.is_empty();
        let table = Table::new(rows, machine_widths(true))
            .header(Row::new(machine_headers(true)).style(muted()))
            .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
            .highlight_symbol("▸ ")
            .block(section(&title));
        let state = if mine_only {
            &mut self.mine_state
        } else {
            &mut self.fleet_state
        };
        f.render_stateful_widget(table, area, state);
        if empty {
            f.render_widget(
                Paragraph::new(Span::styled(
                    if mine_only {
                        "No machine of yours is enrolled. Run: plasmon trainer start"
                    } else {
                        "No machines yet."
                    },
                    muted(),
                )),
                Rect::new(area.x + 2, area.y + 2, area.width.saturating_sub(4), 1),
            );
        }
    }

    fn draw_server(&mut self, f: &mut Frame, area: Rect) {
        let st = &self.data.server;
        if st.is_null() {
            f.render_widget(
                Paragraph::new(Span::styled(
                    "Server status needs the operator role.",
                    muted(),
                ))
                .block(section(" server ")),
                area,
            );
            return;
        }
        let rows = Layout::vertical([
            Constraint::Length(4),
            Constraint::Length(4),
            Constraint::Min(4),
        ])
        .split(area);
        let cards = Layout::horizontal([Constraint::Ratio(1, 4); 4]).split(rows[0]);
        let running = st["scheduler"]["running"] == true;
        self.card(
            f,
            cards[0],
            "scheduler",
            Line::from(Span::styled(
                if running { "running" } else { "stopped" },
                bold().fg(if running { Color::Green } else { Color::Red }),
            )),
            Line::from(Span::styled(
                format!("v{} · {} mode", s(&st["version"]), s(&st["mode"])),
                muted(),
            )),
        );
        self.card(
            f,
            cards[1],
            "database",
            Line::from(vec![
                Span::styled(
                    format!("{} ms", num(&st["db"]["ping_ms"])),
                    bold().fg(ACCENT),
                ),
                Span::styled(" ping", muted()),
            ]),
            Line::from(Span::styled(
                s(&st["db"]["url"])
                    .chars()
                    .take(cards[1].width.saturating_sub(4) as usize)
                    .collect::<String>(),
                muted(),
            )),
        );
        let bytes = st["blobs"]["bytes"].as_f64().unwrap_or(0.0);
        self.card(
            f,
            cards[2],
            "blob store",
            Line::from(Span::styled(human_bytes(bytes), bold().fg(ACCENT))),
            Line::from(Span::styled(
                s(&st["blobs"]["path"])
                    .chars()
                    .take(cards[2].width.saturating_sub(4) as usize)
                    .collect::<String>(),
                muted(),
            )),
        );
        self.card(
            f,
            cards[3],
            "ledger",
            Line::from(vec![
                Span::styled(format!("{}", st["ledger"]["entries"]), bold().fg(ACCENT)),
                Span::styled(" entries", muted()),
            ]),
            Line::from(Span::styled(
                format!(
                    "head {}…",
                    s(&st["ledger"]["head"])
                        .chars()
                        .take(16)
                        .collect::<String>()
                ),
                muted(),
            )),
        );
        let counts = &st["counts"];
        let uptime = st["uptime_s"].as_f64().unwrap_or(0.0);
        let line = Line::from(vec![
            Span::styled("users ", muted()),
            Span::styled(format!("{}", counts["users"]), bold()),
            Span::styled("   machines ", muted()),
            Span::styled(format!("{}", counts["machines"]), bold()),
            Span::styled("   jobs ", muted()),
            Span::styled(format!("{}", counts["jobs"]), bold()),
            Span::styled(format!(" ({} running)", counts["jobs_running"]), muted()),
            Span::styled("   sse clients ", muted()),
            Span::styled(format!("{}", st["sse_clients"]), bold()),
            Span::styled("   uptime ", muted()),
            Span::styled(
                format!(
                    "{} h {} min",
                    (uptime / 3600.0) as u64,
                    ((uptime % 3600.0) / 60.0) as u64
                ),
                bold(),
            ),
        ]);
        f.render_widget(Paragraph::new(line).block(section(" counts ")), rows[1]);
        let trows: Vec<Row> = arr(&st["round_timings"])
            .iter()
            .rev()
            .map(|t| {
                Row::new(vec![
                    s(&t["job"]),
                    t["round"].to_string(),
                    format!("{} s", num(&t["aggregate_s"])),
                    format!("{} s", num(&t["eval_s"])),
                ])
            })
            .collect();
        let table = Table::new(
            trows,
            [
                Constraint::Length(20),
                Constraint::Length(7),
                Constraint::Length(12),
                Constraint::Length(10),
            ],
        )
        .header(Row::new(["job", "round", "aggregate", "eval"]).style(muted()))
        .block(section(" recent round timings "));
        f.render_widget(table, rows[2]);
    }

    fn draw_job(&mut self, f: &mut Frame, area: Rect, id: &str) {
        let job = &self.data.job;
        let points: Vec<(f64, f64)> = arr(&job["rounds"])
            .iter()
            .filter(|r| r["status"] == "closed")
            .filter_map(|r| Some((r["index"].as_f64()?, r["eval_loss"].as_f64()?)))
            .collect();
        let n_rounds = arr(&job["rounds"]).len() as u16;
        let rows = Layout::vertical([
            Constraint::Length(2),
            if points.len() >= 2 {
                Constraint::Percentage(40)
            } else {
                Constraint::Length(3)
            },
            Constraint::Length((n_rounds + 3).clamp(4, area.height / 3)),
            Constraint::Min(4),
        ])
        .split(area);
        let round = job["round"].as_f64().unwrap_or(0.0);
        let total = job["total_rounds"].as_f64().unwrap_or(1.0).max(1.0);
        let head = Line::from(vec![
            Span::styled(format!(" {} ", s(&job["name"])), bold()),
            Span::styled(id.to_string(), muted()),
            Span::raw("  "),
            chip(job["status"].as_str().unwrap_or("")),
            Span::raw("   "),
            progress(round / total, 16),
            Span::styled(
                format!(" round {}/{}", job["round"], job["total_rounds"]),
                muted(),
            ),
            Span::styled("   eval loss ", muted()),
            Span::styled(f3(&job["eval_loss"]), bold()),
            Span::styled("   acc ", muted()),
            Span::styled(pct_str(&job["eval_acc"]), bold()),
            Span::styled(
                format!(
                    "   {} params · {} shards · {}",
                    job["param_count"],
                    job["shards"],
                    s(&job["spec"]["model"]["arch"])
                ),
                muted(),
            ),
        ]);
        let mut head_lines = vec![head];
        if let Some(w) = job["waiting_reason"].as_str() {
            head_lines.push(Line::from(Span::styled(
                format!(" waiting: {w}"),
                Style::default().fg(Color::Yellow),
            )));
        }
        f.render_widget(Paragraph::new(head_lines), rows[0]);

        // eval loss per closed round as a line chart
        if points.len() >= 2 {
            let (xmin, xmax) = (points[0].0, points[points.len() - 1].0);
            let (ymin, ymax) = points.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
                (lo.min(p.1), hi.max(p.1))
            });
            let pad = ((ymax - ymin) * 0.1).max(0.01);
            let dataset = Dataset::default()
                .name("eval loss")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(accent())
                .data(&points);
            let chart = Chart::new(vec![dataset])
                .block(section(" eval loss per round "))
                .x_axis(
                    Axis::default()
                        .style(muted())
                        .bounds([xmin, xmax])
                        .labels(vec![
                            format!("{xmin}"),
                            format!("{}", ((xmin + xmax) / 2.0).round()),
                            format!("{xmax}"),
                        ]),
                )
                .y_axis(
                    Axis::default()
                        .style(muted())
                        .bounds([ymin - pad, ymax + pad])
                        .labels(vec![
                            format!("{:.3}", ymin - pad),
                            format!("{:.3}", (ymin + ymax) / 2.0),
                            format!("{:.3}", ymax + pad),
                        ]),
                );
            f.render_widget(chart, rows[1]);
        } else {
            f.render_widget(
                Paragraph::new(Span::styled(
                    "The chart fills in after two closed rounds.",
                    muted(),
                ))
                .block(section(" eval loss per round ")),
                rows[1],
            );
        }

        let rrows: Vec<Row> = arr(&job["rounds"])
            .iter()
            .rev()
            .map(|r| {
                Row::new(vec![
                    Cell::from(r["index"].to_string()),
                    Cell::from(chip(r["status"].as_str().unwrap_or(""))),
                    Cell::from(format!("{}", r["accepted"])),
                    Cell::from(f3(&r["eval_loss"])),
                    Cell::from(pct_str(&r["eval_acc"])),
                    Cell::from(human_bytes(r["bytes_in"].as_f64().unwrap_or(0.0))),
                    Cell::from(Span::styled(
                        r["timings"]["aggregate_s"]
                            .as_f64()
                            .map(|v| format!("agg {v:.1} s"))
                            .unwrap_or_default(),
                        muted(),
                    )),
                ])
            })
            .collect();
        let table = Table::new(
            rrows,
            [
                Constraint::Length(6),
                Constraint::Length(12),
                Constraint::Length(9),
                Constraint::Length(10),
                Constraint::Length(9),
                Constraint::Length(10),
                Constraint::Min(8),
            ],
        )
        .header(
            Row::new([
                "round",
                "status",
                "trainers",
                "eval loss",
                "acc",
                "bytes in",
                "",
            ])
            .style(muted()),
        )
        .block(section(" rounds "));
        f.render_widget(table, rows[2]);

        let urows: Vec<Row> = arr(&self.data.updates)
            .iter()
            .rev()
            .take(60)
            .map(|u| {
                Row::new(vec![
                    Cell::from(u["round"].to_string()),
                    Cell::from(Span::styled(s(&u["machine"]), bold())),
                    Cell::from(u["shard"].to_string()),
                    Cell::from(chip(u["status"].as_str().unwrap_or(""))),
                    Cell::from(u["samples"].to_string()),
                    Cell::from(f3(&u["loss_end"])),
                    Cell::from(f3(&u["score"])),
                    Cell::from(Span::styled(
                        s(&u["reject_reason"]),
                        Style::default().fg(Color::Red),
                    )),
                ])
            })
            .collect();
        let table = Table::new(
            urows,
            [
                Constraint::Length(6),
                Constraint::Length(18),
                Constraint::Length(6),
                Constraint::Length(12),
                Constraint::Length(8),
                Constraint::Length(9),
                Constraint::Length(7),
                Constraint::Min(8),
            ],
        )
        .header(
            Row::new([
                "round", "machine", "shard", "status", "samples", "loss end", "score", "",
            ])
            .style(muted()),
        )
        .block(section(" trainers "));
        f.render_widget(table, rows[3]);
    }

    fn draw_machine(&mut self, f: &mut Frame, area: Rect, node: &str) {
        let m = &self.data.machine;
        let rows = Layout::vertical([
            Constraint::Length(2),
            Constraint::Length(3),
            Constraint::Length(5),
            Constraint::Min(4),
        ])
        .split(area);
        let status = m["status"].as_str().unwrap_or("");
        let gpu = &m["hardware"]["gpu"];
        let gpu_name = if gpu["kind"].as_str().unwrap_or("none") == "none" {
            "cpu only".to_string()
        } else {
            format!("{} · {} GB", s(&gpu["name"]), num(&gpu["vram_gb"]))
        };
        let head = vec![
            Line::from(vec![
                Span::styled(format!(" {} ", s(&m["name"])), bold()),
                Span::styled(node.chars().take(12).collect::<String>(), muted()),
                Span::raw("  "),
                chip(status),
                Span::raw("  "),
                Span::raw(s(&m["status_detail"])),
                Span::styled(
                    format!(
                        "   {}   seen {} ago",
                        match m["current_job_id"].as_str() {
                            Some(j) => format!("job {j} round {}", s(&m["current_round"])),
                            None => "no job assigned".to_string(),
                        },
                        ago(&m["last_seen_at"])
                    ),
                    muted(),
                ),
            ]),
            Line::from(Span::styled(
                format!(
                    " {} · {} · owner {} · rounds served {} · samples verified {} · honesty {}",
                    s(&m["hardware"]["os"]),
                    gpu_name,
                    s(&m["owner"]),
                    m["rounds_served"],
                    m["samples_verified"],
                    f3(&m["honesty"])
                ),
                muted(),
            )),
        ];
        f.render_widget(Paragraph::new(head), rows[0]);
        let met = &m["metrics"];
        let cols = Layout::horizontal([Constraint::Ratio(1, 3); 3]).split(rows[1]);
        for (i, (label, key)) in [
            ("cpu", "cpu_pct"),
            ("memory", "ram_pct"),
            ("gpu", "gpu_pct"),
        ]
        .iter()
        .enumerate()
        {
            let pct = met[*key].as_f64();
            let g = Gauge::default()
                .block(
                    Block::default()
                        .title(Span::styled(format!(" {label} "), muted()))
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .border_style(muted()),
                )
                .gauge_style(Style::default().fg(match pct {
                    Some(p) if p >= 90.0 => Color::Red,
                    Some(p) if p >= 60.0 => Color::Yellow,
                    Some(_) => Color::Green,
                    None => MUTED,
                }))
                .label(match pct {
                    Some(p) => format!("{p:.0} %"),
                    None => "–".into(),
                })
                .percent(pct.unwrap_or(0.0).clamp(0.0, 100.0) as u16);
            f.render_widget(g, cols[i]);
        }
        let hist: Vec<u64> = arr(&m["history"])
            .iter()
            .filter_map(|h| h["metrics"]["cpu_pct"].as_f64())
            .map(|v| v as u64)
            .collect();
        f.render_widget(
            Sparkline::default()
                .data(&hist)
                .max(100)
                .style(accent())
                .block(section(" cpu, last hour ")),
            rows[2],
        );
        let lines: Vec<Line> = arr(&self.data.logs)
            .iter()
            .map(|l| {
                let at = s(&l["at"]);
                let level = s(&l["level"]);
                let color = match level.as_str() {
                    "error" => Color::Red,
                    "warning" | "warn" => Color::Yellow,
                    _ => MUTED,
                };
                Line::from(vec![
                    Span::styled(format!("{} ", at.get(11..19).unwrap_or(&at)), muted()),
                    Span::styled(format!("{level:<5} "), Style::default().fg(color)),
                    Span::raw(s(&l["message"])),
                ])
            })
            .collect();
        let empty = lines.is_empty();
        f.render_widget(
            Paragraph::new(if empty {
                vec![Line::from(Span::styled("No log lines yet.", muted()))]
            } else {
                lines
            })
            .block(section(" log ")),
            rows[3],
        );
    }
}

/// Strips colours from a line so the reversed selection highlight reads as one block.
fn plainify(line: Line<'static>, plain: bool) -> Line<'static> {
    if !plain {
        return line;
    }
    Line::from(
        line.spans
            .into_iter()
            .map(|sp| Span::raw(sp.content))
            .collect::<Vec<_>>(),
    )
}
fn gpu_short(name: &str) -> String {
    name.replace("NVIDIA GeForce ", "")
        .replace("NVIDIA ", "")
        .replace("AMD Radeon ", "")
}

fn section(title: &str) -> Block<'static> {
    Block::default()
        .borders(Borders::TOP)
        .border_style(muted())
        .title(Span::styled(title.to_string(), bold()))
}

fn human_bytes(b: f64) -> String {
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} kB", b / 1e3)
    } else {
        format!("{b:.0} B")
    }
}

fn machine_headers(full: bool) -> Vec<&'static str> {
    if full {
        vec![
            "machine",
            "status",
            "detail",
            "gpu",
            "cpu",
            "mem",
            "gpu %",
            "job / round",
            "owner",
            "honesty",
            "seen",
        ]
    } else {
        vec![
            "machine",
            "status",
            "detail",
            "cpu",
            "mem",
            "gpu %",
            "job / round",
            "seen",
        ]
    }
}
fn machine_widths(full: bool) -> Vec<Constraint> {
    if full {
        vec![
            Constraint::Length(16),
            Constraint::Length(13),
            Constraint::Length(26),
            Constraint::Length(14),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(20),
            Constraint::Length(18),
            Constraint::Length(7),
            Constraint::Min(6),
        ]
    } else {
        vec![
            Constraint::Length(16),
            Constraint::Length(13),
            Constraint::Length(24),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(20),
            Constraint::Min(6),
        ]
    }
}
fn machine_row(m: &Value, full: bool, plain: bool) -> Row<'static> {
    let met = &m["metrics"];
    let gpu = &m["hardware"]["gpu"];
    let has_gpu = gpu["kind"].as_str().unwrap_or("none") != "none";
    let status = if m["status"] == "idle" && !m["current_job_id"].is_null() {
        "training"
    } else {
        m["status"].as_str().unwrap_or("")
    };
    let job = match (m["current_job_id"].as_str(), m["current_round"].as_i64()) {
        (Some(j), Some(r)) => format!("{} r{r}", j.chars().take(14).collect::<String>()),
        (Some(j), None) => j.to_string(),
        _ => String::new(),
    };
    let detail = if status == "training" {
        match (met["step"].as_u64(), met["steps_total"].as_u64()) {
            (Some(a), Some(b)) if b > 0 && a < b => format!("step {a}/{b}"),
            _ => s(&m["status_detail"]),
        }
    } else {
        s(&m["status_detail"])
    };
    let mut cells: Vec<Line> = vec![
        Line::from(Span::styled(s(&m["name"]), bold())),
        Line::from(chip(status)),
        Line::from(Span::styled(detail, muted())),
    ];
    if full {
        cells.push(Line::from(if has_gpu {
            gpu_short(&s(&gpu["name"]))
        } else {
            "cpu".to_string()
        }));
    }
    cells.push(Line::from(vec![
        bar(met["cpu_pct"].as_f64(), 5),
        Span::styled(format!(" {}", num(&met["cpu_pct"])), muted()),
    ]));
    cells.push(Line::from(vec![
        bar(met["ram_pct"].as_f64(), 5),
        Span::styled(format!(" {}", num(&met["ram_pct"])), muted()),
    ]));
    cells.push(if has_gpu {
        Line::from(vec![
            bar(met["gpu_pct"].as_f64(), 5),
            Span::styled(format!(" {}", num(&met["gpu_pct"])), muted()),
        ])
    } else {
        Line::from(Span::styled("–", muted()))
    });
    cells.push(Line::from(job));
    if full {
        cells.push(Line::from(Span::styled(s(&m["owner"]), muted())));
        cells.push(Line::from(
            m["honesty"]
                .as_f64()
                .map(|v| format!("{v:.2}"))
                .unwrap_or_default(),
        ));
    }
    cells.push(Line::from(Span::styled(ago(&m["last_seen_at"]), muted())));
    Row::new(cells.into_iter().map(|c| Cell::from(plainify(c, plain))))
}

// ----- entry points used by main.rs -----------------------------------------------------------

/// The bare `plasmon` and `plasmon dashboard`: intro, then the tabbed dashboard.
pub fn dashboard(api: &Api, interval: Duration, animate: bool) -> Result<()> {
    Dash::new(api, interval, View::Tab(Tab::Overview), false).run(animate)
}
/// `plasmon job watch <id>`.
pub fn job_watch(api: &Api, id: &str, interval: Duration) -> Result<()> {
    Dash::new(api, interval, View::Job(id.to_string()), true).run(false)
}
/// `plasmon fleet --watch`.
pub fn fleet_watch(api: &Api, interval: Duration) -> Result<()> {
    Dash::new(api, interval, View::Tab(Tab::Fleet), true).run(false)
}
/// `plasmon fleet show <node> --watch`.
pub fn machine_watch(api: &Api, node: &str, interval: Duration) -> Result<()> {
    Dash::new(api, interval, View::Machine(node.to_string()), true).run(false)
}
/// `plasmon server status --watch`.
pub fn server_watch(api: &Api, interval: Duration) -> Result<()> {
    Dash::new(api, interval, View::Tab(Tab::Server), true).run(false)
}

/// Plain-mode summary for the bare `plasmon` without a TTY.
pub struct Home {
    pub me: Value,
    pub summary: Value,
}

pub fn fetch_home(api: &Api) -> Result<Home> {
    Ok(Home {
        me: api.get("/v1/auth/me")?,
        summary: api.get("/v1/fleet/summary")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn sample() -> Data {
        let fleet = serde_json::json!([
            {"node_id": "a".repeat(64), "name": "eduardo-mbp", "owner": "me@example.com", "status": "training", "status_detail": "step 19 of 50",
             "hardware": {"os": "Darwin 24.5.0", "gpu": {"kind": "mps", "name": "Apple GPU", "vram_gb": 32}},
             "metrics": {"cpu_pct": 38, "ram_pct": 46, "gpu_pct": 71, "step": 19, "steps_total": 50}, "current_job_id": "job_1", "current_round": 3,
             "honesty": 0.98, "last_seen_at": "2026-10-02T10:00:00"},
            {"node_id": "b".repeat(64), "name": "win-laptop", "owner": "other@example.com", "status": "paused", "status_detail": "paused by owner",
             "hardware": {"os": "Windows 11", "gpu": {"kind": "none"}}, "metrics": {"cpu_pct": 2, "ram_pct": 51}, "honesty": 1.0, "last_seen_at": "2026-10-02T10:00:00"}
        ]);
        let jobs = serde_json::json!([
            {"id": "job_1", "name": "mnist-cnn", "status": "running", "round": 3, "total_rounds": 20, "eval_loss": 0.41, "eval_acc": 0.883, "owner": "me@example.com",
             "param_count": 101770, "shards": 5, "spec": {"model": {"arch": "mnist_cnn"}},
             "rounds": [{"index": 0, "status": "closed", "accepted": 3, "eval_loss": 1.2, "eval_acc": 0.6, "bytes_in": 12000, "timings": {"aggregate_s": 0.4}},
                        {"index": 1, "status": "closed", "accepted": 3, "eval_loss": 0.8, "eval_acc": 0.75, "bytes_in": 12000},
                        {"index": 2, "status": "closed", "accepted": 2, "eval_loss": 0.41, "eval_acc": 0.883, "bytes_in": 9000},
                        {"index": 3, "status": "open", "accepted": 0}]}
        ]);
        Data {
            me: serde_json::json!({"user": {"email": "me@example.com", "role": "owner"}}),
            summary: serde_json::json!({"machines": 2, "online": 2, "by_status": {"training": 1, "paused": 1}, "rounds_last_hour": 3}),
            server: serde_json::json!({"scheduler": {"running": true}, "uptime_s": 4000, "version": "0.1.0", "mode": "home", "db": {"url": "sqlite:///x", "ping_ms": 1.2},
                                        "blobs": {"bytes": 123456, "path": "/data/blobs"}, "ledger": {"entries": 4, "head": "abcdef0123456789abcdef"},
                                        "counts": {"users": 1, "machines": 2, "jobs": 1, "jobs_running": 1}, "sse_clients": 0, "round_timings": []}),
            job: jobs[0].clone(),
            updates: serde_json::json!([{"round": 3, "machine": "eduardo-mbp", "shard": 1, "status": "revealed", "samples": 3200, "loss_end": 0.39, "score": 0.9}]),
            machine: fleet[0].clone(),
            logs: serde_json::json!([{"at": "2026-10-02T10:00:00", "level": "info", "message": "round 3: training shard 1"}]),
            losses: [("job_1".to_string(), vec![1.2, 0.8, 0.41])]
                .into_iter()
                .collect(),
            open: serde_json::json!([
                {"id": "job_2", "name": "llm-es", "owner": "r.vega@example.com", "funding": 2000, "per_round": 100, "credits_per_1k_samples": 0,
                 "requirements": {"device": "cuda", "min_vram_gb": 12, "min_tflops": 20, "min_honesty": 0}, "enrolment": {"mode": "join", "approval": "owner"},
                 "round": 4, "total_rounds": 20, "trainers_now": 3, "mine": [{"name": "eduardo-mbp", "standing": "can join"}]}
            ]),
            jobs,
            fleet,
        }
    }

    fn render(view: View, locked: bool) -> String {
        let api = Api::new("http://127.0.0.1:1", None).unwrap();
        let mut dash = Dash::new(&api, Duration::from_secs(2), view, locked);
        dash.data = sample();
        dash.last_fetch = Some(Instant::now());
        let mut terminal = Terminal::new(TestBackend::new(140, 40)).unwrap();
        terminal.draw(|f| dash.draw(f)).unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn overview_shows_cards_jobs_and_machines() {
        let text = render(View::Tab(Tab::Overview), false);
        for needle in [
            "machines online",
            "jobs running",
            "scheduler",
            "attention",
            "mnist-cnn",
            "eduardo-mbp",
            "1 overview",
            "me@example.com",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in\n{text}");
        }
        assert!(text.contains("▰"), "progress bars render");
    }

    #[test]
    fn every_view_renders() {
        for (view, needle) in [
            (View::Tab(Tab::Jobs), "all jobs"),
            (View::Tab(Tab::Fleet), "fleet ·"),
            (View::Tab(Tab::Machines), "my machines"),
            (View::Tab(Tab::Server), "blob store"),
            (View::Job("job_1".into()), "eval loss per round"),
            (View::Machine("a".repeat(64)), "cpu, last hour"),
        ] {
            let text = render(view, false);
            assert!(text.contains(needle), "missing {needle:?} in\n{text}");
        }
        let locked = render(View::Job("job_1".into()), true);
        assert!(locked.contains("job job_1") && !locked.contains("1 overview"));
    }

    #[test]
    fn logo_frames_resolve_to_the_wordmark() {
        let last = logo_frame(1.0);
        let text: Vec<String> = last
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.to_string()).collect())
            .collect();
        assert_eq!(text, LOGO.iter().map(|l| l.to_string()).collect::<Vec<_>>());
        let first = logo_frame(0.0);
        assert!(first
            .iter()
            .all(|l| l.spans.iter().all(|s| s.content.trim().is_empty())));
    }

    #[test]
    fn spark_and_bar_scale() {
        assert_eq!(spark(&[1.0, 2.0, 3.0], 10).chars().count(), 3);
        assert_eq!(spark(&[], 10), "");
        assert_eq!(bar(Some(50.0), 4).content, "▰▰▱▱");
        assert_eq!(bar(None, 3).content, "···");
    }
}
