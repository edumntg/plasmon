//! `plasmon`: the fast front end. Native commands talk to the coordinator directly;
//! `job submit`, `trainer` and `server` run through the Python engine.

mod api;
mod commands;
mod paths;
mod python;
mod tui;

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use plasmon_core::identity::Identity;
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "plasmon",
    version,
    about = "Train models on a fleet of volunteer GPUs"
)]
struct Cli {
    /// Print machine-readable JSON instead of text.
    #[arg(long, global = true)]
    json: bool,
    /// Coordinator URL. Default: the one you logged in to.
    #[arg(long, global = true)]
    server: Option<String>,
    /// Plain text, no animation, no alternate screen.
    #[arg(long, global = true)]
    plain: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Create this machine's Ed25519 key.
    Init {
        #[arg(long)]
        force: bool,
    },
    /// Log in to a coordinator with a device code confirmed in the browser.
    Login {
        #[arg(long)]
        server: String,
        #[arg(long)]
        no_browser: bool,
    },
    /// Forget the saved login.
    Logout,
    /// Show the local identity, the server and the user.
    Whoami,
    /// Submit and follow jobs.
    Job {
        #[command(subcommand)]
        cmd: JobCmd,
    },
    /// Offer this machine to the network.
    Trainer {
        #[command(subcommand)]
        cmd: TrainerCmd,
    },
    /// Run and manage a coordinator.
    Server {
        #[command(subcommand)]
        cmd: ServerCmd,
    },
    /// Machines you can see, and control of them (operator role).
    Fleet {
        #[arg(long)]
        status: Option<String>,
        /// Live table, refreshed every two seconds.
        #[arg(short, long)]
        watch: bool,
        #[command(subcommand)]
        cmd: Option<FleetCmd>,
    },
    /// People and roles (admin role).
    Users {
        #[command(subcommand)]
        cmd: UsersCmd,
    },
    /// Who did what, when (operator role).
    Audit {
        #[arg(long, default_value_t = 24)]
        since_hours: u32,
    },
    /// Org trainer policy.
    Policy {
        #[command(subcommand)]
        cmd: PolicyCmd,
    },
    /// The hash-chained ledger.
    Ledger {
        #[command(subcommand)]
        cmd: LedgerCmd,
    },
    /// Credits: your balance and movements; grants and balances for admins.
    Credits {
        #[command(subcommand)]
        cmd: Option<CreditsCmd>,
    },
    /// Full-screen view with tabs: jobs, fleet, my machines, server.
    Dashboard,
    /// Print shell completions.
    Completions {
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

#[derive(Subcommand)]
enum JobCmd {
    /// Validate a job.yaml, upload what the server lacks, create the job.
    Submit {
        file: String,
    },
    List {
        /// Every job in the org (operator role).
        #[arg(long)]
        all: bool,
    },
    Status {
        id: String,
    },
    /// Follow a job: loss, rounds, trainers.
    Watch {
        id: String,
        #[arg(long, default_value_t = 2.0)]
        interval: f64,
    },
    /// Save the latest weights as a safetensors file.
    Download {
        id: String,
        #[arg(short, long)]
        output: Option<String>,
    },
    Cancel {
        id: String,
    },
    /// Running jobs a trainer can join: pay, requirements, who approves.
    Open,
    /// Machines that asked to train a job of yours.
    Approvals {
        id: String,
    },
    /// Let a machine train your job.
    Approve {
        id: String,
        /// Machine name, node id or its prefix.
        machine: String,
        #[arg(long, default_value = "")]
        note: String,
    },
    /// Keep a machine off your job.
    Reject {
        id: String,
        machine: String,
        #[arg(long, default_value = "")]
        note: String,
    },
}

#[derive(Subcommand)]
enum FleetCmd {
    /// Stop a machine from taking rounds, now.
    Pause {
        node: String,
        #[arg(long, default_value = "")]
        reason: String,
    },
    /// Let a paused or draining machine take rounds again.
    Resume { node: String },
    /// Finish the current round, then stop taking rounds.
    Drain {
        node: String,
        #[arg(long, default_value = "")]
        reason: String,
    },
    /// One machine: status, metrics, rounds, log tail. Live with --watch.
    Show {
        node: String,
        #[arg(short, long)]
        watch: bool,
    },
    /// A machine's log. Follow with -f.
    Logs {
        node: String,
        #[arg(short, long)]
        follow: bool,
        #[arg(long)]
        grep: Option<String>,
    },
}

#[derive(Subcommand)]
enum UsersCmd {
    List,
    /// Create an invite link.
    Invite {
        #[arg(long)]
        email: Option<String>,
        #[arg(long, default_value = "member")]
        role: String,
        #[arg(long, default_value_t = 7)]
        days: u32,
    },
    SetRole {
        email: String,
        role: String,
    },
    Disable {
        email: String,
    },
    Enable {
        email: String,
    },
}

#[derive(Subcommand)]
enum CreditsCmd {
    /// Give credits to a user (admin).
    Grant {
        email: String,
        amount: i64,
        #[arg(long, default_value = "")]
        memo: String,
    },
    /// Balances of every user (admin).
    Users,
}

#[derive(Subcommand)]
enum PolicyCmd {
    Show,
    /// Set availability windows and the battery rule. Runs through the Python engine.
    Set {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

#[derive(Subcommand)]
enum TrainerCmd {
    /// Enrol this machine and train rounds until stopped.
    Start {
        #[arg(long)]
        name: Option<String>,
        #[arg(long, default_value = "any")]
        device: String,
        #[arg(long)]
        max_hours: Option<f64>,
        /// Only train inside this window, e.g. "weekdays 19:00-08:00". Repeatable.
        #[arg(long)]
        hours: Vec<String>,
        #[arg(long)]
        never_on_battery: bool,
    },
    /// Offer this machine to one open job (see `plasmon job open`).
    Join {
        job: String,
        /// Another machine of yours: name, node id or its prefix. Default: this machine.
        #[arg(long)]
        machine: Option<String>,
    },
    /// Take this machine off a job.
    Leave {
        job: String,
        #[arg(long)]
        machine: Option<String>,
    },
    /// Run the trainer at login as a user service (systemd, launchd or a scheduled task).
    Enable {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Remove the user service.
    Disable,
}

#[derive(Subcommand)]
enum ServerCmd {
    /// Write plasmon-server.yaml, or a Compose bundle with --bundle compose. All flags pass to the engine.
    Init {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Start the coordinator.
    Start,
    /// Create the owner account on an empty server.
    Bootstrap {
        #[arg(long)]
        owner: String,
        #[arg(long)]
        password: Option<String>,
    },
    /// Coordinator health (operator role). Live with --watch.
    Status {
        #[arg(short, long)]
        watch: bool,
    },
}

#[derive(Subcommand)]
enum LedgerCmd {
    /// Recompute the hash chain and check every signature.
    Verify,
}

fn main() {
    let code = match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            1
        }
    };
    std::process::exit(code);
}

fn run() -> Result<i32> {
    let cli = Cli::parse();
    let server = cli.server.as_deref();
    let plain = cli.plain || !tui::animation_enabled();
    match cli.command {
        None => {
            if paths::Credentials::load()?.is_none() {
                Cli::command().print_help()?;
                println!("\nStart with: plasmon login --server http://<host>:7117");
                return Ok(0);
            }
            let api = commands::api(server)?;
            if plain {
                let home = tui::fetch_home(&api)?;
                println!(
                    "{} · {} of {} machines online · {} jobs running",
                    home.me["user"]["email"].as_str().unwrap_or(""),
                    home.summary["online"],
                    home.summary["machines"],
                    home.summary["jobs_running"]
                );
                commands::job_list(server, false, false)?;
            } else {
                tui::dashboard(&api, Duration::from_secs(2), true)?;
            }
            Ok(0)
        }
        Some(Command::Init { force }) => init(force, cli.json).map(|_| 0),
        Some(Command::Login { server, no_browser }) => {
            commands::login(&server, no_browser, cli.json).map(|_| 0)
        }
        Some(Command::Logout) => commands::logout().map(|_| 0),
        Some(Command::Whoami) => commands::whoami(cli.json).map(|_| 0),
        Some(Command::Job { cmd }) => match cmd {
            JobCmd::Submit { file } => {
                python::run(&with_server(server, &["job", "submit", &file], cli.json))
            }
            JobCmd::List { all } => commands::job_list(server, all, cli.json).map(|_| 0),
            JobCmd::Status { id } => commands::job_status(server, &id, cli.json).map(|_| 0),
            JobCmd::Watch { id, interval } => {
                if plain || cli.json {
                    commands::job_watch_plain(server, &id, interval)
                } else {
                    tui::job_watch(
                        &commands::api(server)?,
                        &id,
                        Duration::from_secs_f64(interval),
                    )
                    .map(|_| 0)
                }
            }
            JobCmd::Download { id, output } => {
                commands::job_download(server, &id, output.as_deref()).map(|_| 0)
            }
            JobCmd::Cancel { id } => commands::job_cancel(server, &id).map(|_| 0),
            JobCmd::Open => commands::job_open(server, cli.json).map(|_| 0),
            JobCmd::Approvals { id } => commands::job_approvals(server, &id, cli.json).map(|_| 0),
            JobCmd::Approve { id, machine, note } => {
                commands::job_decide(server, &id, &machine, true, &note, cli.json).map(|_| 0)
            }
            JobCmd::Reject { id, machine, note } => {
                commands::job_decide(server, &id, &machine, false, &note, cli.json).map(|_| 0)
            }
        },
        Some(Command::Trainer { cmd }) => match cmd {
            TrainerCmd::Start {
                name,
                device,
                max_hours,
                hours,
                never_on_battery,
            } => {
                let mut args = vec![
                    "trainer".to_string(),
                    "start".to_string(),
                    "--device".to_string(),
                    device,
                ];
                if let Some(n) = name {
                    args.extend(["--name".to_string(), n]);
                }
                if let Some(h) = max_hours {
                    args.extend(["--max-hours".to_string(), h.to_string()]);
                }
                for h in hours {
                    args.extend(["--hours".to_string(), h]);
                }
                if never_on_battery {
                    args.push("--never-on-battery".to_string());
                }
                if !plain && !cli.json {
                    let target = server
                        .map(str::to_string)
                        .or_else(|| paths::Credentials::load().ok().flatten().map(|c| c.server))
                        .unwrap_or_default();
                    tui::intro_inline(&format!("trainer · {target} · Ctrl-C stops"))?;
                }
                python::run(&with_server_vec(server, args, false))
            }
            TrainerCmd::Join { job, machine } => {
                commands::trainer_enrol(server, &job, machine.as_deref(), true, cli.json).map(|_| 0)
            }
            TrainerCmd::Leave { job, machine } => {
                commands::trainer_enrol(server, &job, machine.as_deref(), false, cli.json)
                    .map(|_| 0)
            }
            TrainerCmd::Enable { args } => {
                let mut full = vec!["trainer".to_string(), "enable".to_string()];
                full.extend(args);
                python::run(&full)
            }
            TrainerCmd::Disable => python::run(&["trainer".to_string(), "disable".to_string()]),
        },
        Some(Command::Server { cmd }) => match cmd {
            ServerCmd::Init { args } => {
                let mut full = vec!["server".to_string(), "init".to_string()];
                full.extend(args);
                python::run(&full)
            }
            ServerCmd::Start => python::run(&["server".to_string(), "start".to_string()]),
            ServerCmd::Bootstrap { owner, password } => {
                let mut args = vec![
                    "server".to_string(),
                    "bootstrap".to_string(),
                    "--owner".to_string(),
                    owner,
                ];
                if let Some(p) = password {
                    args.extend(["--password".to_string(), p]);
                }
                python::run(&args)
            }
            ServerCmd::Status { watch } => {
                if watch && !plain && !cli.json {
                    tui::server_watch(&commands::api(server)?, Duration::from_secs(3)).map(|_| 0)
                } else {
                    commands::server_status(server, cli.json).map(|_| 0)
                }
            }
        },
        Some(Command::Fleet { status, watch, cmd }) => match cmd {
            None => {
                if watch && !plain && !cli.json {
                    tui::fleet_watch(&commands::api(server)?, Duration::from_secs(2)).map(|_| 0)
                } else {
                    commands::fleet(server, status.as_deref(), cli.json).map(|_| 0)
                }
            }
            Some(FleetCmd::Pause { node, reason }) => {
                commands::fleet_control(server, &node, "pause", &reason).map(|_| 0)
            }
            Some(FleetCmd::Resume { node }) => {
                commands::fleet_control(server, &node, "resume", "").map(|_| 0)
            }
            Some(FleetCmd::Drain { node, reason }) => {
                commands::fleet_control(server, &node, "drain", &reason).map(|_| 0)
            }
            Some(FleetCmd::Show { node, watch }) => {
                if watch && !plain && !cli.json {
                    tui::machine_watch(&commands::api(server)?, &node, Duration::from_secs(2))
                        .map(|_| 0)
                } else {
                    commands::fleet_show(server, &node, cli.json).map(|_| 0)
                }
            }
            Some(FleetCmd::Logs { node, follow, grep }) => {
                commands::fleet_logs(server, &node, follow, grep.as_deref()).map(|_| 0)
            }
        },
        Some(Command::Users { cmd }) => match cmd {
            UsersCmd::List => commands::users_list(server, cli.json).map(|_| 0),
            UsersCmd::Invite { email, role, days } => {
                commands::users_invite(server, email.as_deref(), &role, days, cli.json).map(|_| 0)
            }
            UsersCmd::SetRole { email, role } => {
                commands::users_set_role(server, &email, &role).map(|_| 0)
            }
            UsersCmd::Disable { email } => commands::users_disable(server, &email, true).map(|_| 0),
            UsersCmd::Enable { email } => commands::users_disable(server, &email, false).map(|_| 0),
        },
        Some(Command::Audit { since_hours }) => {
            commands::audit(server, since_hours, cli.json).map(|_| 0)
        }
        Some(Command::Policy { cmd }) => match cmd {
            PolicyCmd::Show => commands::policy_show(server, cli.json).map(|_| 0),
            PolicyCmd::Set { args } => {
                let mut full = vec!["policy".to_string(), "set".to_string()];
                full.extend(args);
                python::run(&with_server_vec(server, full, false))
            }
        },
        Some(Command::Ledger { cmd }) => match cmd {
            LedgerCmd::Verify => commands::ledger_verify(server, cli.json),
        },
        Some(Command::Credits { cmd }) => match cmd {
            None => commands::credits_me(server, cli.json).map(|_| 0),
            Some(CreditsCmd::Grant {
                email,
                amount,
                memo,
            }) => commands::credits_grant(server, &email, amount, &memo).map(|_| 0),
            Some(CreditsCmd::Users) => commands::credits_users(server, cli.json).map(|_| 0),
        },
        Some(Command::Dashboard) => {
            if plain || cli.json {
                commands::job_list(server, false, cli.json)?;
                commands::fleet(server, None, cli.json).map(|_| 0)
            } else {
                tui::dashboard(&commands::api(server)?, Duration::from_secs(2), true).map(|_| 0)
            }
        }
        Some(Command::Completions { shell }) => {
            clap_complete::generate(
                shell,
                &mut Cli::command(),
                "plasmon",
                &mut std::io::stdout(),
            );
            Ok(0)
        }
    }
}

fn with_server(server: Option<&str>, args: &[&str], json: bool) -> Vec<String> {
    with_server_vec(server, args.iter().map(|s| s.to_string()).collect(), json)
}

fn with_server_vec(server: Option<&str>, args: Vec<String>, json: bool) -> Vec<String> {
    let mut out = Vec::new();
    if json {
        out.push("--json".to_string());
    }
    if let Some(s) = server {
        out.extend(["--server".to_string(), s.to_string()]);
    }
    out.extend(args);
    out
}

fn init(force: bool, json: bool) -> Result<()> {
    let path = paths::machine_key();
    let (created, ident) = if path.exists() && !force {
        (
            false,
            Identity::load(&path).with_context(|| format!("reading {}", path.display()))?,
        )
    } else {
        let ident = Identity::generate();
        ident
            .save(&path)
            .with_context(|| format!("writing {}", path.display()))?;
        (true, ident)
    };
    if json {
        println!(
            "{}",
            serde_json::json!({"node_id": ident.node_id(), "path": path, "created": created})
        );
    } else if created {
        println!(
            "created machine key: {}\nnode id: {}",
            path.display(),
            ident.node_id()
        );
    } else {
        println!(
            "machine key exists: {}\nnode id: {}",
            path.display(),
            ident.node_id()
        );
    }
    Ok(())
}
