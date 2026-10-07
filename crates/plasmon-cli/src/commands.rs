//! Network commands implemented natively: login, whoami, job list/status/watch/download,
//! fleet, server status, ledger verify.

use crate::api::Api;
use crate::paths::{self, Credentials};
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::time::{Duration, Instant};

pub fn api(server_override: Option<&str>) -> Result<Api> {
    let creds = Credentials::load()?;
    let server = match (server_override, &creds) {
        (Some(s), _) => s.to_string(),
        (None, Some(c)) => c.server.clone(),
        (None, None) => bail!("not logged in. Run: plasmon login --server http://<host>:7117"),
    };
    let token = creds
        .filter(|c| c.server.trim_end_matches('/') == server.trim_end_matches('/'))
        .map(|c| c.token);
    if token.is_none() {
        bail!("not logged in to {server}. Run: plasmon login --server {server}");
    }
    Api::new(&server, token)
}

pub fn login(server: &str, no_browser: bool, json: bool) -> Result<()> {
    let api = Api::new(server, None)?;
    api.healthz()?;
    let host = hostname();
    let start = api.post(
        "/v1/auth/device",
        &serde_json::json!({"label": format!("cli on {host}")}),
    )?;
    let url = start["verification_uri_complete"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let code = start["user_code"].as_str().unwrap_or("").to_string();
    let styled = crate::tui::animation_enabled();
    let (b, d, r) = if styled {
        ("\x1b[1m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    eprintln!(
        "{b}{} plasmon{r} {d}· log in to {server}{r}\n\n  Open this address in a browser and confirm the code:\n  {}\n\n  code  {b}{code}{r}\n",
        crate::tui::MARK,
        crate::tui::link(&url)
    );
    if !no_browser {
        let _ = open::that(&url);
    }
    let interval = Duration::from_secs(start["interval"].as_u64().unwrap_or(3));
    let deadline =
        Instant::now() + Duration::from_secs(start["expires_in"].as_u64().unwrap_or(600));
    let device_code = start["device_code"].clone();
    let mut spinner = crate::tui::Spinner::new("waiting for the confirmation in the browser");
    while Instant::now() < deadline {
        spinner.tick();
        std::thread::sleep(interval);
        let reply = api.post(
            "/v1/auth/device/token",
            &serde_json::json!({"device_code": device_code}),
        )?;
        if reply.get("status").and_then(Value::as_str) == Some("pending") {
            continue;
        }
        spinner.done(&format!(
            "confirmed as {}",
            reply["user"]["email"].as_str().unwrap_or("")
        ));
        let creds = Credentials {
            server: server.trim_end_matches('/').to_string(),
            user: reply["user"]["email"].as_str().unwrap_or("").to_string(),
            token: reply["token"]
                .as_str()
                .context("no token in reply")?
                .to_string(),
        };
        creds.save()?;
        if json {
            println!(
                "{}",
                serde_json::json!({"server": creds.server, "user": reply["user"]})
            );
        } else {
            println!(
                "logged in to {} as {} ({})",
                creds.server,
                creds.user,
                reply["user"]["role"].as_str().unwrap_or("")
            );
        }
        return Ok(());
    }
    bail!("code expired; run login again")
}

pub fn logout() -> Result<()> {
    if let Some(c) = Credentials::load()? {
        if let Ok(api) = Api::new(&c.server, Some(c.token)) {
            let _ = api.post("/v1/auth/logout", &Value::Null);
        }
        let _ = std::fs::remove_file(paths::credentials());
    }
    println!("logged out");
    Ok(())
}

pub fn whoami(json: bool) -> Result<()> {
    let key_path = paths::machine_key();
    let node_id = key_path
        .exists()
        .then(|| plasmon_core::identity::Identity::load(&key_path).map(|i| i.node_id()))
        .transpose()?;
    let creds = Credentials::load()?;
    let role = match &creds {
        Some(c) => Api::new(&c.server, Some(c.token.clone()))?
            .get("/v1/auth/me")
            .ok()
            .and_then(|m| m["user"]["role"].as_str().map(str::to_string)),
        None => None,
    };
    if json {
        println!(
            "{}",
            serde_json::json!({"node_id": node_id, "server": creds.as_ref().map(|c| &c.server), "user": creds.as_ref().map(|c| &c.user), "role": role})
        );
        return Ok(());
    }
    match node_id {
        Some(id) => println!("node id: {id}"),
        None => println!("no machine key yet. Run: plasmon init"),
    }
    match creds {
        Some(c) => println!(
            "server:  {}\nuser:    {} ({})",
            c.server,
            c.user,
            role.unwrap_or_else(|| "unknown role".into())
        ),
        None => println!("not logged in. Run: plasmon login --server <url>"),
    }
    Ok(())
}

pub fn job_list(server: Option<&str>, all: bool, json: bool) -> Result<()> {
    let api = api(server)?;
    let jobs = api.get(if all { "/v1/jobs?all=true" } else { "/v1/jobs" })?;
    if json {
        println!("{}", serde_json::to_string_pretty(&jobs)?);
        return Ok(());
    }
    let rows: Vec<Vec<String>> = jobs
        .as_array()
        .map(|a| a.iter().map(job_row).collect())
        .unwrap_or_default();
    if rows.is_empty() {
        println!("no jobs");
    } else {
        print!(
            "{}",
            table(
                &["id", "name", "status", "round", "eval loss", "eval acc"],
                &rows
            )
        );
    }
    Ok(())
}

pub fn job_row(j: &Value) -> Vec<String> {
    vec![
        s(&j["id"]),
        s(&j["name"]),
        s(&j["status"]),
        format!("{}/{}", j["round"], j["total_rounds"]),
        j["eval_loss"]
            .as_f64()
            .map(|v| format!("{v:.3}"))
            .unwrap_or_default(),
        j["eval_acc"]
            .as_f64()
            .map(|v| format!("{:.1} %", v * 100.0))
            .unwrap_or_default(),
    ]
}

pub fn job_status(server: Option<&str>, id: &str, json: bool) -> Result<()> {
    let api = api(server)?;
    let job = api.get(&format!("/v1/jobs/{id}"))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&job)?);
        return Ok(());
    }
    println!(
        "{} ({})  {}  round {}/{}  params {}",
        s(&job["name"]),
        s(&job["id"]),
        s(&job["status"]),
        job["round"],
        job["total_rounds"],
        job["param_count"]
    );
    if let Some(w) = job["waiting_reason"].as_str() {
        println!("  waiting: {w}");
    }
    let rows: Vec<Vec<String>> = job["rounds"]
        .as_array()
        .map(|a| a.iter().map(round_row).collect())
        .unwrap_or_default();
    print!(
        "{}",
        table(
            &[
                "round",
                "status",
                "trainers",
                "eval loss",
                "eval acc",
                "bytes in"
            ],
            &rows
        )
    );
    Ok(())
}

pub fn round_row(r: &Value) -> Vec<String> {
    vec![
        r["index"].to_string(),
        s(&r["status"]),
        r["accepted"].to_string(),
        r["eval_loss"]
            .as_f64()
            .map(|v| format!("{v:.4}"))
            .unwrap_or_default(),
        r["eval_acc"]
            .as_f64()
            .map(|v| format!("{:.1} %", v * 100.0))
            .unwrap_or_default(),
        r["bytes_in"]
            .as_u64()
            .map(|v| v.to_string())
            .unwrap_or_default(),
    ]
}

pub fn job_watch_plain(server: Option<&str>, id: &str, interval: f64) -> Result<i32> {
    let api = api(server)?;
    let mut seen: i64 = -1;
    loop {
        let job = api.get(&format!("/v1/jobs/{id}"))?;
        for r in job["rounds"].as_array().into_iter().flatten() {
            let idx = r["index"].as_i64().unwrap_or(0);
            if r["status"] == "closed" && idx > seen {
                seen = idx;
                println!(
                    "round {idx:>4}  trainers {:>3}  eval loss {:.4}  acc {:.1} %  {} B in",
                    r["accepted"],
                    r["eval_loss"].as_f64().unwrap_or(0.0),
                    r["eval_acc"].as_f64().unwrap_or(0.0) * 100.0,
                    r["bytes_in"]
                );
            }
        }
        if job["status"] != "running" {
            println!("job {}", s(&job["status"]));
            return Ok(if job["status"] == "completed" { 0 } else { 1 });
        }
        std::thread::sleep(Duration::from_secs_f64(interval));
    }
}

pub fn job_download(server: Option<&str>, id: &str, output: Option<&str>) -> Result<()> {
    let api = api(server)?;
    let job = api.get(&format!("/v1/jobs/{id}"))?;
    let theta = job["theta"].as_str().context("job has no weights yet")?;
    let bytes = api.get_bytes(&format!("/v1/blobs/{theta}"))?;
    if plasmon_core::hashing::digest(&bytes) != theta {
        bail!("downloaded blob does not match its id");
    }
    let path = output
        .map(str::to_string)
        .unwrap_or_else(|| format!("{}.safetensors", s(&job["name"])));
    std::fs::write(&path, &bytes)?;
    println!(
        "wrote {path} ({} bytes, weights after round {}, blob {}…)",
        bytes.len(),
        job["round"],
        &theta[..12]
    );
    Ok(())
}

pub fn job_cancel(server: Option<&str>, id: &str) -> Result<()> {
    let api = api(server)?;
    let job = api.post(&format!("/v1/jobs/{id}/cancel"), &Value::Null)?;
    println!("job {} {}", s(&job["id"]), s(&job["status"]));
    Ok(())
}

pub fn fleet_rows(machines: &Value) -> Vec<Vec<String>> {
    machines
        .as_array()
        .map(|a| {
            a.iter()
                .map(|m| {
                    let met = &m["metrics"];
                    let gpu = &m["hardware"]["gpu"];
                    let gpu_name = if gpu["kind"].as_str().unwrap_or("none") == "none" {
                        "none".to_string()
                    } else {
                        s(&gpu["name"])
                    };
                    let job = match (m["current_job_id"].as_str(), m["current_round"].as_i64()) {
                        (Some(j), Some(r)) => format!("{j} r{r}"),
                        (Some(j), None) => j.to_string(),
                        _ => String::new(),
                    };
                    vec![
                        s(&m["name"]),
                        s(&m["owner"]),
                        s(&m["status"]),
                        gpu_name,
                        num(&met["gpu_pct"]),
                        num(&met["cpu_pct"]),
                        num(&met["ram_pct"]),
                        job,
                        m["honesty"]
                            .as_f64()
                            .map(|v| format!("{v:.2}"))
                            .unwrap_or_default(),
                        ago(&m["last_seen_at"]),
                    ]
                })
                .collect()
        })
        .unwrap_or_default()
}

pub const FLEET_HEADERS: [&str; 10] = [
    "machine",
    "owner",
    "status",
    "gpu",
    "gpu%",
    "cpu%",
    "ram%",
    "job / round",
    "honesty",
    "seen",
];

pub fn fleet(server: Option<&str>, status: Option<&str>, json: bool) -> Result<()> {
    let api = api(server)?;
    let path = match status {
        Some(st) => format!("/v1/fleet?status={st}"),
        None => "/v1/fleet".to_string(),
    };
    let machines = api.get(&path)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&machines)?);
        return Ok(());
    }
    let rows = fleet_rows(&machines);
    if rows.is_empty() {
        println!("no machines");
    } else {
        print!("{}", table(&FLEET_HEADERS, &rows));
    }
    Ok(())
}

pub fn server_status(server: Option<&str>, json: bool) -> Result<()> {
    let api = api(server)?;
    let st = api.get("/v1/server/status")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&st)?);
        return Ok(());
    }
    println!(
        "version {}  mode {}  uptime {} s\ndb {} ({} ms)\nblobs {} B at {}\nscheduler running: {}  sse clients: {}\nledger entries {}  head {}…\nusers {}  machines {}  jobs {} ({} running)",
        s(&st["version"]), s(&st["mode"]), st["uptime_s"], s(&st["db"]["url"]), st["db"]["ping_ms"], st["blobs"]["bytes"], s(&st["blobs"]["path"]),
        st["scheduler"]["running"], st["sse_clients"], st["ledger"]["entries"], &s(&st["ledger"]["head"])[..16.min(s(&st["ledger"]["head"]).len())],
        st["counts"]["users"], st["counts"]["machines"], st["counts"]["jobs"], st["counts"]["jobs_running"]
    );
    Ok(())
}

pub fn ledger_verify(server: Option<&str>, json: bool) -> Result<i32> {
    let api = api(server)?;
    let out = api.get("/v1/ledger/verify")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!(
            "ledger ok: {}  entries: {}{}",
            out["ok"],
            out["entries"],
            out["problem"]
                .as_str()
                .filter(|p| !p.is_empty())
                .map(|p| format!("  problem: {p}"))
                .unwrap_or_default()
        );
    }
    Ok(if out["ok"] == true { 0 } else { 1 })
}

// ----- formatting helpers ---------------------------------------------------------

pub fn s(v: &Value) -> String {
    match v {
        Value::String(x) => x.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

pub fn num(v: &Value) -> String {
    match v {
        Value::Number(n) => n
            .as_f64()
            .map(|f| {
                if f.fract() == 0.0 {
                    format!("{f:.0}")
                } else {
                    format!("{f:.1}")
                }
            })
            .unwrap_or_default(),
        _ => String::new(),
    }
}

pub fn ago(v: &Value) -> String {
    let Some(text) = v.as_str() else {
        return "never".into();
    };
    let parsed =
        chrono::NaiveDateTime::parse_from_str(&text[..19.min(text.len())], "%Y-%m-%dT%H:%M:%S");
    match parsed {
        Ok(t) => {
            let secs = (chrono::Utc::now().naive_utc() - t).num_seconds().max(0);
            if secs < 60 {
                format!("{secs} s")
            } else if secs < 3600 {
                format!("{} min", secs / 60)
            } else if secs < 86400 {
                format!("{} h", secs / 3600)
            } else {
                format!("{} d", secs / 86400)
            }
        }
        Err(_) => text.to_string(),
    }
}

pub fn table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for r in rows {
        for (i, c) in r.iter().enumerate() {
            widths[i] = widths[i].max(c.chars().count());
        }
    }
    let line = |cells: Vec<String>| {
        cells
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{:<w$}", c, w = widths[i]))
            .collect::<Vec<_>>()
            .join("  ")
            .trim_end()
            .to_string()
            + "\n"
    };
    let mut out = line(headers.iter().map(|h| h.to_string()).collect());
    for r in rows {
        out += &line(r.clone());
    }
    out
}

pub fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .unwrap_or_else(|| "machine".into())
}

// ----- fleet control, users, audit, policy (M3) -----------------------------------------

pub fn fleet_control(server: Option<&str>, node: &str, action: &str, reason: &str) -> Result<()> {
    let api = api(server)?;
    let body = if action == "resume" {
        Value::Null
    } else {
        serde_json::json!({"reason": reason})
    };
    let out = api.post(&format!("/v1/fleet/{node}/{action}"), &body)?;
    println!(
        "{}: {action} requested (paused_by_admin={}, draining={})",
        s(&out["name"]),
        out["paused_by_admin"],
        out["draining"]
    );
    Ok(())
}

pub fn fleet_show(server: Option<&str>, node: &str, json: bool) -> Result<()> {
    let api = api(server)?;
    let m = api.get(&format!("/v1/fleet/{node}"))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&m)?);
        return Ok(());
    }
    let met = &m["metrics"];
    println!(
        "{}  {}  {}",
        s(&m["name"]),
        s(&m["status"]),
        s(&m["status_detail"])
    );
    println!(
        "owner {}  node {}…  seen {} ago",
        s(&m["owner"]),
        &s(&m["node_id"])[..16.min(s(&m["node_id"]).len())],
        ago(&m["last_seen_at"])
    );
    println!(
        "cpu {} %  ram {} %  gpu {} %  job {} round {}",
        num(&met["cpu_pct"]),
        num(&met["ram_pct"]),
        num(&met["gpu_pct"]),
        s(&m["current_job_id"]),
        s(&m["current_round"])
    );
    println!(
        "rounds served {}  samples verified {}  honesty {}",
        m["rounds_served"], m["samples_verified"], m["honesty"]
    );
    println!(
        "paused by admin {}  draining {}  tags {}",
        m["paused_by_admin"], m["draining"], m["tags"]
    );
    Ok(())
}

pub fn fleet_logs(
    server: Option<&str>,
    node: &str,
    follow: bool,
    grep: Option<&str>,
) -> Result<()> {
    let api = api(server)?;
    let mut since: i64 = 0;
    loop {
        let mut path = format!("/v1/fleet/{node}/logs?limit=200");
        if since > 0 {
            path += &format!("&since_id={since}");
        }
        if let Some(g) = grep {
            path += &format!("&grep={g}");
        }
        let lines = api.get(&path)?;
        for line in lines.as_array().into_iter().flatten() {
            since = since.max(line["id"].as_i64().unwrap_or(0));
            let at = s(&line["at"]);
            println!(
                "{} {:<7} {}",
                at.get(11..19).unwrap_or(&at),
                s(&line["level"]).to_uppercase(),
                s(&line["message"])
            );
        }
        if !follow {
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

pub fn users_list(server: Option<&str>, json: bool) -> Result<()> {
    let api = api(server)?;
    let users = api.get("/v1/users")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&users)?);
        return Ok(());
    }
    let rows: Vec<Vec<String>> = users
        .as_array()
        .into_iter()
        .flatten()
        .map(|u| {
            vec![
                s(&u["email"]),
                s(&u["name"]),
                s(&u["role"]),
                u["machines"].to_string(),
                if u["disabled"] == true {
                    "disabled".into()
                } else {
                    String::new()
                },
            ]
        })
        .collect();
    print!(
        "{}",
        table(&["email", "name", "role", "machines", ""], &rows)
    );
    Ok(())
}

pub fn users_invite(
    server: Option<&str>,
    email: Option<&str>,
    role: &str,
    days: u32,
    json: bool,
) -> Result<()> {
    let api = api(server)?;
    let out = api.post(
        "/v1/users/invite",
        &serde_json::json!({"email": email, "role": role, "expires_days": days}),
    )?;
    if json {
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!(
            "invite link ({}, expires {}):\n  {}",
            s(&out["role"]),
            &s(&out["expires_at"])[..10.min(s(&out["expires_at"]).len())],
            s(&out["url"])
        );
    }
    Ok(())
}

fn user_id_by_email(api: &Api, email: &str) -> Result<String> {
    let users = api.get("/v1/users")?;
    users
        .as_array()
        .into_iter()
        .flatten()
        .find(|u| u["email"] == email.to_lowercase())
        .map(|u| s(&u["id"]))
        .ok_or_else(|| anyhow::anyhow!("no user {email}"))
}

pub fn users_set_role(server: Option<&str>, email: &str, role: &str) -> Result<()> {
    let api = api(server)?;
    let id = user_id_by_email(&api, email)?;
    let out = api.post(
        &format!("/v1/users/{id}/role"),
        &serde_json::json!({"role": role}),
    )?;
    println!("{} is now {}", s(&out["email"]), s(&out["role"]));
    Ok(())
}

pub fn users_disable(server: Option<&str>, email: &str, disabled: bool) -> Result<()> {
    let api = api(server)?;
    let id = user_id_by_email(&api, email)?;
    let out = api.post(
        &format!("/v1/users/{id}/disable"),
        &serde_json::json!({"disabled": disabled}),
    )?;
    println!(
        "{} {}",
        s(&out["email"]),
        if out["disabled"] == true {
            "disabled"
        } else {
            "enabled"
        }
    );
    Ok(())
}

pub fn audit(server: Option<&str>, since_hours: u32, json: bool) -> Result<()> {
    let api = api(server)?;
    let rows = api.get(&format!("/v1/audit?since_hours={since_hours}&limit=200"))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    let table_rows: Vec<Vec<String>> = rows
        .as_array()
        .into_iter()
        .flatten()
        .map(|r| {
            vec![
                s(&r["at"]).chars().take(19).collect(),
                s(&r["actor"]),
                s(&r["action"]),
                s(&r["target"]),
                if r["detail"]
                    .as_object()
                    .map(|o| o.is_empty())
                    .unwrap_or(true)
                {
                    String::new()
                } else {
                    r["detail"].to_string()
                },
            ]
        })
        .collect();
    if table_rows.is_empty() {
        println!("no audit events");
    } else {
        print!(
            "{}",
            table(
                &["when (UTC)", "who", "action", "target", "detail"],
                &table_rows
            )
        );
    }
    Ok(())
}

pub fn policy_show(server: Option<&str>, json: bool) -> Result<()> {
    let api = api(server)?;
    let pol = api.get("/v1/policy")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&pol)?);
        return Ok(());
    }
    let windows: Vec<String> = pol["windows"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|w| {
            format!(
                "{} {}-{}",
                w["days"]
                    .as_array()
                    .map(|d| d.iter().map(s).collect::<Vec<_>>().join(","))
                    .unwrap_or_default(),
                s(&w["start"]),
                s(&w["end"])
            )
        })
        .collect();
    println!(
        "windows: {}\npause on battery: {}\ndrain at window end: {}",
        if windows.is_empty() {
            "always".to_string()
        } else {
            windows.join("; ")
        },
        pol["pause_on_battery"],
        pol["drain_at_window_end"]
    );
    Ok(())
}

// ----- open jobs and enrolment -----------------------------------------------------------

fn pays(j: &Value) -> String {
    if j["funding"].as_i64().unwrap_or(0) > 0 {
        format!("{}/round of {}", j["per_round"], j["funding"])
    } else if j["credits_per_1k_samples"].as_f64().unwrap_or(0.0) > 0.0 {
        format!("{}/1k samples", num(&j["credits_per_1k_samples"]))
    } else {
        "reputation".to_string()
    }
}

fn needs(req: &Value) -> String {
    let mut parts = Vec::new();
    if req["device"].as_str().unwrap_or("any") != "any" {
        parts.push(s(&req["device"]));
    }
    if req["min_vram_gb"].as_f64().unwrap_or(0.0) > 0.0 {
        parts.push(format!("{} GB VRAM", num(&req["min_vram_gb"])));
    }
    if req["min_tflops"].as_f64().unwrap_or(0.0) > 0.0 {
        parts.push(format!("{} TFLOPS", num(&req["min_tflops"])));
    }
    if req["min_honesty"].as_f64().unwrap_or(0.0) > 0.0 {
        parts.push(format!("honesty {}", num(&req["min_honesty"])));
    }
    if parts.is_empty() {
        "any machine".to_string()
    } else {
        parts.join(", ")
    }
}

pub fn enrolment_label(j: &Value) -> String {
    let mut label = s(&j["enrolment"]["mode"]);
    if j["enrolment"]["approval"] == "owner" {
        label.push_str(" + approval");
    }
    label
}

pub fn mine_label(j: &Value) -> String {
    let parts: Vec<String> = arr_of(&j["mine"])
        .iter()
        .map(|m| format!("{}: {}", s(&m["name"]), s(&m["standing"])))
        .collect();
    if parts.is_empty() {
        "-".to_string()
    } else {
        parts.join(", ")
    }
}

fn arr_of(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}

pub fn open_job_row(j: &Value) -> Vec<String> {
    vec![
        s(&j["id"]),
        s(&j["name"]),
        s(&j["owner"]),
        pays(j),
        needs(&j["requirements"]),
        enrolment_label(j),
        format!("{}/{}", j["round"], j["total_rounds"]),
        j["trainers_now"].to_string(),
        mine_label(j),
    ]
}

pub const OPEN_HEADERS: [&str; 9] = [
    "id",
    "name",
    "owner",
    "pays",
    "needs",
    "enrolment",
    "round",
    "trainers",
    "your machines",
];

pub fn job_open(server: Option<&str>, json: bool) -> Result<()> {
    let api = api(server)?;
    let jobs = api.get("/v1/jobs/open")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&jobs)?);
        return Ok(());
    }
    let rows: Vec<Vec<String>> = arr_of(&jobs).iter().map(open_job_row).collect();
    if rows.is_empty() {
        println!("no job is running");
    } else {
        print!("{}", table(&OPEN_HEADERS, &rows));
        println!("\njoin one with: plasmon trainer join <id>");
    }
    Ok(())
}

pub fn job_approvals(server: Option<&str>, id: &str, json: bool) -> Result<()> {
    let api = api(server)?;
    let rows_v = api.get(&format!("/v1/jobs/{id}/enrolments"))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&rows_v)?);
        return Ok(());
    }
    let rows: Vec<Vec<String>> = arr_of(&rows_v)
        .iter()
        .map(|e| {
            vec![
                s(&e["machine"]),
                s(&e["status"]),
                s(&e["owner"]),
                s(&e["hardware_text"]),
                if e["tflops"].is_null() {
                    "-".to_string()
                } else {
                    num(&e["tflops"])
                },
                e["honesty"]
                    .as_f64()
                    .map(|v| format!("{v:.2}"))
                    .unwrap_or_default(),
                e["rounds_served"].to_string(),
                ago(&e["requested_at"]),
                s(&e["note"]),
            ]
        })
        .collect();
    if rows.is_empty() {
        println!("no machine has asked to join this job");
        return Ok(());
    }
    print!(
        "{}",
        table(
            &[
                "machine", "status", "owner", "hardware", "tflops", "honesty", "rounds", "asked",
                "note"
            ],
            &rows
        )
    );
    let pending = arr_of(&rows_v)
        .iter()
        .filter(|e| e["status"] == "pending")
        .count();
    if pending > 0 {
        println!(
            "\n{pending} waiting: plasmon job approve {id} <machine>   or   plasmon job reject {id} <machine>"
        );
    }
    Ok(())
}

pub fn job_decide(
    server: Option<&str>,
    id: &str,
    machine: &str,
    approve: bool,
    note: &str,
    json: bool,
) -> Result<()> {
    let api = api(server)?;
    let verb = if approve { "approve" } else { "reject" };
    let out = api.post(
        &format!("/v1/jobs/{id}/enrolments/{machine}/{verb}"),
        &serde_json::json!({"note": note}),
    )?;
    if json {
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!(
            "{} {} for job {id}{}",
            s(&out["machine"]),
            s(&out["status"]),
            if approve {
                "; it takes a round at its next heartbeat"
            } else {
                ""
            }
        );
    }
    Ok(())
}

pub fn trainer_enrol(
    server: Option<&str>,
    job: &str,
    machine: Option<&str>,
    join: bool,
    json: bool,
) -> Result<()> {
    let api = api(server)?;
    let node = match machine {
        Some(m) => Some(m.to_string()),
        None => {
            let key = paths::machine_key();
            key.exists()
                .then(|| plasmon_core::identity::Identity::load(&key).map(|i| i.node_id()))
                .transpose()?
        }
    };
    let verb = if join { "join" } else { "leave" };
    let out = api.post(
        &format!("/v1/jobs/{job}/{verb}"),
        &serde_json::json!({"node_id": node}),
    )?;
    if json {
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }
    let machine_name = s(&out["machine"]);
    match out["status"].as_str().unwrap_or("") {
        "approved" => println!("{machine_name} joined job {job}; it takes a round at its next heartbeat"),
        "pending" => println!("{machine_name} asked to join job {job}; the owner decides and you get a mail either way"),
        "left" => println!("{machine_name} left job {job}; it finishes its current round and takes no more"),
        other => println!("{machine_name}: {other}"),
    }
    Ok(())
}

// ----- credits (M6) ------------------------------------------------------------------

pub fn credits_me(server: Option<&str>, json: bool) -> Result<()> {
    let api = api(server)?;
    let me = api.get("/v1/credits/me?limit=30")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&me)?);
        return Ok(());
    }
    if me["enabled"] != true {
        println!("credits are off on this server");
        return Ok(());
    }
    println!("balance: {} {}s", me["balance"], s(&me["unit"]));
    let rows: Vec<Vec<String>> = me["entries"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|e| {
            vec![
                s(&e["at"]).chars().take(19).collect(),
                s(&e["kind"]),
                e["amount"].to_string(),
                s(&e["job_id"]),
                s(&e["memo"]),
            ]
        })
        .collect();
    if !rows.is_empty() {
        print!(
            "{}",
            table(&["when (UTC)", "kind", "amount", "job", "memo"], &rows)
        );
    }
    Ok(())
}

pub fn credits_grant(server: Option<&str>, email: &str, amount: i64, memo: &str) -> Result<()> {
    let api = api(server)?;
    let out = api.post(
        "/v1/credits/grant",
        &serde_json::json!({"email": email, "amount": amount, "memo": memo}),
    )?;
    println!(
        "granted {} to {}; balance now {}",
        out["granted"],
        s(&out["email"]),
        out["balance"]
    );
    Ok(())
}

pub fn credits_users(server: Option<&str>, json: bool) -> Result<()> {
    let api = api(server)?;
    let rows_v = api.get("/v1/credits/users")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&rows_v)?);
        return Ok(());
    }
    let rows: Vec<Vec<String>> = rows_v
        .as_array()
        .into_iter()
        .flatten()
        .map(|r| vec![s(&r["email"]), s(&r["role"]), r["balance"].to_string()])
        .collect();
    print!("{}", table(&["user", "role", "balance"], &rows));
    Ok(())
}
