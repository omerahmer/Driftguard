//! driftguard — the CLI binary.
//!
//! Thin layer: parse args, call into driftguard-core, format output (human or
//! `--json`), set the exit code. No business logic lives here.

use std::io::{IsTerminal, Read};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use driftguard_core::db::{self, Pool};
use driftguard_core::embed::VoyageEmbedder;
use driftguard_core::llm::GoLlm;
use driftguard_core::prompts::{self, CreateVersionOutcome};
use driftguard_core::segment::{DiffOpType, PromptDiff, compute_diff};
use driftguard_core::{evals, fixtures, scoring, selection, validation};

/// `#[derive(Parser)]` generates the whole argument parser for this struct:
/// flag/positional parsing, `--help`/`--version`, error messages, and exit
/// codes. We use derive (over the builder API) because the command tree is
/// static and the structs *are* the spec — easier to read and to defend than an
/// imperative builder.
#[derive(Parser)]
#[command(
    name = "driftguard",
    version,
    about = "Version LLM prompts and predict which evals a change affects."
)]
struct Cli {
    /// Emit machine-readable JSON instead of human-readable text.
    ///
    /// `global = true` makes this a single root-level flag that is still
    /// accepted after any subcommand (`driftguard prompt list --json`), instead
    /// of being redeclared on every subcommand.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Manage prompts and their versions.
    Prompt {
        #[command(subcommand)]
        action: PromptCmd,
    },
    /// Manage eval cases.
    Eval {
        #[command(subcommand)]
        action: EvalCmd,
    },
    /// Predict which eval cases a prompt version's change likely affects.
    SelectEvals(SelectEvalsArgs),
    /// Manage the fixtures file (Phase 4).
    Fixtures {
        #[command(subcommand)]
        action: FixturesCmd,
    },
    /// Run all eval cases for a prompt version through the model under test.
    RunEvals(RunEvalsArgs),
    /// Run the full validation pipeline over a fixtures file (run-evals + judge
    /// + ground truth) — produces the data `score` reports on.
    Validate(ValidateArgs),
    /// Human spot-check of judge verdicts to report a judge-agreement rate.
    JudgeValidate(JudgeValidateArgs),
    /// Report precision/recall/% reduction across a similarity-threshold sweep.
    Score(ScoreArgs),
}

#[derive(Subcommand)]
enum FixturesCmd {
    /// Load prompts, eval cases, and edits from a fixtures TOML file.
    Load(FixturesLoadArgs),
}

#[derive(Args)]
struct FixturesLoadArgs {
    #[arg(long = "file")]
    file: String,
    /// Delete and recreate prompts that already exist (clean reload).
    #[arg(long)]
    force: bool,
}

#[derive(Args)]
struct RunEvalsArgs {
    #[arg(long = "prompt")]
    prompt: String,
    #[arg(long, default_value = "latest")]
    version: String,
}

#[derive(Args)]
struct ValidateArgs {
    #[arg(long = "fixtures")]
    fixtures: String,
    /// Selection threshold used while labeling (the sweep in `score` is separate).
    #[arg(long, default_value_t = 0.5)]
    threshold: f64,
}

#[derive(Args)]
struct JudgeValidateArgs {
    #[arg(long = "sample-size", default_value_t = 10)]
    sample_size: i64,
}

#[derive(Args)]
struct ScoreArgs {
    /// Optionally restrict scoring to a single prompt.
    #[arg(long = "prompt")]
    prompt: Option<String>,
}

#[derive(Subcommand)]
enum EvalCmd {
    /// Add an eval case to a prompt.
    Add(EvalAddArgs),
    /// List a prompt's eval cases.
    List(EvalListArgs),
}

#[derive(Args)]
struct EvalAddArgs {
    #[arg(long = "prompt")]
    prompt: String,
    /// The test input given to the model.
    #[arg(long)]
    input: String,
    /// Natural-language description of what a correct response must do.
    #[arg(long = "expected")]
    expected_behavior: String,
}

#[derive(Args)]
struct EvalListArgs {
    #[arg(long = "prompt")]
    prompt: String,
}

#[derive(Args)]
struct SelectEvalsArgs {
    #[arg(long = "prompt")]
    prompt: String,
    /// Version ref to select for (latest, parent, vN, or an id).
    #[arg(long, default_value = "latest")]
    version: String,
    /// Cosine-similarity cutoff in [0,1]; cases at or above it are selected.
    #[arg(long, default_value_t = 0.5)]
    threshold: f64,
}

#[derive(Subcommand)]
enum PromptCmd {
    /// Create a new (empty) prompt; the first `version` becomes v1.
    Create(CreateArgs),
    /// Add a new version of a prompt, recording its diff from the parent.
    Version(VersionArgs),
    /// Show the diff between two versions of a prompt.
    Diff(DiffArgs),
    /// List all prompts.
    List,
    /// Show the version history of a prompt.
    History(HistoryArgs),
}

#[derive(Args)]
struct CreateArgs {
    #[arg(long)]
    name: String,
}

#[derive(Args)]
struct VersionArgs {
    #[arg(long = "prompt")]
    prompt: String,
    /// Content source: a file path, `-` for stdin, or omit to read stdin.
    #[arg(long)]
    content: Option<String>,
}

#[derive(Args)]
struct DiffArgs {
    #[arg(long = "prompt")]
    prompt: String,
    /// Source version ref (latest, parent, vN, or an id).
    #[arg(long)]
    from: String,
    /// Target version ref (latest, parent, vN, or an id).
    #[arg(long)]
    to: String,
}

#[derive(Args)]
struct HistoryArgs {
    #[arg(long = "prompt")]
    prompt: String,
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();
    let json = cli.json;

    if let Err(err) = run(cli).await {
        if json {
            // Machine consumers (CI, the dashboard) get a structured error too.
            let payload = serde_json::json!({ "error": err.to_string() });
            println!("{}", serde_json::to_string_pretty(&payload).unwrap());
        } else {
            // `{:#}` prints the full anyhow context chain.
            eprintln!("Error: {err:#}");
        }
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<()> {
    let database_url = std::env::var("DATABASE_URL")
        .context("DATABASE_URL is not set (copy .env.example to .env)")?;
    let pool = db::connect(&database_url)
        .await
        .context("connecting to Postgres")?;

    match cli.command {
        Command::Prompt { action } => run_prompt(&pool, action, cli.json).await,
        Command::Eval { action } => run_eval(&pool, action, cli.json).await,
        Command::SelectEvals(args) => run_select_evals(&pool, args, cli.json).await,
        Command::Fixtures { action } => run_fixtures(&pool, action, cli.json).await,
        Command::RunEvals(args) => run_run_evals(&pool, args, cli.json).await,
        Command::Validate(args) => run_validate(&pool, args, cli.json).await,
        Command::JudgeValidate(args) => run_judge_validate(&pool, args, cli.json).await,
        Command::Score(args) => run_score(&pool, args, cli.json).await,
    }
}

async fn run_fixtures(pool: &Pool, cmd: FixturesCmd, json: bool) -> Result<()> {
    match cmd {
        FixturesCmd::Load(args) => {
            let text = std::fs::read_to_string(&args.file)
                .with_context(|| format!("reading fixtures file \"{}\"", args.file))?;
            let parsed = fixtures::parse_fixtures(&text)?;
            let summary = fixtures::load_fixtures(pool, &parsed, args.force).await?;
            if json {
                print_json(&serde_json::json!({
                    "prompts": summary.prompts,
                    "eval_cases": summary.eval_cases,
                    "versions": summary.versions,
                    "skipped_existing": summary.skipped_existing,
                }))?;
            } else {
                println!(
                    "Loaded {} prompt(s), {} eval case(s), {} version(s).{}",
                    summary.prompts,
                    summary.eval_cases,
                    summary.versions,
                    if summary.skipped_existing > 0 {
                        format!(
                            " Skipped {} already-loaded prompt(s) (use --force to reload).",
                            summary.skipped_existing
                        )
                    } else {
                        String::new()
                    }
                );
            }
        }
    }
    Ok(())
}

async fn run_run_evals(pool: &Pool, args: RunEvalsArgs, json: bool) -> Result<()> {
    let llm = GoLlm::from_env()?;
    let count = validation::run_evals(pool, &llm, &args.prompt, &args.version).await?;
    if json {
        print_json(&serde_json::json!({ "eval_runs": count }))?;
    } else {
        println!(
            "Ran {count} eval case(s) for \"{}\" {}.",
            args.prompt, args.version
        );
    }
    Ok(())
}

async fn run_validate(pool: &Pool, args: ValidateArgs, json: bool) -> Result<()> {
    if !(0.0..=1.0).contains(&args.threshold) {
        anyhow::bail!("--threshold must be in [0, 1] (got {})", args.threshold);
    }
    let text = std::fs::read_to_string(&args.fixtures)
        .with_context(|| format!("reading fixtures file \"{}\"", args.fixtures))?;
    let parsed = fixtures::parse_fixtures(&text)?;
    // Ensure fixtures are loaded (no-op if already present), then validate.
    fixtures::load_fixtures(pool, &parsed, false).await?;

    let llm = GoLlm::from_env()?;
    let embedder = VoyageEmbedder::from_env()?;
    if !json {
        eprintln!("Running validation (this makes many API calls, sequentially)…");
    }
    let summary =
        validation::validate_fixtures(pool, &llm, &embedder, &parsed, args.threshold).await?;

    if json {
        print_json(&summary)?;
    } else {
        println!(
            "Validated {} prompt(s), {} edit(s): {} eval run(s), {} behavior diff(s).\nNow run `score`.",
            summary.prompts, summary.edits, summary.eval_runs, summary.behavior_diffs
        );
    }
    Ok(())
}

async fn run_judge_validate(pool: &Pool, args: JudgeValidateArgs, json: bool) -> Result<()> {
    let samples = validation::sample_unvalidated_diffs(pool, args.sample_size).await?;
    if samples.is_empty() {
        println!("No un-reviewed judge verdicts. Run `validate` first, or all are reviewed.");
        return Ok(());
    }
    if json {
        // Non-interactive: just emit the sample for external review tooling.
        print_json(&samples)?;
        return Ok(());
    }

    println!(
        "Reviewing {} judge verdict(s). For each: [y]=agree, [n]=disagree, [s]=skip.\n",
        samples.len()
    );
    let stdin = std::io::stdin();
    for (i, s) in samples.iter().enumerate() {
        println!("── {}/{} ──────────────────────────────", i + 1, samples.len());
        println!("input:    {}", truncate(&s.input, 100));
        println!("expected: {}", truncate(&s.expected_behavior, 100));
        println!("before:   {}", truncate(&s.before_output, 140));
        println!("after:    {}", truncate(&s.after_output, 140));
        println!(
            "judge:    behavior_changed = {}",
            s.judge_behavior_changed
                .map(|b| b.to_string())
                .unwrap_or_else(|| "null".to_string())
        );
        if let Some(j) = &s.judge_justification {
            println!("          {}", truncate(j, 160));
        }
        print!("agree? [y/n/s]: ");
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break; // EOF
        }
        match line.trim().to_lowercase().as_str() {
            "y" | "yes" => validation::set_human_agreed(pool, s.behavior_diff_id, true).await?,
            "n" | "no" => validation::set_human_agreed(pool, s.behavior_diff_id, false).await?,
            _ => continue, // skip
        }
    }

    let (agreed, total) = validation::judge_agreement(pool).await?;
    if total == 0 {
        println!("\nNo verdicts reviewed yet.");
    } else {
        println!(
            "\nJudge agreement: {agreed}/{total} ({:.0}%).",
            agreed as f64 / total as f64 * 100.0
        );
    }
    Ok(())
}

async fn run_score(pool: &Pool, args: ScoreArgs, json: bool) -> Result<()> {
    let thresholds = scoring::default_thresholds();
    let report = scoring::score(pool, args.prompt.as_deref(), &thresholds).await?;

    if json {
        print_json(&report)?;
        return Ok(());
    }

    if report.samples == 0 {
        println!("No labeled selection records yet. Run `validate` first.");
        return Ok(());
    }

    println!(
        "score{} — {} labeled selection record(s)\n",
        report.prompt.as_deref().map(|p| format!(" \"{p}\"")).unwrap_or_default(),
        report.samples
    );
    println!(
        "{:>9}  {:>9}  {:>19}  {:>19}",
        "threshold", "reduction", "behavior P / R", "pass-flip P / R"
    );
    for row in &report.rows {
        println!(
            "{:>9.2}  {:>8.0}%  {:>8.2} / {:>5.2}   {:>8.2} / {:>5.2}",
            row.threshold,
            row.reduction * 100.0,
            row.behavior.precision,
            row.behavior.recall,
            row.flip.precision,
            row.flip.recall
        );
    }
    println!(
        "\nGround truth = judge 'behavior changed' (primary); 'pass-flip' shown for comparison."
    );
    Ok(())
}

async fn run_eval(pool: &Pool, cmd: EvalCmd, json: bool) -> Result<()> {
    match cmd {
        EvalCmd::Add(args) => {
            let case =
                evals::create_eval_case(pool, &args.prompt, &args.input, &args.expected_behavior)
                    .await?;
            if json {
                print_json(&case)?;
            } else {
                println!(
                    "Added eval case {} to \"{}\".",
                    short(&case.id),
                    args.prompt
                );
            }
        }
        EvalCmd::List(args) => {
            let prompt = prompts::get_prompt_by_name(pool, &args.prompt)
                .await?
                .with_context(|| format!("no prompt named \"{}\"", args.prompt))?;
            let cases = evals::list_eval_cases(pool, prompt.id).await?;
            if json {
                print_json(&cases)?;
            } else if cases.is_empty() {
                println!("No eval cases for \"{}\". Add one with `eval add`.", args.prompt);
            } else {
                for c in &cases {
                    println!("{}\t{}", short(&c.id), truncate(&c.expected_behavior, 72));
                }
            }
        }
    }
    Ok(())
}

async fn run_select_evals(pool: &Pool, args: SelectEvalsArgs, json: bool) -> Result<()> {
    if !(0.0..=1.0).contains(&args.threshold) {
        anyhow::bail!("--threshold must be in [0, 1] (got {})", args.threshold);
    }
    // Construct the embedder up front so a missing key fails fast and clearly.
    let embedder = VoyageEmbedder::from_env()?;
    let outcome =
        selection::select_evals(pool, &embedder, &args.prompt, &args.version, args.threshold)
            .await?;

    if json {
        print_json(&outcome)?;
        return Ok(());
    }

    println!(
        "select-evals \"{}\" {} (threshold {:.2})\n",
        outcome.prompt,
        short(&outcome.version_id),
        outcome.threshold
    );
    for item in &outcome.items {
        let marker = if item.was_selected { "●" } else { "○" };
        println!(
            "  {marker} {:.4}  {}  {}",
            item.similarity,
            short(&item.eval_case_id),
            truncate(&item.expected_behavior, 60)
        );
    }
    println!(
        "\nSelected {} of {} eval case(s) ({:.0}% reduction).",
        outcome.selected,
        outcome.total,
        outcome.reduction * 100.0
    );
    Ok(())
}

async fn run_prompt(pool: &Pool, cmd: PromptCmd, json: bool) -> Result<()> {
    match cmd {
        PromptCmd::Create(args) => {
            let prompt = prompts::create_prompt(pool, &args.name).await?;
            if json {
                print_json(&prompt)?;
            } else {
                println!(
                    "Created prompt \"{}\" ({}). Add its first version with `prompt version`.",
                    prompt.name,
                    short(&prompt.id)
                );
            }
        }

        PromptCmd::Version(args) => {
            let content = read_content(args.content.as_deref())?;
            let outcome = prompts::create_version(pool, &args.prompt, &content).await?;
            let ordinal = prompts::list_versions(pool, outcome.version.prompt_id)
                .await?
                .len();
            if json {
                print_json(&serde_json::json!({
                    "ordinal": ordinal,
                    "version": outcome.version,
                    "parent_version_id": outcome.parent_version_id,
                    "diff": outcome.diff,
                    "unchanged": outcome.unchanged,
                }))?;
            } else {
                print_version_human(&args.prompt, ordinal, &outcome);
            }
        }

        PromptCmd::Diff(args) => {
            let prompt = prompts::get_prompt_by_name(pool, &args.prompt)
                .await?
                .with_context(|| format!("no prompt named \"{}\"", args.prompt))?;
            let from = prompts::resolve_version_ref(pool, &prompt, &args.from).await?;
            let to = prompts::resolve_version_ref(pool, &prompt, &args.to).await?;
            // Same primitive used at version-create time, so an arbitrary-pair
            // diff is consistent with stored parent diffs.
            let diff = compute_diff(&from.content, &to.content);
            if json {
                print_json(&serde_json::json!({
                    "prompt": prompt.name,
                    "from": { "id": from.id },
                    "to": { "id": to.id },
                    "diff": diff,
                }))?;
            } else {
                println!("{}", render_diff(&diff, &args.from, &args.to));
            }
        }

        PromptCmd::List => {
            let rows = prompts::list_prompts(pool).await?;
            if json {
                print_json(&rows)?;
            } else if rows.is_empty() {
                println!("No prompts yet. Create one with `prompt create`.");
            } else {
                for p in &rows {
                    println!(
                        "{}\t{}\tcurrent={}",
                        p.name,
                        short(&p.id),
                        p.current_version_id
                            .map(|id| short(&id))
                            .unwrap_or_else(|| "—".to_string())
                    );
                }
            }
        }

        PromptCmd::History(args) => {
            let prompt = prompts::get_prompt_by_name(pool, &args.prompt)
                .await?
                .with_context(|| format!("no prompt named \"{}\"", args.prompt))?;
            let versions = prompts::list_versions(pool, prompt.id).await?;
            if json {
                print_json(&versions)?;
            } else {
                for (i, v) in versions.iter().enumerate() {
                    let marker = if Some(v.id) == prompt.current_version_id {
                        "*"
                    } else {
                        " "
                    };
                    println!(
                        "{marker} v{}\t{}\t{}\t{}",
                        i + 1,
                        short(&v.id),
                        v.created_at.to_rfc3339(),
                        summarize_stored_diff(v.diff_from_parent.as_deref()),
                    );
                }
            }
        }
    }
    Ok(())
}

/// Read prompt content from a file path, `-` (stdin), or — when omitted — stdin.
fn read_content(arg: Option<&str>) -> Result<String> {
    match arg {
        Some("-") | None => {
            if std::io::stdin().is_terminal() {
                anyhow::bail!(
                    "no content: pass --content <file>, --content -, or pipe input via stdin"
                );
            }
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .context("reading content from stdin")?;
            if buf.trim().is_empty() {
                anyhow::bail!("stdin was empty");
            }
            Ok(buf)
        }
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("reading content file \"{path}\"")),
    }
}

fn print_version_human(prompt: &str, ordinal: usize, outcome: &CreateVersionOutcome) {
    println!(
        "Created v{ordinal} of \"{prompt}\" ({}).",
        short(&outcome.version.id)
    );
    if outcome.unchanged {
        println!("⚠️  Content is identical to the parent version — empty changed span.");
    } else if let Some(diff) = &outcome.diff {
        println!(
            "   Diff from parent: +{} −{} ({} unchanged).",
            diff.stats.added, diff.stats.removed, diff.stats.unchanged
        );
    } else {
        println!("   First version — no parent to diff against.");
    }
}

/// Render a `PromptDiff` as a unified-ish, human-readable block.
fn render_diff(diff: &PromptDiff, from: &str, to: &str) -> String {
    let mut out = format!(
        "diff {from} → {to}  (+{} −{}, {} unchanged)\n\n",
        diff.stats.added, diff.stats.removed, diff.stats.unchanged
    );
    for op in &diff.ops {
        let prefix = match op.kind {
            DiffOpType::Equal => "  ",
            DiffOpType::Added => "+ ",
            DiffOpType::Removed => "- ",
        };
        out.push_str(prefix);
        out.push_str(&op.text);
        out.push('\n');
    }
    out
}

fn summarize_stored_diff(raw: Option<&str>) -> String {
    match raw {
        None => "(first version)".to_string(),
        Some(json) => match serde_json::from_str::<PromptDiff>(json) {
            Ok(d) => format!("+{} −{}", d.stats.added, d.stats.removed),
            Err(_) => "(unparseable diff)".to_string(),
        },
    }
}

fn print_json<T: serde::Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

/// First 8 chars of an id, for compact human output.
fn short<T: std::fmt::Display>(id: &T) -> String {
    id.to_string().chars().take(8).collect()
}

/// Truncate to `max` chars with an ellipsis, for compact table cells.
fn truncate(text: &str, max: usize) -> String {
    let flat = text.replace('\n', " ");
    if flat.chars().count() <= max {
        flat
    } else {
        let head: String = flat.chars().take(max.saturating_sub(1)).collect();
        format!("{head}…")
    }
}
