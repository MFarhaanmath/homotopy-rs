//! homotopy-agent: headless JSON CLI for RL agent training against homotopy.io.
//!
//! Sibling binary to `homotopy-cli`. Reuses the same `homotopy-model` API
//! but speaks JSON over stdout, and exposes an action-enumeration command
//! for RL agents to select from.

use std::{
    fs::{read, write},
    path::PathBuf,
};

use anyhow::{anyhow, Context, Result};
use homotopy_core::common::{Boundary, Direction, Height, SliceIndex};
use homotopy_core::signature::Signature;
use homotopy_model::{
    history::Proof,
    migration,
    proof::{homotopy, Action},
    serialize,
};
use serde::Serialize;
use structopt::StructOpt;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, StructOpt)]
#[structopt(
    name = "homotopy-agent",
    about = "Headless JSON interface for RL agent training against homotopy.io."
)]
struct Opt {
    /// Input .hom state file. If omitted, starts from empty state.
    #[structopt(short, long, parse(from_os_str))]
    input: Option<PathBuf>,

    /// Output .hom state file. Written after the command completes.
    #[structopt(short, long, parse(from_os_str))]
    output: Option<PathBuf>,

    #[structopt(subcommand)]
    command: Command,
}

#[derive(Debug, StructOpt)]
enum Command {
    /// List all valid actions from the current state, as JSON.
    Actions,

    /// Apply the action at the given index from the last `actions` call.
    Apply {
        #[structopt(short, long)]
        index: usize,
    },

    /// Print the current state as JSON.
    State,

    /// Replay a sequence of actions from a JSON file.
    Replay {
        #[structopt(parse(from_os_str))]
        file: PathBuf,

        /// Don't crash if an action fails; report the failure index instead.
        #[structopt(long)]
        no_crash: bool,
    },
}

// ---------------------------------------------------------------------------
// I/O helpers (mirrors homotopy-cli/src/main.rs)
// ---------------------------------------------------------------------------

fn import_hom(path: &PathBuf) -> Result<Proof> {
    let data = read(path)?;
    let ((signature, workspace), metadata) = match serialize::deserialize(&data) {
        Some(res) => res,
        None => migration::deserialize(&data)
            .context("Failed to deserialize or migrate from legacy format.")?,
    };

    let mut proof: Proof = Default::default();
    proof.signature = signature;
    proof.workspace = workspace;
    proof.metadata = metadata;
    Ok(proof)
}

fn export_hom(path: &PathBuf, proof: &Proof) -> Result<()> {
    let data = serialize::serialize(
        proof.signature.clone(),
        proof.workspace.clone(),
        proof.metadata.clone(),
    );
    write(path, data).context("Could not write .hom file.")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Action enumeration — the core of the RL interface
// ---------------------------------------------------------------------------

/// Enumerate every valid action from the current proof state.
///
/// This is the finite branching factor the RL agent selects from.
/// We generate *candidates* and let `Action::is_valid` filter them,
/// which is exactly how the engine itself decides availability.
///
/// TODO: add `Action::Attach` once `AttachOption` shape is known.
fn enumerate_actions(proof: &Proof) -> Vec<Action> {
    let mut out: Vec<Action> = Vec::new();

    // --- Always available ---
    out.push(Action::CreateGeneratorZero);

    // --- One SelectGenerator per generator in the signature ---
    for generator in proof.signature.generators() {
        out.push(Action::SelectGenerator(generator));
    }

    // --- Signature-level operations ---
    out.push(Action::SuspendSignature);

    // --- Workspace-dependent moves ---
    if let Some(ws) = &proof.workspace {
        out.push(Action::TakeIdentityDiagram);
        out.push(Action::ClearWorkspace);

        // Ascend: from depth 1 up to current path length
        let depth = ws.path.len();
        for n in 1..=depth {
            out.push(Action::AscendSlice(n));
        }

        // Set boundary: Source or Target
        out.push(Action::SetBoundary(Boundary::Source));
        out.push(Action::SetBoundary(Boundary::Target));

        // Boundary-related operations
        out.push(Action::ClearBoundary);
        out.push(Action::FlipBoundary);
        out.push(Action::RecoverBoundary);

        // Switch between adjacent slices (both directions)
        out.push(Action::SwitchSlice(Direction::Forward));
        out.push(Action::SwitchSlice(Direction::Backward));

        // Descend + homotopy enumeration over the visible diagram.
        let visible = ws.visible_diagram();
        if let Some(size) = visible.size() {
            for slice in SliceIndex::for_size(size) {
                out.push(Action::DescendSlice(slice));
            }

            for i in 0..size {
                for direction in [Direction::Forward, Direction::Backward] {
                    // Contract at singular height i.
                    // SingularHeight is `pub type SingularHeight = usize`,
                    // so we pass `i` directly.
                    out.push(Action::Homotopy(homotopy::Homotopy::Contract(
                        homotopy::Contract {
                            height: i,
                            direction,
                            step: 0,
                            bias: None,
                            location: vec![],
                        },
                    )));

                    // Expand at the singular point i.
                    out.push(Action::Homotopy(homotopy::Homotopy::Expand(
                        homotopy::Expand {
                            point: [Height::Singular(i), Height::Singular(i)],
                            direction,
                            location: vec![],
                        },
                    )));
                }
            }
        }

        // Higher-level structural operations
        out.push(Action::Theorem);
        out.push(Action::Squash);
        out.push(Action::Behead);
        out.push(Action::Befoot);
        out.push(Action::Invert);
        out.push(Action::Restrict);
    }

    // --- Stash ---
    out.push(Action::Stash);
    out.push(Action::StashDrop);
    out.push(Action::StashPop);
    out.push(Action::StashApply);

    // --- Filter by validity (the engine is the source of truth) ---
    out.retain(|a| a.is_valid(proof));
    out
}

// ---------------------------------------------------------------------------
// JSON response shapes
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ActionDescriptor {
    id: usize,
    action: serde_json::Value,
}

#[derive(Serialize)]
struct ActionsResponse {
    count: usize,
    actions: Vec<ActionDescriptor>,
}

#[derive(Serialize)]
struct StateResponse {
    signature_size: usize,
    workspace_dimension: Option<usize>,
    visible_dimension: Option<usize>,
    has_boundary: bool,
    stash_size: usize,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let opt = Opt::from_args();

    let mut proof = match &opt.input {
        Some(path) => import_hom(path).context("Could not import .hom file.")?,
        None => Default::default(),
    };

    match opt.command {
        Command::Actions => {
            let actions = enumerate_actions(&proof);
            let descriptors: Vec<ActionDescriptor> = actions
                .iter()
                .enumerate()
                .map(|(id, a)| ActionDescriptor {
                    id,
                    action: serde_json::to_value(a).unwrap_or(serde_json::Value::Null),
                })
                .collect();
            let resp = ActionsResponse {
                count: descriptors.len(),
                actions: descriptors,
            };
            println!("{}", serde_json::to_string_pretty(&resp)?);
        }

        Command::Apply { index } => {
            let actions = enumerate_actions(&proof);
            let action = actions
                .get(index)
                .ok_or_else(|| {
                    anyhow!(
                        "action index {} out of range (have {} actions)",
                        index,
                        actions.len()
                    )
                })?
                .clone();
            let changed = proof.update(&action)?;
            if let Some(path) = &opt.output {
                export_hom(path, &proof)?;
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "ok",
                    "changed": changed,
                    "action": serde_json::to_value(&action)?,
                }))?
            );
        }

        Command::State => {
            let (workspace_dimension, visible_dimension) = match &proof.workspace {
                Some(ws) => (
                    Some(ws.diagram.dimension()),
                    Some(ws.visible_dimension()),
                ),
                None => (None, None),
            };
            let resp = StateResponse {
                signature_size: proof.signature.iter().count(),
                workspace_dimension,
                visible_dimension,
                has_boundary: proof.boundary.is_some(),
                stash_size: proof.stash.len(),
            };
            println!("{}", serde_json::to_string_pretty(&resp)?);
        }

        Command::Replay { file, no_crash } => {
            let data = read(&file)?;
            let actions: Vec<Action> = serde_json::from_slice(&data)
                .context("Failed to parse actions JSON file.")?;
            let total = actions.len();
            let mut completed = 0;
            let mut no_ops = 0;
            let mut earliest_failure = None;
            for (i, action) in actions.iter().enumerate() {
                match proof.update(action) {
                    Ok(true) => completed += 1,
                    Ok(false) => {
                        completed += 1;
                        no_ops += 1;
                    }
                    Err(e) => {
                        earliest_failure = Some(i);
                        if !no_crash {
                            return Err(anyhow!("action {} failed: {}", i, e));
                        }
                        break;
                    }
                }
            }
            if let Some(path) = &opt.output {
                export_hom(path, &proof)?;
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "steps_completed": completed,
                    "no_ops": no_ops,
                    "total_steps": total,
                    "earliest_failure": earliest_failure,
                }))?
            );
        }
    }

    Ok(())
}