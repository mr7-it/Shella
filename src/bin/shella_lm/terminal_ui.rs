use shella::{TrainingPhase, TrainingProgress};
use std::{
    env,
    io::{self, IsTerminal, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy)]
pub struct TerminalUi {
    interactive: bool,
    color: bool,
}

impl TerminalUi {
    #[must_use]
    pub fn detect() -> Self {
        let interactive =
            io::stderr().is_terminal() && !env::var("TERM").is_ok_and(|term| term == "dumb");
        let color = interactive && env::var_os("NO_COLOR").is_none();
        Self { interactive, color }
    }

    pub fn header(self, title: &str, subtitle: &str) {
        if !self.interactive {
            return;
        }
        let mut stderr = io::stderr().lock();
        let _ = writeln!(
            stderr,
            "{}",
            self.paint(
                "╭─ SHELLA · NEUROMORPHIC LANGUAGE ─────────────────────────╮",
                "36;1"
            )
        );
        let _ = writeln!(stderr, "{}", self.paint(&format!("│ {title:<56}│"), "1"));
        let _ = writeln!(stderr, "│ {subtitle:<56}│");
        let _ = writeln!(
            stderr,
            "{}",
            self.paint(
                "╰──────────────────────────────────────────────────────────╯",
                "36;1"
            )
        );
        let _ = stderr.flush();
    }

    pub fn info(self, message: &str) {
        if self.interactive {
            eprintln!("{} {message}", self.paint("›", "36;1"));
        } else {
            eprintln!("[info] {message}");
        }
    }

    pub fn error(self, message: &str) {
        if self.interactive {
            eprintln!("{} {message}", self.paint("✗", "31;1"));
        } else {
            eprintln!("[error] {message}");
        }
    }

    pub fn run_stage<T, E>(
        self,
        label: &str,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let spinner = StageSpinner::start(self, label);
        let result = operation();
        spinner.finish(result.is_ok());
        result
    }

    fn paint(self, text: &str, style: &str) -> String {
        if self.color {
            format!("\x1b[{style}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    }
}

struct StageSpinner {
    ui: TerminalUi,
    label: String,
    started: Instant,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    finished: bool,
}

impl StageSpinner {
    fn start(ui: TerminalUi, label: &str) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let worker = if ui.interactive {
            let thread_stop = Arc::clone(&stop);
            let thread_label = label.to_owned();
            let color = ui.color;
            Some(thread::spawn(move || {
                const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                let started = Instant::now();
                let mut frame = 0usize;
                while !thread_stop.load(Ordering::Acquire) {
                    let glyph = FRAMES[frame % FRAMES.len()];
                    let elapsed = format_duration(started.elapsed());
                    let prefix = if color {
                        format!("\x1b[35;1m{glyph}\x1b[0m")
                    } else {
                        glyph.to_owned()
                    };
                    let mut stderr = io::stderr().lock();
                    let _ = write!(stderr, "\r\x1b[2K{prefix} {thread_label} · {elapsed}");
                    let _ = stderr.flush();
                    frame += 1;
                    drop(stderr);
                    thread::sleep(Duration::from_millis(90));
                }
            }))
        } else {
            eprintln!("[....] {label}");
            None
        };
        Self {
            ui,
            label: label.to_owned(),
            started: Instant::now(),
            stop,
            worker,
            finished: false,
        }
    }

    fn finish(mut self, success: bool) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        let elapsed = format_duration(self.started.elapsed());
        let mut stderr = io::stderr().lock();
        if self.ui.interactive {
            let (symbol, style) = if success {
                ("✓", "32;1")
            } else {
                ("✗", "31;1")
            };
            let _ = write!(
                stderr,
                "\r\x1b[2K{} {} · {elapsed}\n",
                self.ui.paint(symbol, style),
                self.label
            );
        } else {
            let state = if success { "ok" } else { "gagal" };
            let _ = writeln!(stderr, "[{state}] {} · {elapsed}", self.label);
        }
        let _ = stderr.flush();
        self.finished = true;
    }
}

impl Drop for StageSpinner {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        if self.ui.interactive {
            let _ = write!(io::stderr(), "\r\x1b[2K");
        }
    }
}

pub struct TrainingProgressUi {
    ui: TerminalUi,
    started: Instant,
    last_draw: Instant,
    last_logged_percent: usize,
    spinner_frame: usize,
    epoch_started: Option<Instant>,
    active_phase: Option<TrainingPhase>,
}

impl TrainingProgressUi {
    #[must_use]
    pub fn new(ui: TerminalUi) -> Self {
        Self {
            ui,
            started: Instant::now(),
            last_draw: Instant::now() - Duration::from_secs(1),
            last_logged_percent: usize::MAX,
            spinner_frame: 0,
            epoch_started: None,
            active_phase: None,
        }
    }

    pub fn update(&mut self, progress: TrainingProgress) {
        let now = Instant::now();
        let phase_changed = self.active_phase != Some(progress.phase);
        if progress.phase == TrainingPhase::Optimizing && (phase_changed || progress.completed == 0)
        {
            self.epoch_started = Some(now);
        }
        let percent = percentage(progress.completed, progress.total);
        let done = progress.total > 0 && progress.completed >= progress.total;
        if self.ui.interactive {
            if !phase_changed
                && !done
                && now.duration_since(self.last_draw) < Duration::from_millis(100)
            {
                return;
            }
            self.draw_interactive(progress, percent, done, now);
        } else {
            let should_log =
                phase_changed || done || percent >= self.last_logged_percent.saturating_add(5);
            if !should_log {
                return;
            }
            let mut stderr = io::stderr().lock();
            let _ = writeln!(
                stderr,
                "[progress] {} epoch={}/{} {}/{} ({percent}%) loss={} elapsed={}",
                phase_label(progress.phase),
                progress.epoch,
                progress.total_epochs,
                progress.completed,
                progress.total,
                format_loss(progress.latest_loss),
                format_duration(now.duration_since(self.started))
            );
            let _ = stderr.flush();
            self.last_logged_percent = percent;
        }
        self.active_phase = Some(progress.phase);
        self.last_draw = now;
    }

    pub fn finish(&mut self) {
        if self.ui.interactive && self.active_phase.is_some() {
            let _ = writeln!(io::stderr());
            self.active_phase = None;
        }
    }

    fn draw_interactive(
        &mut self,
        progress: TrainingProgress,
        percent: usize,
        done: bool,
        now: Instant,
    ) {
        const FRAMES: [&str; 4] = ["◐", "◓", "◑", "◒"];
        let frame = FRAMES[self.spinner_frame % FRAMES.len()];
        self.spinner_frame += 1;
        let bar = progress_bar(progress.completed, progress.total, 20);
        let status = match progress.phase {
            TrainingPhase::PreparingData => format!(
                "siapkan data  {bar} {percent:>3}%  dokumen {}/{}",
                progress.completed, progress.total
            ),
            TrainingPhase::Optimizing => {
                let elapsed = self
                    .epoch_started
                    .map_or(Duration::ZERO, |started| now.duration_since(started));
                let eta = estimate_remaining(progress, elapsed);
                format!(
                    "latih {}/{}  {bar} {percent:>3}%  step {}/{}  batch-loss {}  ETA {}",
                    progress.epoch,
                    progress.total_epochs,
                    progress.completed,
                    progress.total,
                    format_loss(progress.latest_loss),
                    eta
                )
            }
            TrainingPhase::Validating => format!(
                "validasi {}/{}  {bar} {percent:>3}%  batch {}/{}  val-loss {}",
                progress.epoch,
                progress.total_epochs,
                progress.completed,
                progress.total,
                format_loss(progress.latest_loss)
            ),
        };
        let symbol = if done {
            self.ui.paint("✓", "32;1")
        } else {
            self.ui.paint(frame, "35;1")
        };
        let mut stderr = io::stderr().lock();
        let _ = write!(stderr, "\r\x1b[2K{symbol} {status}");
        let _ = stderr.flush();
    }
}

impl Drop for TrainingProgressUi {
    fn drop(&mut self) {
        if self.ui.interactive && self.active_phase.is_some() {
            let _ = writeln!(io::stderr());
        }
    }
}

fn percentage(completed: usize, total: usize) -> usize {
    if total == 0 {
        0
    } else {
        completed
            .saturating_mul(100)
            .checked_div(total)
            .unwrap_or(100)
            .min(100)
    }
}

fn progress_bar(completed: usize, total: usize, width: usize) -> String {
    let filled = if total == 0 {
        0
    } else {
        completed
            .saturating_mul(width)
            .checked_div(total)
            .unwrap_or(width)
            .min(width)
    };
    format!("[{}{}]", "█".repeat(filled), "·".repeat(width - filled))
}

fn phase_label(phase: TrainingPhase) -> &'static str {
    match phase {
        TrainingPhase::PreparingData => "persiapan",
        TrainingPhase::Optimizing => "optimizer",
        TrainingPhase::Validating => "validasi",
    }
}

fn format_loss(loss: Option<f64>) -> String {
    loss.map_or_else(|| "—".to_owned(), |value| format!("{value:.4}"))
}

fn estimate_remaining(progress: TrainingProgress, elapsed: Duration) -> String {
    if progress.completed < 5 || progress.total == 0 || elapsed.is_zero() {
        return "menghitung…".to_owned();
    }
    let steps_per_second = progress.completed as f64 / elapsed.as_secs_f64();
    if !steps_per_second.is_finite() || steps_per_second <= 0.0 {
        return "—".to_owned();
    }
    let later_epochs = progress.total_epochs.saturating_sub(progress.epoch);
    let remaining_steps = progress
        .total
        .saturating_sub(progress.completed)
        .saturating_add(later_epochs.saturating_mul(progress.total));
    let seconds = (remaining_steps as f64 / steps_per_second).ceil();
    let seconds = if seconds >= u64::MAX as f64 {
        u64::MAX
    } else {
        seconds as u64
    };
    format_duration(Duration::from_secs(seconds))
}

fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    if hours > 0 {
        format!("{hours}j {minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m {seconds:02}dt")
    } else {
        format!("{seconds}dt")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_math_handles_empty_and_completed_work() {
        assert_eq!(percentage(0, 0), 0);
        assert_eq!(percentage(1, 3), 33);
        assert_eq!(percentage(3, 3), 100);
        assert_eq!(progress_bar(2, 4, 4), "[██··]");
    }

    #[test]
    fn eta_only_appears_after_real_steps_have_completed() {
        let mut progress = TrainingProgress {
            phase: TrainingPhase::Optimizing,
            epoch: 1,
            total_epochs: 1,
            completed: 0,
            total: 10,
            latest_loss: None,
        };
        assert_eq!(estimate_remaining(progress, Duration::ZERO), "menghitung…");
        progress.completed = 5;
        assert_eq!(
            estimate_remaining(progress, Duration::from_secs(10)),
            "10dt"
        );
    }
}
