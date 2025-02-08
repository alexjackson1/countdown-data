use clap::{Parser, ValueEnum};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::Itertools;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use rayon::prelude::*;
use zstd::Encoder;

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use countdown::{find_closest_solution, generate_numbers, generate_target, GameInstance};

const CHUNK_SIZE: usize = 250;
const CHANNEL_BUFFER: usize = 200;

const SMALL_NUMBERS: [u64; 20] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10];
const LARGE_NUMBERS: [u64; 4] = [25, 50, 75, 100];

//
// Filtering functions
//

fn perfect_filter(instance: &GameInstance) -> bool {
    instance.closest == instance.target
}

fn countdown_score(instance: &GameInstance) -> u64 {
    let t = instance.target as i32;
    let c = instance.closest as i32;
    (10 - (t - c).abs()).max(0) as u64
}

fn positive_score(instance: &GameInstance) -> bool {
    countdown_score(instance) >= 1
}

//
// Generators
//

/// Generates a chunk of games (of size `size`) that pass the given filter.
fn generate_chunk_with_filter<F>(num_count: usize, size: usize, filter: F) -> Vec<GameInstance>
where
    F: Fn(&GameInstance) -> bool + Sync,
{
    let mut results = Vec::with_capacity(size);
    // Keep generating until we have enough instances that satisfy the filter.
    while results.len() < size {
        let mut chunk: Vec<GameInstance> = (0..CHUNK_SIZE)
            .into_par_iter()
            .map(|_| {
                let starting = generate_numbers(num_count);
                let target = generate_target();
                let (closest, expression) = find_closest_solution(&starting, target);
                GameInstance::new(starting, target, closest, expression)
            })
            .filter(|instance| filter(instance))
            .collect();
        results.append(&mut chunk);
    }
    results.truncate(size);
    results
}

/// Exhaustively generate all game instances given a domain and target range.
fn generate_exhaustive_instances(
    num_count: usize,
    domain: &[u64],
    target_range: std::ops::RangeInclusive<u64>,
) -> Vec<GameInstance> {
    let mut instances = Vec::new();
    for combination in domain.iter().copied().combinations(num_count) {
        for target in target_range.clone() {
            let (closest, expression) = find_closest_solution(&combination, target);
            instances.push(GameInstance::new(
                combination.clone(),
                target,
                closest,
                expression,
            ));
        }
    }
    instances
}

/// Generate Countdown numbers by sampling from the small and large pools.
fn generate_countdown_numbers(num_count: usize) -> Vec<u64> {
    let mut rng = rand::rng();
    let max_large = num_count.min(LARGE_NUMBERS.len());
    let large_count = rng.random_range(0..=max_large);
    let small_count = num_count - large_count;

    let small: Vec<u64> = SMALL_NUMBERS
        .choose_multiple(&mut rng, small_count)
        .cloned()
        .collect();
    let large: Vec<u64> = LARGE_NUMBERS
        .choose_multiple(&mut rng, large_count)
        .cloned()
        .collect();

    let mut chosen: Vec<u64> = small.into_iter().chain(large.into_iter()).collect();
    chosen.shuffle(&mut rng);
    chosen
}

/// Generate a chunk of Countdown game instances.
fn generate_countdown_chunk(size: usize, num_count: usize) -> Vec<GameInstance> {
    (0..size)
        .into_par_iter()
        .map(|_| {
            let starting = generate_countdown_numbers(num_count);
            let target = generate_target();
            let (closest, expression) = find_closest_solution(&starting, target);
            GameInstance::new(starting, target, closest, expression)
        })
        .collect()
}

/// Exhaustively generate Countdown instances.
fn generate_exhaustive_countdown_instances(
    num_count: usize,
    target_range: std::ops::RangeInclusive<u64>,
) -> Vec<GameInstance> {
    let mut pool = Vec::new();
    pool.extend_from_slice(&SMALL_NUMBERS);
    pool.extend_from_slice(&LARGE_NUMBERS);

    let combinations = pool.into_iter().combinations(num_count).collect::<Vec<_>>();
    let mut instances = Vec::new();
    for nums in combinations {
        for target in target_range.clone() {
            let (closest, expression) = find_closest_solution(&nums, target);
            instances.push(GameInstance::new(nums.clone(), target, closest, expression));
        }
    }
    instances
}

//
// Command-line definitions
//

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Mode {
    Random,
    RandomPerfect,
    RandomNear,
    Exhaustive,
    CountdownRandom,
    CountdownExhaustive,
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Mode of dataset generation.
    #[arg(long, value_enum, default_value_t = Mode::Random)]
    mode: Mode,

    /// Output file name.
    #[arg(short, long, default_value = "countdown_dataset.jsonl.zst")]
    outfile: String,

    /// Number of instances to generate.
    #[arg(long, default_value = "1000")]
    num_instances: usize,

    /// Minimum number of numbers in a game (used in Random modes).
    #[arg(long, default_value = "4")]
    min_size: usize,

    /// Maximum number of numbers in a game (used in Random modes).
    #[arg(long, default_value = "6")]
    max_size: usize,

    /// Number of numbers in a Countdown game (used in Countdown modes).
    #[arg(long, default_value = "6")]
    num_count: usize,
}

/// Message type for communicating chunks to the writer thread.
enum ChunkMessage {
    Data(Vec<GameInstance>),
    Terminate,
}

/// Spawns a writer thread that listens on a channel for game instance chunks
/// and writes them to the given file with Zstd compression.
fn spawn_writer_thread(
    file: File,
    compression_level: i32,
    progress_bar: ProgressBar,
    rx: mpsc::Receiver<ChunkMessage>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        // Wrap the file in a BufWriter for efficiency.
        let buf_writer = BufWriter::new(file);
        let mut encoder =
            Encoder::new(buf_writer, compression_level).expect("Failed to create encoder");
        let mut total_written = 0;
        while let Ok(message) = rx.recv() {
            match message {
                ChunkMessage::Data(instances) => {
                    let mut buffer = String::with_capacity(CHUNK_SIZE * 100);
                    for instance in instances {
                        let json = serde_json::to_string(&instance).expect("Serialization failed");
                        buffer.push_str(&json);
                        buffer.push('\n');
                        total_written += 1;
                    }
                    if let Err(e) = encoder.write_all(buffer.as_bytes()) {
                        eprintln!("Failed to write to encoder: {}", e);
                        break;
                    }
                    progress_bar.set_position(total_written);
                }
                ChunkMessage::Terminate => break,
            }
        }
        encoder.finish().expect("Failed to finish encoding");
    })
}

//
// Helper function for processing chunks in random modes
//
fn process_random_chunks<F>(
    num_instances: usize,
    min_size: usize,
    max_size: usize,
    tx: &mpsc::SyncSender<ChunkMessage>,
    chunk_generator: F,
) where
    F: Fn(usize) -> Vec<GameInstance> + Sync,
{
    for num_count in min_size..=max_size {
        // Calculate the number of chunks needed so that total instances >= num_instances.
        let chunks_needed = (num_instances + CHUNK_SIZE - 1) / CHUNK_SIZE;
        for chunk_index in 0..chunks_needed {
            let mut instances = chunk_generator(num_count);
            // For the last chunk, truncate the vector if it exceeds the desired count.
            if chunk_index == chunks_needed - 1 {
                let total_needed = num_instances - (chunks_needed - 1) * CHUNK_SIZE;
                instances.truncate(total_needed);
            }
            tx.send(ChunkMessage::Data(instances))
                .expect("Failed to send chunk");
        }
    }
}

//
// Mode-specific runner functions
//

fn run_random_mode(
    num_instances: usize,
    min_size: usize,
    max_size: usize,
    file: File,
) -> io::Result<()> {
    let total_instances = num_instances * (max_size - min_size + 1);
    let m = MultiProgress::new();
    m.set_draw_target(ProgressDrawTarget::stderr_with_hz(5));
    let pb = m.add(ProgressBar::new(total_instances as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {wide_msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let (tx, rx) = mpsc::sync_channel(CHANNEL_BUFFER);
    let writer_handle = spawn_writer_thread(file, 3, pb.clone(), rx);

    process_random_chunks(num_instances, min_size, max_size, &tx, |num_count| {
        (0..CHUNK_SIZE)
            .into_par_iter()
            .map(|_| {
                let numbers = generate_numbers(num_count);
                let target = generate_target();
                let (closest, expression) = find_closest_solution(&numbers, target);
                GameInstance::new(numbers, target, closest, expression)
            })
            .collect()
    });

    tx.send(ChunkMessage::Terminate)
        .expect("Failed to send termination signal");
    writer_handle.join().expect("Writer thread panicked");
    pb.finish_with_message("Completed");
    Ok(())
}

fn run_random_perfect_mode(
    num_instances: usize,
    min_size: usize,
    max_size: usize,
    file: File,
) -> io::Result<()> {
    let total_instances = num_instances * (max_size - min_size + 1);
    let m = MultiProgress::new();
    m.set_draw_target(ProgressDrawTarget::stderr_with_hz(5));
    let pb = m.add(ProgressBar::new(total_instances as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.green/white} {pos:>7}/{len:7} {wide_msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let (tx, rx) = mpsc::sync_channel(CHANNEL_BUFFER);
    let writer_handle = spawn_writer_thread(file, 3, pb.clone(), rx);

    process_random_chunks(num_instances, min_size, max_size, &tx, |num_count| {
        generate_chunk_with_filter(num_count, CHUNK_SIZE, perfect_filter)
    });

    tx.send(ChunkMessage::Terminate)
        .expect("Failed to send termination signal");
    writer_handle.join().expect("Writer thread panicked");
    pb.finish_with_message("Completed");
    Ok(())
}

fn run_random_near_mode(
    num_instances: usize,
    min_size: usize,
    max_size: usize,
    file: File,
) -> io::Result<()> {
    let total_instances = num_instances * (max_size - min_size + 1);
    let m = MultiProgress::new();
    m.set_draw_target(ProgressDrawTarget::stderr_with_hz(5));
    let pb = m.add(ProgressBar::new(total_instances as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.yellow/black} {pos:>7}/{len:7} {wide_msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let (tx, rx) = mpsc::sync_channel(CHANNEL_BUFFER);
    let writer_handle = spawn_writer_thread(file, 3, pb.clone(), rx);

    process_random_chunks(num_instances, min_size, max_size, &tx, |num_count| {
        generate_chunk_with_filter(num_count, CHUNK_SIZE, positive_score)
    });

    tx.send(ChunkMessage::Terminate)
        .expect("Failed to send termination signal");
    writer_handle.join().expect("Writer thread panicked");
    pb.finish_with_message("Completed");
    Ok(())
}

fn run_countdown_random_mode(num_instances: usize, num_count: usize, file: File) -> io::Result<()> {
    let m = MultiProgress::new();
    m.set_draw_target(ProgressDrawTarget::stderr_with_hz(5));
    let pb = m.add(ProgressBar::new(num_instances as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.magenta/black} {pos:>7}/{len:7} {wide_msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let (tx, rx) = mpsc::sync_channel(CHANNEL_BUFFER);
    let writer_handle = spawn_writer_thread(file, 3, pb.clone(), rx);

    let chunks_needed = (num_instances + CHUNK_SIZE - 1) / CHUNK_SIZE;
    for chunk_index in 0..chunks_needed {
        let mut instances = generate_countdown_chunk(CHUNK_SIZE, num_count);
        if chunk_index == chunks_needed - 1 {
            let total_needed = num_instances - (chunks_needed - 1) * CHUNK_SIZE;
            instances.truncate(total_needed);
        }
        tx.send(ChunkMessage::Data(instances))
            .expect("Failed to send chunk");
    }

    tx.send(ChunkMessage::Terminate)
        .expect("Failed to send termination signal");
    writer_handle.join().expect("Writer thread panicked");
    pb.finish_with_message("Completed");
    Ok(())
}

// For exhaustive modes we don’t use a separate writer thread.
fn run_exhaustive_mode(num_count: usize, file: File) -> io::Result<()> {
    let buf_writer = BufWriter::new(file);
    let mut encoder = Encoder::new(buf_writer, 3)?;
    let domain: Vec<u64> = (1..=100).collect();
    let target_range = 100..=999;
    println!(
        "Performing exhaustive search for {} numbers (domain: 1..=100, target: 100..=999)...",
        num_count
    );
    let instances = generate_exhaustive_instances(num_count, &domain, target_range);
    for instance in instances {
        let json = serde_json::to_string(&instance).unwrap();
        encoder.write_all(json.as_bytes())?;
        encoder.write_all(b"\n")?;
    }
    encoder.finish()?;
    Ok(())
}

fn run_countdown_exhaustive_mode(num_count: usize, file: File) -> io::Result<()> {
    let buf_writer = BufWriter::new(file);
    let mut encoder = Encoder::new(buf_writer, 3)?;
    let target_range = 100..=999;
    println!(
        "Performing exhaustive search for Countdown games ({} numbers)...",
        num_count
    );
    let instances = generate_exhaustive_countdown_instances(num_count, target_range);
    for instance in instances {
        let json = serde_json::to_string(&instance).unwrap();
        encoder.write_all(json.as_bytes())?;
        encoder.write_all(b"\n")?;
    }
    encoder.finish()?;
    Ok(())
}

//
// Main
//
fn main() -> io::Result<()> {
    let args = Args::parse();
    println!("Starting dataset generation in mode: {:?}", args.mode);
    let start_time = Instant::now();

    // Create the output file.
    let file = File::create(&args.outfile)?;

    // Dispatch based on mode.
    match args.mode {
        Mode::Random => run_random_mode(args.num_instances, args.min_size, args.max_size, file)?,
        Mode::RandomPerfect => {
            run_random_perfect_mode(args.num_instances, args.min_size, args.max_size, file)?
        }
        Mode::RandomNear => {
            run_random_near_mode(args.num_instances, args.min_size, args.max_size, file)?
        }
        Mode::Exhaustive => run_exhaustive_mode(args.num_count, file)?,
        Mode::CountdownRandom => {
            run_countdown_random_mode(args.num_instances, args.num_count, file)?
        }
        Mode::CountdownExhaustive => run_countdown_exhaustive_mode(args.num_count, file)?,
    }

    let elapsed = start_time.elapsed();
    println!("\nFinal Statistics:");
    println!("Total time: {:?}", elapsed);
    println!("Output written to: {}", args.outfile);
    Ok(())
}
