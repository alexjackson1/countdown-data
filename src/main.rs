use countdown::{find_closest_solution, generate_numbers, generate_target, GameInstance};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::prelude::*;
use std::io::{self, Write};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use zstd::Encoder;

const CHUNK_SIZE: usize = 250;
const CHANNEL_BUFFER: usize = 200;

fn generate_chunk(num_count: usize, size: usize) -> Vec<GameInstance> {
    (0..size)
        .into_par_iter()
        .map(|_| {
            let numbers = generate_numbers(num_count);
            let target = generate_target();
            let (closest, expression) = find_closest_solution(&numbers, target);
            GameInstance {
                starting: numbers,
                target,
                closest,
                expression,
            }
        })
        .collect()
}

fn main() -> io::Result<()> {
    const INSTANCES_PER_SIZE: usize = 250_000;

    println!("Generating Countdown dataset...");
    let start_time = Instant::now();

    let file = std::fs::File::create("countdown_dataset.jsonl.zst")?;
    let mut encoder = Encoder::new(file, 3)?;

    // Create progress bars
    let m = MultiProgress::new();
    m.set_draw_target(ProgressDrawTarget::stderr_with_hz(5));

    let total_instances = INSTANCES_PER_SIZE * (8 - 3 + 1);
    let total_pb = m.add(ProgressBar::new(total_instances as u64));
    total_pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {wide_msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let size_pb = m.add(ProgressBar::new(INSTANCES_PER_SIZE as u64));
    size_pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.green/white} {pos:>7}/{len:7} Current size")
            .unwrap()
            .progress_chars("##-"),
    );

    let (tx, rx) = mpsc::sync_channel(CHANNEL_BUFFER);
    let total_pb_writer = total_pb.clone();
    let size_pb_writer = size_pb.clone();

    // Writer thread
    let writer_handle = thread::spawn(move || {
        let mut total_written = 0;
        let mut size_written = 0;
        let mut current_size = 0;
        let mut buffer = String::with_capacity(CHUNK_SIZE * 100);

        while let Ok((instances, num_count, is_last)) = rx.recv() {
            if current_size != num_count {
                size_written = 0;
                current_size = num_count;
                size_pb_writer.reset();
            }

            buffer.clear();
            for instance in instances {
                let json = serde_json::to_string(&instance).unwrap();
                buffer.push_str(&json);
                buffer.push('\n');
                total_written += 1;
                size_written += 1;
            }

            encoder.write_all(buffer.as_bytes()).unwrap();

            total_pb_writer.set_position(total_written as u64);
            size_pb_writer.set_position(size_written as u64);

            let instances_per_sec = total_written as f64 / start_time.elapsed().as_secs_f64();
            total_pb_writer.set_message(format!(
                "Size: {} numbers, Speed: {:.0} instances/sec",
                num_count, instances_per_sec
            ));

            if is_last {
                break;
            }
        }

        encoder.finish().unwrap()
    });

    // Process each size
    for num_count in 3..=8 {
        size_pb.set_length(INSTANCES_PER_SIZE as u64);

        let chunks_needed = (INSTANCES_PER_SIZE + CHUNK_SIZE - 1) / CHUNK_SIZE;

        (0..chunks_needed)
            .into_par_iter()
            .try_for_each(|chunk_idx| -> Result<(), mpsc::SendError<_>> {
                let chunk_size = if chunk_idx == chunks_needed - 1 {
                    INSTANCES_PER_SIZE - (chunks_needed - 1) * CHUNK_SIZE
                } else {
                    CHUNK_SIZE
                };

                let instances = generate_chunk(num_count, chunk_size);
                tx.send((instances, num_count, false))?;
                Ok(())
            })
            .unwrap();
    }

    // Signal writer thread to finish
    tx.send((Vec::new(), 0, true)).unwrap();
    writer_handle.join().unwrap();

    let elapsed = start_time.elapsed();
    total_pb.finish_with_message(format!("Completed in {:?}", elapsed));
    size_pb.finish_and_clear();

    // Print final statistics
    let instances_per_sec = total_instances as f64 / elapsed.as_secs_f64();
    println!("\nFinal Statistics:");
    println!("Total time: {:?}", elapsed);
    println!("Average speed: {:.2} instances/second", instances_per_sec);
    println!("Output written to: countdown_dataset.jsonl.zst");

    Ok(())
}
