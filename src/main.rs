use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::io::Write;
use std::process::exit;
use std::rc::Rc;

use finalfusion::io::ReadEmbeddings;
use finalfusion::prelude::Embeddings;
use finalfusion::storage::StorageViewWrap;
use finalfusion::vocab::SimpleVocab;
use finalfusion::vocab::Vocab;
use itertools::Itertools;
use rand::thread_rng;
use rand::Rng;
use rustyline::config::Builder;
use rustyline::history::MemHistory;
use rustyline::Editor;

struct Command<'a> {
    command: &'a str,
    usage: &'a str,
    description: &'a str,
    run: Box<
        dyn Fn(
            Box<Vec<&str>>,
            HashMap<&'a str, Vec<f32>>,
            HashMap<&'a str, Vec<f32>>,
            Vec<(String, f32)>,
            &'a str,
            Vec<Command>,
        ) -> (Option<i32>, HashMap<&'a str, Vec<f32>>, Vec<(String, f32)>),
    >,
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    match args.len() {
        0 => println!("Invalid mode"),
        1 => println!("Usage: {} <solve|play>", args.get(0).unwrap()),
        2 => {
            let path = args.get(0).unwrap();
            let mode = args.get(1).unwrap();
            match mode.as_str() {
                "solve" => start_solver(),
                "play" => start_game(),
                _ => println!("Usage: {} <solve|play>", path),
            }
        }
        _ => println!("Usage: {} <solve|play>", args.get(0).unwrap()),
    }
}

fn start_solver() {
    println!("Loading...");
    let mut reader = BufReader::new(File::open("./words.bin").unwrap());

    let embeddings: &mut Embeddings<SimpleVocab, StorageViewWrap> =
        Box::<Embeddings<_, _>>::leak(Box::new(Embeddings::read_embeddings(&mut reader).unwrap()));
    let mut words_to_vecs = HashMap::new();
    for word in embeddings.vocab().words().iter() {
        words_to_vecs.insert(word.as_str(), embeddings.embedding(word).unwrap().to_vec());
    }
    let original_words = words_to_vecs.clone();
    let mut log = Vec::new();
    let mut rl: Editor<(), MemHistory> = Editor::with_history(
        Builder::new().auto_add_history(true).build(),
        MemHistory::new(),
    )
    .unwrap();
    let mut commands = HashMap::new();
    let command_vec = init_commands();
    for c in command_vec.iter() {
        commands.insert(c.command, c);
    }
    print!("\x1B[2J\x1B[1;1H");
    let _ = io::stdout().flush();
    println!("Ready! Type a valid command or type h for help.");
    loop {
        let line = rl.readline("semantle> ");
        print!("\x1B[2J\x1B[1;1H");
        let _ = io::stdout().flush();
        if line.is_err() {
            exit(0);
        }
        let command = Rc::new(RefCell::new(
            line.unwrap().to_lowercase().trim().to_string(),
        ));
        if command.borrow().is_empty() {
            continue;
        }
        let t = (*command).borrow().to_owned();
        let terms = t.split(" ").collect::<Vec<_>>();
        let command = terms.get(0).unwrap();

        let func = commands.get(command);

        match func {
            None => println!("Unknown command, please try again."),
            Some(x) => {
                let exit_code;
                (exit_code, words_to_vecs, log) = (x.run)(
                    Box::new(terms.clone()),
                    original_words.clone(),
                    words_to_vecs.clone(),
                    log.clone(),
                    x.usage,
                    init_commands(),
                );
                if let Some(code) = exit_code {
                    exit(code);
                }
            }
        }
    }
}

fn init_commands() -> Vec<Command<'static>> {
    vec![
        Command {
            command: "w",
            usage: "w <word> <value|-r|value -e>",
            description: "Add a word with its similarity, edit an existing word's similarity, or remove a word",
            run: Box::new(|params: Box<Vec<&str>>, original_words: HashMap<&str, Vec<f32>>, mut words_to_vecs: HashMap<&str, Vec<f32>>, mut log: Vec<(String, f32)>, usage: &str, _: Vec<Command>,| {
                let mut params = params.into_iter().skip(1);
                let mut state = AddWordState::Normal;
                let mut word_count = 0;
                let mut word = None;
                let mut val = None;
                while let Some(term) = params.next() {
                    match term {
                        "-n" => state = AddWordState::Normal,
                        "-e" => state = AddWordState::Edit,
                        "-r" => state = AddWordState::Remove,
                        x => match word_count {
                            0 => {
                                if !original_words.contains_key(x) {
                                    println!("Unknown word {}", x);
                                    return (None, words_to_vecs, log);
                                }
                                word = Some(x.to_string());
                                word_count += 1;
                            }
                            1 => match term.parse::<f32>() {
                                Err(_) => {
                                    println!("Usage: {usage}");
                                    return (None, words_to_vecs, log);
                                }
                                Ok(y) => {
                                    val = Some(y);
                                    word_count += 1;
                                }
                            },
                            _ => {
                                println!("Usage: {usage}");
                                return (None, words_to_vecs, log);
                            }
                        },
                    }
                }
                if word.is_none() {
                    println!("Usage: {usage}");
                    return (None, words_to_vecs, log);
                }
                let word = word.unwrap();
                match state {
                    AddWordState::Normal => {
                        if val.is_none() {
                            println!("Usage: {usage}");
                            return (None, words_to_vecs, log);
                        }
                        if log.iter().any(|(a, _)| *a == word) {
                            println!("This word already has a value. Try using -e to change an existing value.");
                            return (None, words_to_vecs, log);
                        }
                        words_to_vecs.retain(|_, value| {
                            filter_embeddings(
                                original_words.get(word.as_str()).unwrap(),
                                value.as_slice(),
                                val.unwrap(),
                            )
                        });
                        log.push((word, val.unwrap()));
                    }
                    AddWordState::Edit => {
                        if val.is_none() || !log.iter().any(|(a, _)| *a == word) {
                            println!("Usage: {usage}");
                            return (None, words_to_vecs, log);
                        }
                        log = log
                            .into_iter()
                            .map(|(a, b)| (a.clone(), if *a == word { val.unwrap() } else { b }))
                            .collect();
                        update_words(&original_words, &mut words_to_vecs, &log);
                    }
                    AddWordState::Remove => {
                        if !log.iter().any(|(a, _)| *a == word) {
                            println!("Usage: {usage}");
                            return (None, words_to_vecs, log);
                        }
                        log = log.into_iter().filter(|(a, _)| *a != word).collect();
                        update_words(&original_words, &mut words_to_vecs, &log);
                    }
                }
                return (None, words_to_vecs, log);
            })
        },
        Command {
            command: "l",
            usage: "l [-d]",
            description: "List the guessed words with their similarities in human-readable or debug mode",
            run: Box::new(|params: Box<Vec<&str>>, _original_words: HashMap<&str, Vec<f32>>, words_to_vecs: HashMap<&str, Vec<f32>>, log: Vec<(String, f32)>, usage: &str, _: Vec<Command>,| {
                let mut params = params.into_iter().skip(1);
                match params.next() {
                    None => {
                        println!("Here are the words and similarities you've provided so far:");
                        log.iter().enumerate().for_each(|(i, (a, b))| {
                            println!("\t{}. `{}` with a similarity of `{}`", i + 1, a, b)
                        });
                        return (None, words_to_vecs, log);
                    }
                    Some("-d") => match params.next() {
                        None => {
                            println!("{:?}", log);
                            return (None, words_to_vecs, log);
                        }
                        Some(_) => {
                            println!("Usage: {usage}");
                            return (None, words_to_vecs, log);
                        }
                    },
                    Some(_) => {
                        println!("Usage: {usage}");
                        return (None, words_to_vecs, log);
                    }
                }
            })
        },
        Command {
            command: "p",
            usage: "p",
            description: "View remaining possible words",
            run: Box::new(|params: Box<Vec<&str>>, _original_words: HashMap<&str, Vec<f32>>, words_to_vecs: HashMap<&str, Vec<f32>>, log: Vec<(String, f32)>, usage: &str, _: Vec<Command>,| {
                let mut params = params.into_iter().skip(1);
                let mut debug_mode = false;
                let mut show_embeddings = false;
                while let Some(term) = params.next() {
                    match term {
                        "-d" => {
                            debug_mode = true;
                        }
                        "-e" => {
                            debug_mode = true;
                            show_embeddings = true;
                        }
                        _ => {
                            println!("Usage: {usage}");
                            return (None, words_to_vecs, log);
                        }
                    }
                }
                match debug_mode {
                    true => {
                        if show_embeddings {
                            println!("{:?}", words_to_vecs);
                        } else {
                            println!("{:?}", words_to_vecs.keys());
                        }
                    }
                    false => {
                        words_to_vecs.keys().for_each(|k| println!("{}", k));
                    }
                }
                return (None, words_to_vecs, log);
            })
        },
        Command {
            command: "q",
            usage: "q",
            description: "Quit",
            run: Box::new(|params: Box<Vec<&str>>, _original_words: HashMap<&str, Vec<f32>>, words_to_vecs: HashMap<&str, Vec<f32>>, log: Vec<(String, f32)>, usage: &str, _: Vec<Command>,| {
                let mut params = params.into_iter().skip(1);
                if let Some(_) = params.next() {
                    println!("Usage: {usage}");
                    return (None, words_to_vecs, log);
                }
                return (Some(0), words_to_vecs, log);
            })
        },
        Command {
            command: "h",
            usage: "h",
            description: "Display this help message",
            run: Box::new(|params: Box<Vec<&str>>, _original_words: HashMap<&str, Vec<f32>>, words_to_vecs: HashMap<&str, Vec<f32>>, log: Vec<(String, f32)>, usage: &str, commands: Vec<Command>,| {
                let mut params = params.into_iter().skip(1);
                if let Some(_) = params.next() {
                    println!("Usage: {usage}");
                    return (None, words_to_vecs, log);
                }
                println!("Type one of the following commands:");
                commands.iter().for_each(|a| { println!("\t{}", a.usage); println!("\t\t{}", a.description) });
                return (None, words_to_vecs, log);
            })
        },
        Command {
            command: "fb",
            usage: "fb",
            description: "Find the best word according to current information",
            run: Box::new(|params: Box<Vec<&str>>, _original_words: HashMap<&str, Vec<f32>>, words_to_vecs: HashMap<&str, Vec<f32>>, log: Vec<(String, f32)>, usage: &str, _: Vec<Command>,| {
                let mut params = params.into_iter().skip(1);
                if let Some(_) = params.next() {
                    println!("Usage: {usage}");
                    return (None, words_to_vecs, log);
                }
                if log.is_empty() {
                    println!("The optimal word based on your current information is eget",);
                    return (None, words_to_vecs, log);
                }
                let best = words_to_vecs
                    .iter()
                    .map(|(a, b)| {
                        (
                            a,
                            words_to_vecs
                                .iter()
                                .map(|(_, d)| (dot_product(b, d) * 10000.).round() as i32)
                                .unique()
                                .count(),
                        )
                    })
                    .fold(("-", 0), |a, b| (b.0, a.1.max(b.1)));
                println!(
                    "The optimal word based on your current information is {}",
                    best.0
                );
                return (None, words_to_vecs, log);
            })
        }
    ]
}

fn start_game() {
    println!("Loading...");
    let mut reader = BufReader::new(File::open("./words.bin").unwrap());

    let embeddings: Embeddings<SimpleVocab, StorageViewWrap> =
        Embeddings::read_embeddings(&mut reader).unwrap();
    let mut words_to_vecs = HashMap::new();
    for word in embeddings.vocab().words().iter() {
        words_to_vecs.insert(word.as_str(), embeddings.embedding(word).unwrap().to_vec());
    }

    let mut rl: Editor<(), MemHistory> = Editor::with_history(
        Builder::new().auto_add_history(true).build(),
        MemHistory::new(),
    )
    .unwrap();

    let words: Vec<_> = words_to_vecs.keys().collect();
    let answer = words.get(thread_rng().gen_range(0..words.len())).unwrap();
    let ans_embedding = words_to_vecs.get(**answer).unwrap();
    let mut similarities = HashMap::new();
    words_to_vecs.iter().for_each(|(a, b)| {
        similarities.insert(a, (10000. * dot_product(ans_embedding, b)).round() / 100.);
    });
    let mut most_similar = similarities.iter().collect::<Vec<_>>();
    most_similar.sort_by(|(_, b), (_, d)| d.total_cmp(b));
    let mut similarities = HashMap::new();
    most_similar
        .iter()
        .enumerate()
        .for_each(|(index, (word, similarity))| {
            similarities.insert(word, (similarity, index));
        });
    let mut guesses = 0;
    let mut guessed = HashSet::new();
    let mut log = Vec::new();
    print!("\x1B[2J\x1B[1;1H");
    let _ = io::stdout().flush();
    let mut max_lens = (0, 0, 0, 0);
    let mut most_recent = 0;
    let screen_width;
    let screen_height;
    if let Some((width, height)) = term_size::dimensions() {
        screen_width = width;
        screen_height = height - 6;
    } else {
        screen_width = 80;
        screen_height = 34;
    }
    println!("Ready! Enter a word to start. Similarity ranges from -100 (worst) to 100 (best). Type !quit to exit or !help for help.");
    loop {
        let line = rl.readline("semantle> ");
        if line.is_err() {
            exit(0);
        }
        let word = line.unwrap();
        let word = word.trim().to_string();

        if word == ***answer {
            println!("You found it in {guesses}! The word is {answer}.");
            exit(0);
        }

        match word.as_str() {
            "!quit" => exit(0),
            "!help" => {
                println!("Enter a word. You'll receive a number, which represents the semantic similarity between your word and the answer. -100 is the worst, 100 is the best. Type !quit to exit or !help to see this message again.");
                continue;
            }
            _ => {}
        }

        if let Some(x) = similarities.get(&&&word.as_str()) {
            if guessed.insert(word.clone()) {
                max_lens.1 = max_lens.1.max(word.len());
                guesses += 1;
                max_lens.0 = max_lens.0.max(guesses.to_string().len());
                if **x.0 == 100. {}
                max_lens.2 = max_lens.2.max(x.0.to_string().len());
                log.push((guesses, word.clone(), x.0, x.1));
                most_recent = guesses;
            } else {
                most_recent = log.iter().filter(|i| i.1 == word).next().unwrap().0;
            }
            print!("\x1B[2J\x1B[2;1H");
        } else {
            print!("\x1B[2J\x1B[1;1H");
            println!("Unknown word {word}");
            if most_recent == 0 {
                continue;
            }
        }
        let mut temp_log = log
            .iter()
            .filter(|i| i.0 != most_recent)
            .collect::<Vec<_>>();
        temp_log.sort_by(|(_, _, a, _), (_, _, b, _)| (***a).total_cmp(**b).reverse());
        let (guess, word, sim, index) = log.get(most_recent - 1).unwrap();
        let num_spaces_4;
        match (sim, index) {
            (_, 0..=999) => {
                max_lens.3 = max_lens.3.max(5 + (1000 - index).to_string().len());
                num_spaces_4 = max_lens.3 - (5 + (1000 - index).to_string().len())
            }
            (x, _) => {
                if ***x >= 20. {
                    max_lens.3 = max_lens.3.max(7);
                    num_spaces_4 = max_lens.3 - 7;
                } else {
                    max_lens.3 = max_lens.3.max(6);
                    num_spaces_4 = max_lens.3 - 6;
                }
            }
        }
        let width = max_lens.0 + max_lens.1 + max_lens.2 + max_lens.3 + 5;
        println!("┌{}┐", "─".repeat(width));
        print!("│ ");
        print!("{guess}");
        let num_spaces_1 = max_lens.0 - guess.to_string().len() + 1;
        print!("{}", " ".repeat(num_spaces_1));
        print!("{word}");
        let num_spaces_2 = max_lens.1 - word.len() + 1;
        print!("{}", " ".repeat(num_spaces_2));
        print!("{sim}");
        let num_spaces_3 = max_lens.2 - sim.to_string().len() + 1;
        print!("{}", " ".repeat(num_spaces_3));
        match (sim, index) {
            (_, 0..=999) => {
                print!("{}/1000", 1000 - index);
            }
            (x, _) => {
                if ***x >= 20. {
                    print!("(tepid)");
                } else {
                    print!("(cold)");
                }
            }
        }
        print!("{} │", " ".repeat(num_spaces_4));

        let columns = 1.max((temp_log.len() + screen_height - 1) / screen_height);
        let temp_log_formatted = temp_log
            .into_iter()
            .map(|(guess, word, sim, index)| {
                format_string(
                    word,
                    *guess,
                    max_lens,
                    ***sim,
                    match (sim, index) {
                        (_, 0..=999) => {
                            format!("{}/1000", 1000 - index)
                        }
                        (x, _) => {
                            if ***x >= 20. {
                                format!("(tepid)")
                            } else {
                                format!("(cold)")
                            }
                        }
                    }
                    .as_str(),
                )
            })
            .collect::<Vec<_>>();
        for i in 0..columns {
            let words = if i == columns - 1 {
                &temp_log_formatted[screen_height * i..]
            } else {
                &temp_log_formatted[screen_height * i..screen_height * (i + 1)]
            };
            print_column(words, width, i, 3, i == columns - 1, screen_height);
        }
        print!("\x1B[{};1H", if temp_log_formatted.len() == 0 {5} else {6}+(screen_height.min(temp_log_formatted.len())));
        let _ = io::stdout().flush();
    }
}

fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|i| *i.0 * *i.1).sum()
}

fn filter_embeddings(v1: &[f32], v2: &[f32], target_val: f32) -> bool {
    let res = dot_product(v1, v2) * 100.0;
    res >= target_val - 0.005 && res < target_val + 0.005
}

fn update_words<'a>(
    original_words: &HashMap<&'a str, Vec<f32>>,
    words_to_vecs: &mut HashMap<&'a str, Vec<f32>>,
    log: &Vec<(String, f32)>,
) {
    *words_to_vecs = original_words.clone();
    for (word, val) in log.iter() {
        let current_vec = original_words.get(word.as_str()).unwrap().clone();
        let current_vec = current_vec.as_slice();
        words_to_vecs.retain(|_, value| filter_embeddings(current_vec, value.as_slice(), *val));
    }
}

enum AddWordState {
    Normal,
    Edit,
    Remove,
}

fn print_column(
    words: &[String],
    width: usize,
    column: usize,
    lines_above: usize,
    last_column: bool,
    height: usize,
) {
    let mut lines_above = lines_above;
    if words.len() == 0 && column == 0 {
        lines_above -= 1;
    } else {
        print!(
            "{}{}",
            if column == 0 {
                format!("\x1B[{};1H├{}", lines_above + 1, "─".repeat(width))
            } else if column == 1 {
                format!(
                    "\x1B[{};{}H┼{}",
                    lines_above + 1,
                    column * width + 2,
                    "─".repeat(width)
                )
            } else {
                format!("\x1B[{};{}H┬{}", lines_above + 1, column*width + 2, "─".repeat(width))
            },
            if last_column && column == 0 {
                "┤"
            } else if last_column {
                "┐"
            } else {
                ""
            }
        );
        for (index, word) in words.iter().enumerate() {
            print!(
                "\x1B[{};{}H│ {} {}",
                index + 2 + lines_above,
                if column == 0 {1} else {column * width + 2},
                word,
                if last_column { "│" } else { "" }
            );
        }
        if column != 0 {
            for i in words.len()..height {
                print!(
                    "\x1B[{};{}H│",
                    i + 2 + lines_above,
                    if column == 0 {1} else {column * width + 2},
                );
            }
        print!("\x1B[{};{}H┘",height+2+lines_above,if column == 0 {1} else {column * width + 2},);
        }
    }
    print!(
        "{}{}",
        if column == 0 {
            format!(
                "\x1B[{};1H└{}",
                words.len() + lines_above + 2,
                "─".repeat(width)
            )
        } else if words.len() != height {
            format!(
                "\x1B[{};{}H├{}",
                words.len() + lines_above + 2,
                column * width + 2,
                "─".repeat(width)
            )
        } else {
            format!(
                "\x1B[{};{}H┴{}",
                words.len() + lines_above + 2,
                column * width + 2,
                "─".repeat(width)
            )
        },
        if last_column { "┘" } else { "" }
    );
    if last_column {
        println!();
    }
}

fn format_string(
    s: &str,
    index: usize,
    widths: (usize, usize, usize, usize),
    sim: f32,
    ranking: &str,
) -> String {
    format!(
        "{}{}{}{}{}{}{}{}",
        index,
        " ".repeat(1 + widths.0 - index.to_string().len()),
        s,
        " ".repeat(1 + widths.1 - s.len()),
        sim,
        " ".repeat(1 + widths.2 - sim.to_string().len()),
        ranking,
        " ".repeat(widths.3 - ranking.len())
    )
}