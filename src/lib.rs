use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const OPS: [(char, bool); 4] = [('+', true), ('-', false), ('*', true), ('/', false)];
const MAX_TARGET_NUMBER: u64 = 999;
const MAX_STARTING_NUMBER: u64 = 100;

#[derive(Debug, Serialize, Deserialize)]
pub struct GameInstance {
    pub starting: Vec<u64>,
    pub target: u64,
    pub closest: u64,
    pub expression: String,
}

impl GameInstance {
    pub fn new(starting: Vec<u64>, target: u64, closest: u64, expression: String) -> Self {
        Self {
            starting,
            target,
            closest,
            expression,
        }
    }
}

#[derive(Debug, Clone)]
struct Expression {
    value: u64,
    expr_string: String,
}

/// Helper: try to combine two numbers (with given string forms) using the operator.
/// Returns `Some((result, formatted_expression))` if the operation is valid,
/// produces an integer result, and that result lies in 1..=K. Otherwise returns `None`.
fn try_combine(a: u64, b: u64, a_str: &str, b_str: &str, op: char) -> Option<(u64, String)> {
    // Avoid negative results.
    if op == '-' && a < b {
        return None;
    }

    // Avoid division by zero and non-integer division.
    if op == '/' && (b == 0 || a % b != 0) {
        return None;
    }

    let value = match op {
        '+' => a + b,
        '-' => a - b,
        '*' => a * b,
        '/' => a / b,
        _ => unreachable!(),
    };

    // NOTE: As far as I am aware, this is an unproven conjecture in the folklore of
    // countdown: all valid solutions can be rewritten with intermediate values in the
    // range 1..=K.
    if !(1..=MAX_TARGET_NUMBER).contains(&value) {
        return None;
    }

    Some((value, format!("({} {} {})", a_str, op, b_str)))
}

/// Generates `count` random numbers in the range 1..=J.
pub fn generate_numbers(count: usize) -> Vec<u64> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| rng.random_range(1..=MAX_STARTING_NUMBER) as u64)
        .collect()
}

/// Generates a random target value in the range 1..=K.
pub fn generate_target() -> u64 {
    let mut rng = rand::rng();
    rng.random_range(1..=MAX_TARGET_NUMBER) as u64
}

/// Finds the expression that produces a value closest to `target` using a dynamic
/// programming approach where each key is a bitmask representing which numbers have been
/// used.
pub fn find_closest_solution(numbers: &[u64], target: u64) -> (u64, String) {
    // Guard against empty input.
    if numbers.is_empty() {
        return (0, String::new());
    }

    let mut dp: HashMap<u64, Vec<Expression>> = HashMap::new();
    let mut best_diff = u64::MAX;
    let mut best_value = numbers[0];
    let mut best_expr = numbers[0].to_string();

    // Initialize dp with single numbers.
    for (i, &num) in numbers.iter().enumerate() {
        let key = 1u64 << i; // Bitmask for the singleton set.
        let diff = target.abs_diff(num);
        if diff < best_diff {
            best_diff = diff;
            best_value = num;
            best_expr = num.to_string();
        }
        if diff == 0 {
            return (num, num.to_string());
        }
        dp.entry(key).or_default().push(Expression {
            value: num,
            expr_string: num.to_string(),
        });
    }

    // Iterate over subset sizes from 2 up to numbers.len().
    for size in 2..=numbers.len() {
        let mut new_expressions: HashMap<u64, Vec<Expression>> = HashMap::new();

        // Process only those dp entries that use exactly size-1 numbers.
        for (&mask, expressions) in dp.iter() {
            if mask.count_ones() as usize != size - 1 {
                continue;
            }

            // For each number not yet used in this subset.
            for i in 0..numbers.len() {
                if mask & (1 << i) != 0 {
                    continue;
                }
                let new_mask = mask | (1 << i);
                let num = numbers[i];

                // Combine the new number with each expression from the current subset.
                for expr in expressions {
                    for &(op, commutative) in &OPS {
                        // For non-commutative operations, try both orderings.
                        let candidate_combinations = if commutative {
                            vec![(expr.value, num, expr.expr_string.clone(), num.to_string())]
                        } else {
                            vec![
                                (expr.value, num, expr.expr_string.clone(), num.to_string()),
                                (num, expr.value, num.to_string(), expr.expr_string.clone()),
                            ]
                        };

                        for (a, b, a_str, b_str) in candidate_combinations {
                            if let Some((value, expr_str)) = try_combine(a, b, &a_str, &b_str, op) {
                                let diff = (target as i32 - value as i32).abs() as u64;
                                if diff < best_diff {
                                    best_diff = diff;
                                    best_value = value;
                                    best_expr = expr_str.clone();
                                    if diff == 0 {
                                        return (value, best_expr);
                                    }
                                }
                                new_expressions
                                    .entry(new_mask)
                                    .or_default()
                                    .push(Expression {
                                        value,
                                        expr_string: expr_str,
                                    });
                            }
                        }
                    }
                }
            }
        }

        // Instead of sorting then deduplicating, use a HashMap to keep one Expression per value.
        for (mask, exprs) in new_expressions {
            let mut unique: HashMap<u64, Expression> = HashMap::new();
            for e in exprs {
                unique.entry(e.value).or_insert(e);
            }
            let mut deduped: Vec<_> = unique.into_values().collect();
            // Sort so that expressions closer to the target come first.
            deduped.sort_by_key(|e| target.abs_diff(e.value));
            dp.entry(mask).or_default().extend(deduped);
        }
    }

    (best_value, best_expr)
}

/// Generates a dataset of countdown game instances, each with `numbers_per_instance`
/// starting numbers.
pub fn generate_dataset(instances: usize, numbers_per_instance: usize) -> Vec<GameInstance> {
    let mut dataset = Vec::with_capacity(instances);

    for i in 0..instances {
        let numbers = generate_numbers(numbers_per_instance);
        let target = generate_target();
        let (closest, expression) = find_closest_solution(&numbers, target);

        dataset.push(GameInstance {
            starting: numbers,
            target,
            closest,
            expression,
        });

        if (i + 1) % 10 == 0 {
            println!("Generated {} instances", i + 1);
        }
    }

    dataset
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::iter::Peekable;
    use std::str::Chars;

    const SAMPLE_SIZE: usize = 100_000;

    // === Helpers for Statistical Tests ===

    /// Compute the chi-square statistic given counts and the number of categories.
    fn chi_square_test(observations: &HashMap<u64, usize>, n_categories: usize) -> f64 {
        let expected = SAMPLE_SIZE as f64 / n_categories as f64;
        observations.values().fold(0.0, |acc, &count| {
            let diff = count as f64 - expected;
            acc + (diff * diff) / expected
        })
    }

    /// Return the 95% critical value for a given degree of freedom.
    fn get_critical_value(df: usize) -> f64 {
        match df {
            1 => 3.841,
            2 => 5.991,
            3 => 7.815,
            4 => 9.488,
            5 => 11.070,
            6 => 12.592,
            7 => 14.067,
            8 => 15.507,
            9 => 16.919,
            10 => 18.307,
            df => (df as f64 * 1.5) + 3.0, // rough approximation for higher df
        }
    }

    /// Count the number of times each number appears.
    fn count_occurrences<I: IntoIterator<Item = u64>>(iter: I) -> HashMap<u64, usize> {
        let mut counts = HashMap::new();
        for num in iter {
            *counts.entry(num).or_insert(0) += 1;
        }
        counts
    }

    // === Tests for the Generators ===

    #[test]
    fn test_generate_numbers_distribution() {
        let counts = count_occurrences(generate_numbers(SAMPLE_SIZE));
        let df = MAX_STARTING_NUMBER as usize - 1;
        let chi_square = chi_square_test(&counts, MAX_STARTING_NUMBER as usize);
        let critical_value = get_critical_value(df);
        assert!(
            chi_square < critical_value,
            "Chi-square value {} exceeds critical value {}",
            chi_square,
            critical_value
        );
    }

    #[test]
    fn test_generate_target_distribution() {
        let counts = count_occurrences((0..SAMPLE_SIZE).map(|_| generate_target()));
        let df = MAX_TARGET_NUMBER as usize - 1;
        let chi_square = chi_square_test(&counts, MAX_TARGET_NUMBER as usize);
        let critical_value = get_critical_value(df);
        assert!(
            chi_square < critical_value,
            "Chi-square value {} exceeds critical value {}",
            chi_square,
            critical_value
        );
    }

    #[test]
    fn test_exact_solutions() {
        let test_cases = vec![
            (vec![2, 3], 5, 5, "Addition: 2 + 3 = 5"),
            (
                vec![4, 5, 2],
                22,
                22,
                "Multiplication and addition: (4 * 5) + 2 = 22",
            ),
            (
                vec![10, 2, 3],
                15,
                15,
                "Division and multiplication: (10 / 2) * 3 = 15",
            ),
            (vec![6, 2], 3, 3, "Division: 6 / 2 = 3"),
            (vec![7, 8], 56, 56, "Multiplication: 7 * 8 = 56"),
            (
                vec![20, 5, 3],
                1,
                1,
                "Division and subtraction: (20 / 5) - 3 = 1",
            ),
            (
                vec![12, 3, 4],
                16,
                16,
                "Division and multiplication: (12 / 3) * 4 = 16",
            ),
            (
                vec![15, 3, 2],
                10,
                10,
                "Division and addition: (15 / 3) + 2 = 7",
            ),
            (vec![9, 3, 4, 2], 14, 14, "Complex: ((9 / 3) * 4) + 2 = 14"),
            (vec![16, 4, 2, 3], 10, 10, "Complex: ((16 / 4) / 2) * 3 = 6"),
        ];

        for test in test_cases {
            let (result, _) = find_closest_solution(&test.0, test.1);
            assert_eq!(result, test.2, "{}", test.3);
        }
    }

    #[test]
    fn test_closest_solutions() {
        let cases = [
            (vec![5, 4], 21, 20),   // 5 * 4 = 20
            (vec![11, 9], 100, 99), // 11 * 9 = 99
        ];
        for (numbers, target, expected) in cases {
            let (result, _) = find_closest_solution(&numbers, target);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_expression_format() {
        let cases = vec![
            (vec![2, 3], 5, vec!["(2 + 3)", "(3 + 2)"]),
            (vec![4, 2], 8, vec!["(4 * 2)", "(2 * 4)"]),
        ];
        for (numbers, target, valid_exprs) in cases {
            let (_, expr) = find_closest_solution(&numbers, target);
            assert!(
                valid_exprs.contains(&expr.as_str()),
                "Unexpected expression: {}",
                expr
            );
        }
    }

    #[test]
    fn test_edge_cases() {
        // Single number case.
        let (result, expr) = find_closest_solution(&[42], 42);
        assert_eq!(result, 42);
        assert_eq!(expr, "42");

        // Avoid division by zero.
        let (result, _) = find_closest_solution(&[10, 0, 5], 2);
        assert!(result > 0);

        // Ensure intermediate values do not exceed MAX_TARGET_NUMBER.
        let (result, _) = find_closest_solution(&[100, 100, 100], 100);
        assert!(result <= MAX_TARGET_NUMBER);
    }

    #[test]
    fn test_number_generation() {
        let numbers = generate_numbers(6);
        assert_eq!(numbers.len(), 6);
        assert!(numbers
            .iter()
            .all(|&num| num >= 1 && num <= MAX_STARTING_NUMBER));
    }

    #[test]
    fn test_target_generation() {
        for _ in 0..100 {
            let target = generate_target();
            assert!(target >= 1 && target <= MAX_TARGET_NUMBER);
        }
    }

    #[test]
    fn test_solution_completeness() {
        let cases = [(vec![2, 3, 4, 5], 14, 14), (vec![2, 3, 4, 5], 46, 46)];
        for (numbers, target, expected) in cases {
            let (result, _) = find_closest_solution(&numbers, target);
            assert_eq!(result, expected);
        }
    }

    // === Expression Evaluator and Its Tests ===

    struct ExprEvaluator<'a> {
        chars: Peekable<Chars<'a>>,
    }

    impl<'a> ExprEvaluator<'a> {
        fn new(expr: &'a str) -> Self {
            Self {
                chars: expr.chars().peekable(),
            }
        }

        fn skip_whitespace(&mut self) {
            while self.chars.peek().map_or(false, |c| c.is_whitespace()) {
                self.chars.next();
            }
        }

        fn parse_number(&mut self) -> Option<u64> {
            self.skip_whitespace();
            let mut num_str = String::new();
            while let Some(&c) = self.chars.peek() {
                if c.is_ascii_digit() {
                    num_str.push(c);
                    self.chars.next();
                } else {
                    break;
                }
            }
            num_str.parse().ok()
        }

        fn evaluate(&mut self) -> Option<u64> {
            self.evaluate_term()
        }

        fn evaluate_term(&mut self) -> Option<u64> {
            self.skip_whitespace();
            let mut value = self.evaluate_factor()?;
            loop {
                self.skip_whitespace();
                match self.chars.peek() {
                    Some(&'+') => {
                        self.chars.next();
                        value += self.evaluate_factor()?;
                    }
                    Some(&'-') => {
                        self.chars.next();
                        value -= self.evaluate_factor()?;
                    }
                    _ => break,
                }
            }
            Some(value)
        }

        fn evaluate_factor(&mut self) -> Option<u64> {
            self.skip_whitespace();
            let mut value = self.evaluate_primary()?;
            loop {
                self.skip_whitespace();
                match self.chars.peek() {
                    Some(&'*') => {
                        self.chars.next();
                        value *= self.evaluate_primary()?;
                    }
                    Some(&'/') => {
                        self.chars.next();
                        let divisor = self.evaluate_primary()?;
                        if divisor == 0 || value % divisor != 0 {
                            return None;
                        }
                        value /= divisor;
                    }
                    _ => break,
                }
            }
            Some(value)
        }

        fn evaluate_primary(&mut self) -> Option<u64> {
            self.skip_whitespace();
            match self.chars.peek() {
                Some('(') => {
                    self.chars.next(); // consume '('
                    let result = self.evaluate_term()?;
                    self.skip_whitespace();
                    if self.chars.next() == Some(')') {
                        Some(result)
                    } else {
                        None
                    }
                }
                Some(c) if c.is_ascii_digit() => self.parse_number(),
                _ => None,
            }
        }
    }

    /// Checks that all numbers in the expression come from the given list
    /// and that the expression evaluates to the expected value.
    fn verify_solution(numbers: &[u64], expr: &str, expected_value: u64) -> bool {
        let expr_nums: Vec<u64> = expr
            .chars()
            .filter(|c| c.is_ascii_digit() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .filter_map(|s| s.parse::<u64>().ok())
            .collect();

        let mut available_nums = numbers.to_vec();
        for num in expr_nums {
            if let Some(pos) = available_nums.iter().position(|&x| x == num) {
                available_nums.remove(pos);
            } else {
                println!("Number {} not found in available numbers", num);
                return false;
            }
        }

        match ExprEvaluator::new(expr).evaluate() {
            Some(result) if result == expected_value => true,
            Some(result) => {
                println!(
                    "Expression evaluates to {} but expected {}",
                    result, expected_value
                );
                false
            }
            None => {
                println!("Failed to evaluate expression: {}", expr);
                false
            }
        }
    }

    #[test]
    fn test_expression_evaluator() {
        let valid_cases = [
            ("42", 42),
            ("(2 + 3)", 5),
            ("(2 * 3)", 6),
            ("(10 / 2)", 5),
            ("(5 - 3)", 2),
            ("((2 + 3) * 4)", 20),
            ("((10 / 2) + 3)", 8),
            ("((4 * 5) + (2 * 3))", 26),
        ];
        for (expr, expected) in valid_cases {
            assert_eq!(
                ExprEvaluator::new(expr).evaluate(),
                Some(expected),
                "Failed to evaluate {}",
                expr
            );
        }

        let invalid_cases = ["(2 + )", "(2 / 0)", "((2 + 3)", "(2 % 3)", "(5 / 2)"];
        for expr in invalid_cases.iter() {
            assert!(
                ExprEvaluator::new(expr).evaluate().is_none(),
                "Expected failure for invalid expression: {}",
                expr
            );
        }
    }

    #[test]
    fn test_classic_countdown_problems() {
        let test_cases: Vec<(Vec<u64>, u64, bool)> = vec![
            (vec![100, 25, 7, 4, 3, 2], 368, true),
            (vec![50, 75, 25, 100, 3, 6], 952, false),
            (vec![1, 3, 7, 10, 25, 50], 765, true),
            (vec![100, 50, 25, 4, 3, 2], 397, true),
        ];

        for (numbers, target, should_be_exact) in test_cases {
            let (result, expr) = find_closest_solution(&numbers, target);
            if should_be_exact {
                assert_eq!(
                    result, target,
                    "Failed to find exact solution for target {} with numbers {:?}",
                    target, numbers
                );
            }
            assert!(
                verify_solution(&numbers, &expr, result),
                "Invalid solution {} for numbers {:?}",
                expr,
                numbers
            );
        }
    }

    #[test]
    fn test_solution_properties() {
        let numbers = vec![2, 3, 4, 5, 6, 7];
        let target = 120;
        let (result, expr) = find_closest_solution(&numbers, target);

        // Check result bounds.
        assert!(result <= MAX_TARGET_NUMBER && result > 0);

        // Ensure only valid characters are used.
        let valid_chars = "0123456789+-*/() ";
        assert!(
            expr.chars().all(|c| valid_chars.contains(c)),
            "Expression contains invalid characters: {}",
            expr
        );

        // Check balanced parentheses.
        let open_count = expr.chars().filter(|&c| c == '(').count();
        let close_count = expr.chars().filter(|&c| c == ')').count();
        assert_eq!(
            open_count, close_count,
            "Unbalanced parentheses in {}",
            expr
        );
    }

    #[test]
    fn test_difficult_cases() {
        let test_cases = vec![
            (
                "tempting_large_numbers",
                vec![25, 50, 75, 100, 3, 6],
                947,
                false,
            ),
            ("careful_ordering", vec![2, 3, 4, 5, 6, 7], 719, false),
            (
                "multiplication_before_addition",
                vec![4, 5, 6, 7, 8, 9],
                567,
                true,
            ),
            ("early_division", vec![100, 25, 4, 3, 2, 1], 97, true),
        ];

        for (name, numbers, target, should_be_exact) in test_cases {
            println!("Running test case: {}", name);
            let start = std::time::Instant::now();
            let (result, expr) = find_closest_solution(&numbers, target);
            println!("Test case {} took: {:?}", name, start.elapsed());

            if should_be_exact {
                assert_eq!(
                    result, target,
                    "Failed to find exact solution for target {} with numbers {:?}",
                    target, numbers
                );
            }
            assert!(
                verify_solution(&numbers, &expr, result),
                "Invalid solution {} for numbers {:?}",
                expr,
                numbers
            );
        }
    }

    #[test]
    fn test_boundary_cases() {
        // Test with repeated numbers.
        let (result, _) = find_closest_solution(&[7, 7, 7, 7], 28);
        assert_eq!(result, 28);

        // Test with all the same numbers.
        let (result, expr) = find_closest_solution(&[5, 5, 5, 5, 5, 5], 125);
        assert!(verify_solution(&[5, 5, 5, 5, 5, 5], &expr, result));

        // Ensure integer overflow is avoided.
        let (result, _) = find_closest_solution(&[100, 100, 100, 100], MAX_TARGET_NUMBER);
        assert!(result <= MAX_TARGET_NUMBER);

        // Test where division must be performed first.
        let (result, expr) = find_closest_solution(&[100, 2, 3, 4], 150);
        assert!(verify_solution(&[100, 2, 3, 4], &expr, result));
    }

    #[test]
    fn test_dataset_generation() {
        let dataset = generate_dataset(10, 6);
        assert_eq!(dataset.len(), 10);
        for instance in dataset {
            assert_eq!(instance.starting.len(), 6);
            assert!(instance
                .starting
                .iter()
                .all(|&x| x >= 1 && x <= MAX_STARTING_NUMBER));
            assert!(instance.target >= 1 && instance.target <= MAX_TARGET_NUMBER);
            assert!(instance.closest >= 1 && instance.closest <= MAX_TARGET_NUMBER);
            assert!(verify_solution(
                &instance.starting,
                &instance.expression,
                instance.closest
            ));
        }
    }

    #[test]
    fn test_pathological_cases() {
        // Case where all numbers are used.
        let numbers = vec![2, 3, 4, 5, 6, 7];
        let (result, expr) = find_closest_solution(&numbers, MAX_TARGET_NUMBER);
        assert!(result <= MAX_TARGET_NUMBER);
        assert!(verify_solution(&numbers, &expr, result));

        // Case that requires avoiding “tempting” large numbers.
        let numbers = vec![100, 100, 10, 1, 1, 1];
        let target = 12;
        let (result, expr) = find_closest_solution(&numbers, target);
        assert!(verify_solution(&numbers, &expr, result));
    }
}
