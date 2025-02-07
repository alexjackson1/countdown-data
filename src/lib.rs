use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Serialize, Deserialize)]
pub struct GameInstance {
    pub starting: Vec<i32>,
    pub target: i32,
    pub closest: i32,
    pub expression: String,
}

#[derive(Debug, Clone)]
struct Expression {
    value: i32,
    expr_string: String,
}

pub fn generate_numbers(count: usize) -> Vec<i32> {
    let mut rng = rand::rng();
    let mut numbers = Vec::with_capacity(count);

    for _ in 0..count {
        numbers.push(rng.random_range(1..=100));
    }

    numbers
}

pub fn generate_target() -> i32 {
    let mut rng = rand::rng();
    rng.random_range(1..=999)
}

// Convert a set of indices into a unique key for the HashMap
fn set_to_key(set: &HashSet<usize>) -> u64 {
    let mut key = 0u64;
    for &idx in set {
        key |= 1 << idx;
    }
    key
}

pub fn find_closest_solution(numbers: &[i32], target: i32) -> (i32, String) {
    let mut dp: HashMap<u64, Vec<Expression>> = HashMap::new();
    let mut best_diff = i32::MAX;
    let mut best_value = numbers[0];
    let mut best_expr = numbers[0].to_string();

    // Initialize with single numbers
    for (i, &num) in numbers.iter().enumerate() {
        let mut set = HashSet::new();
        set.insert(i);
        let key = set_to_key(&set);

        let diff = (target - num).abs();
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

    // Helper function to estimate best possible result using remaining numbers
    fn estimate_bounds(nums: &[i32], current: i32) -> (i32, i32) {
        let product = nums.iter().fold(
            current,
            |acc, &x| {
                if acc <= 999 / x {
                    acc * x
                } else {
                    acc
                }
            },
        );
        let sum = nums.iter().fold(
            current,
            |acc, &x| {
                if acc <= 999 - x {
                    acc + x
                } else {
                    acc
                }
            },
        );
        (1, product.min(999))
    }

    // Try all possible subset sizes
    for size in 2..=numbers.len() {
        let mut new_expressions: HashMap<u64, Vec<Expression>> = HashMap::new();

        // For each existing subset
        for (&set_key, expressions) in dp.iter() {
            let current_set: HashSet<usize> = (0..numbers.len())
                .filter(|i| (set_key & (1 << i)) != 0)
                .collect();

            // Get remaining numbers
            let remaining: Vec<i32> = numbers
                .iter()
                .enumerate()
                .filter(|(i, _)| !current_set.contains(i))
                .map(|(_, &x)| x)
                .collect();

            // For each unused number
            for (i, &num) in numbers.iter().enumerate() {
                if current_set.contains(&i) {
                    continue;
                }

                let mut new_set = current_set.clone();
                new_set.insert(i);
                let new_key = set_to_key(&new_set);

                // Try combining with each existing expression
                for expr in expressions {
                    // Try all operations
                    for &(op, commutative) in
                        &[('+', true), ('-', false), ('*', true), ('/', false)]
                    {
                        let combinations = if commutative {
                            vec![(expr.value, num, expr.expr_string.clone(), num.to_string())]
                        } else {
                            vec![
                                (expr.value, num, expr.expr_string.clone(), num.to_string()),
                                (num, expr.value, num.to_string(), expr.expr_string.clone()),
                            ]
                        };

                        for (a, b, a_str, b_str) in combinations {
                            if op == '/' && (b == 0 || a % b != 0) {
                                continue;
                            }

                            let value = match op {
                                '+' => a + b,
                                '-' => a - b,
                                '*' => a * b,
                                '/' => a / b,
                                _ => unreachable!(),
                            };

                            if value <= 0 || value > 999 {
                                continue;
                            }

                            let diff = (target - value).abs();
                            if diff < best_diff {
                                best_diff = diff;
                                best_value = value;
                                best_expr = format!("({} {} {})", a_str, op, b_str);

                                if diff == 0 {
                                    return (value, best_expr);
                                }
                            }

                            new_expressions
                                .entry(new_key)
                                .or_default()
                                .push(Expression {
                                    value,
                                    expr_string: format!("({} {} {})", a_str, op, b_str),
                                });
                        }
                    }
                }
            }
        }

        // Add new expressions to dp, keeping only most promising ones
        for (key, mut exprs) in new_expressions {
            exprs.sort_by_key(|e| (target - e.value).abs());
            // Keep more expressions, but remove duplicates
            exprs.dedup_by_key(|e| e.value);
            dp.entry(key).or_default().extend(exprs);
        }
    }

    (best_value, best_expr)
}
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

    use std::iter::Peekable;
    use std::str::Chars;

    #[test]
    fn test_set_to_key() {
        let mut set = HashSet::new();
        assert_eq!(set_to_key(&set), 0);

        set.insert(0);
        assert_eq!(set_to_key(&set), 1);

        set.insert(1);
        assert_eq!(set_to_key(&set), 3);

        set.insert(5);
        assert_eq!(set_to_key(&set), 35); // 100011 in binary
    }

    #[test]
    fn test_exact_solutions() {
        // Test case 1: 2 + 3 = 5
        let numbers = vec![2, 3];
        let target = 5;
        let (result, _) = find_closest_solution(&numbers, target);
        assert_eq!(result, target);

        // Test case 2: (4 * 5) + 2 = 22
        let numbers = vec![4, 5, 2];
        let target = 22;
        let (result, _) = find_closest_solution(&numbers, target);
        assert_eq!(result, target);

        // Test case 3: (10 / 2) * 3 = 15
        let numbers = vec![10, 2, 3];
        let target = 15;
        let (result, _) = find_closest_solution(&numbers, target);
        assert_eq!(result, target);
    }

    #[test]
    fn test_closest_solutions() {
        // Test case 1: Target 21 with [5, 4] - closest should be 20
        let numbers = vec![5, 4];
        let target = 21;
        let (result, _) = find_closest_solution(&numbers, target);
        assert_eq!(result, 20); // 5 * 4 = 20

        // Test case 2: Target 100 with [7, 11] - closest should be 99
        let numbers = vec![11, 9];
        let target = 100;
        let (result, _) = find_closest_solution(&numbers, target);
        assert_eq!(result, 99); // 11 * 9 = 99
    }

    #[test]
    fn test_expression_format() {
        let numbers = vec![2, 3];
        let target = 5;
        let (_, expr) = find_closest_solution(&numbers, target);
        assert!(expr == "(2 + 3)" || expr == "(3 + 2)");

        let numbers = vec![4, 2];
        let target = 8;
        let (_, expr) = find_closest_solution(&numbers, target);
        assert!(expr == "(4 * 2)" || expr == "(2 * 4)");
    }

    #[test]
    fn test_edge_cases() {
        // Single number
        let numbers = vec![42];
        let target = 42;
        let (result, expr) = find_closest_solution(&numbers, target);
        assert_eq!(result, 42);
        assert_eq!(expr, "42");

        // Division by zero should be avoided
        let numbers = vec![10, 0, 5];
        let target = 2;
        let (result, _) = find_closest_solution(&numbers, target);
        assert!(result > 0); // Should find some valid solution

        // Numbers that could exceed 999 in intermediate steps
        let numbers = vec![100, 100, 100];
        let target = 100;
        let (result, _) = find_closest_solution(&numbers, target);
        assert!(result <= 999);
    }

    #[test]
    fn test_number_generation() {
        let numbers = generate_numbers(6);
        assert_eq!(numbers.len(), 6);
        for &num in &numbers {
            assert!(num >= 1 && num <= 100);
        }
    }

    #[test]
    fn test_target_generation() {
        for _ in 0..100 {
            let target = generate_target();
            assert!(target >= 1 && target <= 999);
        }
    }

    #[test]
    fn test_solution_completeness() {
        // Test case where multiple combinations are possible
        let numbers = vec![2, 3, 4, 5];
        let target = 14;
        let (result, _) = find_closest_solution(&numbers, target);
        assert_eq!(result, 14); // Can be reached in multiple ways: 2*7, 3+4+7, etc.

        // Test case requiring multi-step operations
        let numbers = vec![2, 3, 4, 5];
        let target = 46;
        let (result, _) = find_closest_solution(&numbers, target);
        assert_eq!(result, 46); // (5 * 9) + 1
    }

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
            while let Some(&c) = self.chars.peek() {
                if !c.is_whitespace() {
                    break;
                }
                self.chars.next();
            }
        }

        fn parse_number(&mut self) -> Option<i32> {
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

        fn evaluate(&mut self) -> Option<i32> {
            self.evaluate_term()
        }

        fn evaluate_term(&mut self) -> Option<i32> {
            self.skip_whitespace();

            let mut value = self.evaluate_factor()?;

            loop {
                self.skip_whitespace();
                match self.chars.peek() {
                    Some(&'+') => {
                        self.chars.next();
                        value = value + self.evaluate_factor()?;
                    }
                    Some(&'-') => {
                        self.chars.next();
                        value = value - self.evaluate_factor()?;
                    }
                    _ => break,
                }
            }

            Some(value)
        }

        fn evaluate_factor(&mut self) -> Option<i32> {
            self.skip_whitespace();

            let mut value = self.evaluate_primary()?;

            loop {
                self.skip_whitespace();
                match self.chars.peek() {
                    Some(&'*') => {
                        self.chars.next();
                        value = value * self.evaluate_primary()?;
                    }
                    Some(&'/') => {
                        self.chars.next();
                        let divisor = self.evaluate_primary()?;
                        if divisor == 0 || value % divisor != 0 {
                            return None;
                        }
                        value = value / divisor;
                    }
                    _ => break,
                }
            }

            Some(value)
        }

        fn evaluate_primary(&mut self) -> Option<i32> {
            self.skip_whitespace();

            match self.chars.peek() {
                Some('(') => {
                    self.chars.next(); // consume '('
                    let result = self.evaluate_term()?;
                    self.skip_whitespace();
                    match self.chars.next() {
                        Some(')') => Some(result),
                        _ => None, // Unmatched parenthesis
                    }
                }
                Some(c) if c.is_ascii_digit() => self.parse_number(),
                _ => None,
            }
        }
    }

    fn verify_solution(numbers: &[i32], expr: &str, expected_value: i32) -> bool {
        // First check all numbers in the expression are from the input
        let mut expr_str = expr.to_string();
        // Remove all operators and parentheses, leaving just numbers and spaces
        for c in ['(', ')', '+', '-', '*', '/'] {
            expr_str = expr_str.replace(c, " ");
        }
        let expr_nums: Vec<i32> = expr_str
            .split_whitespace()
            .filter_map(|s| s.parse::<i32>().ok())
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

        // Evaluate the expression
        if let Some(result) = ExprEvaluator::new(expr).evaluate() {
            if result != expected_value {
                println!(
                    "Expression evaluates to {} but expected {}",
                    result, expected_value
                );
                return false;
            }
            true
        } else {
            println!("Failed to evaluate expression: {}", expr);
            false
        }
    }

    #[test]
    fn test_expression_evaluator() {
        let test_cases = vec![
            ("42", 42),
            ("(2 + 3)", 5),
            ("(2 * 3)", 6),
            ("(10 / 2)", 5),
            ("(5 - 3)", 2),
            ("((2 + 3) * 4)", 20),
            ("((10 / 2) + 3)", 8),
            ("((4 * 5) + (2 * 3))", 26),
        ];

        for (expr, expected) in test_cases {
            let result = ExprEvaluator::new(expr).evaluate();
            assert_eq!(result, Some(expected), "Failed to evaluate {}", expr);
        }

        // Test invalid expressions
        let invalid_cases = vec![
            "(2 + )",   // Missing operand
            "(2 / 0)",  // Division by zero
            "((2 + 3)", // Unmatched parenthesis
            "(2 % 3)",  // Invalid operator
            "(5 / 2)",  // Non-integer division
        ];

        for expr in invalid_cases {
            let result = ExprEvaluator::new(expr).evaluate();
            assert_eq!(
                result, None,
                "Expected None for invalid expression: {}",
                expr
            );
        }
    }
    #[test]
    fn test_classic_countdown_problems() {
        // These are actual problems from the show that should be solvable
        let test_cases = vec![
            // numbers, target, should_be_exact
            (vec![100, 25, 7, 4, 3, 2], 368, true), // 100 * 3 + 7 * (25 - 4 * 2)
            (vec![50, 75, 25, 100, 3, 6], 952, false), // Known to require multiple steps
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

        // Solution shouldn't exceed target bounds
        assert!(result <= 999);
        assert!(result > 0);

        // Expression should contain valid operators
        let valid_chars = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '(', ')', ' ',
        ];
        assert!(
            expr.chars().all(|c| valid_chars.contains(&c)),
            "Expression contains invalid characters: {}",
            expr
        );

        // Parentheses should be balanced
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
        // Break down into separate tests to identify the slow case
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
            let duration = start.elapsed();
            println!("Test case {} took: {:?}", name, duration);

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
        // Test with repeated numbers
        let (result, _expr) = find_closest_solution(&[7, 7, 7, 7], 28);
        assert_eq!(result, 28); // Should be able to do 7 * 4

        // Test with all same numbers
        let (result, expr) = find_closest_solution(&[5, 5, 5, 5, 5, 5], 125);
        assert!(verify_solution(&[5, 5, 5, 5, 5, 5], &expr, result));

        // Test with numbers that could cause integer overflow if not careful
        let (result, _expr) = find_closest_solution(&[100, 100, 100, 100], 999);
        assert!(result <= 999);

        // Test with numbers that require division first
        let (result, expr) = find_closest_solution(&[100, 2, 3, 4], 150);
        assert!(verify_solution(&[100, 2, 3, 4], &expr, result));
    }

    #[test]
    fn test_dataset_generation() {
        let dataset = generate_dataset(10, 6);
        assert_eq!(dataset.len(), 10);

        for instance in dataset {
            // Verify number count
            assert_eq!(instance.starting.len(), 6);

            // Verify number ranges
            assert!(instance.starting.iter().all(|&x| x >= 1 && x <= 100));
            assert!(instance.target >= 1 && instance.target <= 999);
            assert!(instance.closest >= 1 && instance.closest <= 999);

            // Verify solution validity
            assert!(verify_solution(
                &instance.starting,
                &instance.expression,
                instance.closest
            ));
        }
    }

    #[test]
    fn test_pathological_cases() {
        // Test when solution requires using all numbers
        let numbers = vec![2, 3, 4, 5, 6, 7];
        let _target = 2 * 3 * 4 * 5 * 6 * 7;
        let (result, expr) = find_closest_solution(&numbers, 999); // since target > 999
        assert!(result <= 999);
        assert!(verify_solution(&numbers, &expr, result));

        // Test when best solution requires avoiding tempting large numbers
        let numbers = vec![100, 100, 10, 1, 1, 1];
        let target = 12;
        let (result, expr) = find_closest_solution(&numbers, target);
        assert!(verify_solution(&numbers, &expr, result));
    }
}
