//! Formatting helpers for the one place where byte-for-byte agreement matters: the
//! strings the CLI writes to stderr.
//!
//! The reference builds its messages with Python f-strings, so a value reaches the
//! terminal through `str()` or `repr()`. Neither matches Rust's default `Display`:
//! `{-24.0}` prints `-24` in Rust and `-24.0` in Python, and a quoted token is `'x64'`
//! in Python but `"x64"` in Rust. Those are not cosmetic differences — scripts grep
//! usage errors, and a message that changed shape is a broken script. So every message
//! that mirrors a reference message formats its values here instead of inline.

/// Python's `repr()` of a string, which is what `f"{s!r}"` prints.
///
/// Python prefers single quotes and only switches to double quotes when the value itself
/// contains a `'` and no `"`; Rust's `{:?}` always uses double quotes and escapes
/// differently. The difference is visible in exactly the messages users read:
/// `over-sampling ratio must be 2, 4, 8, 16 or 32 (got 'x64')`.
#[must_use]
pub fn py_repr(value: &str) -> String {
    let has_single = value.contains('\'');
    let has_double = value.contains('"');
    if has_single && !has_double {
        return format!("\"{value}\"");
    }
    let mut out = String::with_capacity(value.len() + 2);
    out.push('\'');
    for c in value.chars() {
        match c {
            '\'' => out.push_str("\\'"),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('\'');
    out
}

/// Python's `str()` of a float, which is what `f"{x}"` prints.
///
/// Rust's `Display` drops the `.0` that Python keeps (`-24` vs `-24.0`), so a domain
/// error read back from the terminal would name a different number than the reference
/// did. Rust's `Debug` for `f64` is the shortest form that round-trips — the same rule
/// CPython uses — so it agrees on `-24.0`, `0.0`, `0.75`, and `inf`. Python formats
/// NaN as lowercase `nan`.
#[must_use]
pub fn py_float(value: f64) -> String {
    if value.is_nan() {
        "nan".to_string()
    } else {
        format!("{value:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_float_keeps_the_digit_python_keeps() {
        // The whole reason this exists: Rust's `{}` prints "-24", argparse's message says
        // "-24.0", and the parity harness compares the line word for word.
        assert_eq!(py_float(-24.0), "-24.0");
        assert_eq!(py_float(0.0), "0.0");
        assert_eq!(py_float(-30.0), "-30.0");
        assert_eq!(py_float(0.75), "0.75");
        assert_eq!(py_float(2.5), "2.5");
        // The values the reports print as words still round-trip.
        for v in [-1.0_f64, -0.0, 1e-3, 1234.5, f64::INFINITY] {
            assert_eq!(py_float(v).parse::<f64>().unwrap_or(v), v, "{v}");
        }
        assert_eq!(py_float(f64::NAN), "nan", "Python spells it nan too");
    }

    #[test]
    fn a_quoted_token_is_quoted_the_python_way() {
        assert_eq!(py_repr("x64"), "'x64'");
        assert_eq!(py_repr("20"), "'20'");
        // Python switches to double quotes only when that avoids an escape.
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("say \"hi\""), "'say \"hi\"'");
        assert_eq!(py_repr("both ' and \""), "'both \\' and \"'");
        assert_eq!(py_repr("tab\t"), "'tab\\t'");
    }

    #[test]
    fn the_messages_the_cli_actually_prints_read_like_the_reference() {
        // Pinned as whole sentences, because these are the strings the harness compares.
        assert_eq!(
            format!(
                "over-sampling ratio must be 2, 4, 8, 16 or 32 (got {})",
                py_repr("x64")
            ),
            "over-sampling ratio must be 2, 4, 8, 16 or 32 (got 'x64')"
        );
        assert_eq!(
            format!(
                "true-peak ceiling must be between {} and {} dBTP (got {})",
                py_float(-24.0),
                py_float(0.0),
                py_float(-30.0)
            ),
            "true-peak ceiling must be between -24.0 and 0.0 dBTP (got -30.0)"
        );
    }
}
