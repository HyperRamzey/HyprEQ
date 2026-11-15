//! The crate's error type.
//!
//! Deliberately small: a free-form message, a passed-through I/O error, and the
//! three numeric-domain conditions the limiter and meters can actually refuse on.
//! Every DSP stage reports through this, so a CLI run has one failure type to print
//! and one place where message wording is decided.

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// Free-form message, the equivalent of `Error::msg(..)`.
    Msg(String),
    /// Passthrough I/O failure.
    Io(std::io::Error),
    /// A limiter configuration value was out of its legal range.
    Cfg(String),
    /// A dB value could not be mapped into the parameter range.
    OutOfRange(f64),
    /// Input contained a denormal or non-finite sample.
    DenormalInput,
}

impl Error {
    pub fn msg(s: impl Into<String>) -> Self {
        Error::Msg(s.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Msg(s) => f.write_str(s),
            Error::Io(e) => write!(f, "{e}"),
            Error::Cfg(s) => write!(f, "limiter config: {s}"),
            Error::OutOfRange(v) => write!(f, "value {v} is outside the parameter range"),
            Error::DenormalInput => write!(f, "input contains a denormal or non-finite sample"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
