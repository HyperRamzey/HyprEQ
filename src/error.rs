//! The crate's error type.
//!
//! Deliberately small: a free-form message and a passed-through I/O error. Every DSP
//! stage reports through this, so a CLI run has one failure type to print and one place
//! where message wording is decided. The domain refusals (a limiter value out of range, a
//! bad notch mode, a model with too few knots) are all *messages* with wording chosen to
//! match the reference, so they arrive as [`Error::Msg`]; there is no separate variant for
//! them because nothing in this crate branches on the difference.

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// Free-form message, the equivalent of `Error::msg(..)`.
    Msg(String),
    /// Passthrough I/O failure.
    Io(std::io::Error),
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
