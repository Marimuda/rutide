//! Explicit timestamp conversion at application and binding boundaries.
//!
//! The numerical solvers intentionally accept only strictly increasing Modified
//! Julian Days (MJD). This module converts common `UTide` numeric epochs and civil
//! or Rust timestamps into that one internal representation.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::AnalysisError;

const MODIFIED_JULIAN_DAY_AT_UNIX_EPOCH: f64 = 40_587.0;
const PYTHON_GREGORIAN_DAY_AT_MJD_ZERO: f64 = 678_576.0;
const MATLAB_DAY_AT_MJD_ZERO: f64 = 678_942.0;
const MILLISECONDS_PER_DAY: f64 = 86_400_000.0;

/// Epoch attached to a numeric array measured in days.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimeEpoch {
    /// Days since 1858-11-17 00:00:00 UTC.
    ModifiedJulian,
    /// Days since 1970-01-01 00:00:00 UTC.
    Unix,
    /// Python ordinal days, where 0001-01-01 is day 1.
    PythonGregorian,
    /// MATLAB serial days, which are 366 greater than Python ordinal days.
    Matlab,
    /// Days since an explicitly constructed proleptic-Gregorian UTC epoch.
    Gregorian(GregorianDateTime),
}

impl TimeEpoch {
    fn modified_julian_day(self) -> f64 {
        match self {
            Self::ModifiedJulian => 0.0,
            Self::Unix => MODIFIED_JULIAN_DAY_AT_UNIX_EPOCH,
            Self::PythonGregorian => -PYTHON_GREGORIAN_DAY_AT_MJD_ZERO,
            Self::Matlab => -MATLAB_DAY_AT_MJD_ZERO,
            Self::Gregorian(epoch) => epoch.modified_julian_day(),
        }
    }
}

/// Millisecond-resolution proleptic-Gregorian UTC date and time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GregorianDateTime {
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
    millisecond: u16,
}

impl GregorianDateTime {
    /// Construct a validated UTC civil timestamp.
    ///
    /// Astronomical year numbering is used, so year zero is permitted. Leap
    /// seconds are not represented; `second` must be in `0..=59`.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::InvalidGregorianDateTime`] when any component is
    /// outside its valid range.
    pub fn new(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
        millisecond: u16,
    ) -> Result<Self, AnalysisError> {
        if !(1..=12).contains(&month) {
            return Err(AnalysisError::InvalidGregorianDateTime { component: "month" });
        }
        let month_length = days_in_month(year, month);
        if day == 0 || day > month_length {
            return Err(AnalysisError::InvalidGregorianDateTime { component: "day" });
        }
        if hour > 23 {
            return Err(AnalysisError::InvalidGregorianDateTime { component: "hour" });
        }
        if minute > 59 {
            return Err(AnalysisError::InvalidGregorianDateTime {
                component: "minute",
            });
        }
        if second > 59 {
            return Err(AnalysisError::InvalidGregorianDateTime {
                component: "second",
            });
        }
        if millisecond > 999 {
            return Err(AnalysisError::InvalidGregorianDateTime {
                component: "millisecond",
            });
        }
        Ok(Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
            millisecond,
        })
    }

    /// Convert this timestamp to a Modified Julian Day.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "all i32 civil years and sub-day milliseconds are exact within f64 integer precision"
    )]
    pub fn modified_julian_day(self) -> f64 {
        let whole_days = days_from_unix_epoch(self.year, self.month, self.day);
        let milliseconds = i64::from(self.hour) * 3_600_000
            + i64::from(self.minute) * 60_000
            + i64::from(self.second) * 1_000
            + i64::from(self.millisecond);
        (whole_days as f64 + MODIFIED_JULIAN_DAY_AT_UNIX_EPOCH)
            + milliseconds as f64 / MILLISECONDS_PER_DAY
    }
}

/// A clean MJD time axis and its mapping back to the supplied values.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedTimeAxis {
    modified_julian_days: Vec<f64>,
    retained_indices: Vec<usize>,
    source_count: usize,
}

impl NormalizedTimeAxis {
    /// Strictly increasing finite Modified Julian Days.
    #[must_use]
    pub fn modified_julian_days(&self) -> &[f64] {
        &self.modified_julian_days
    }

    /// Source positions retained after non-finite timestamps were removed.
    #[must_use]
    pub fn retained_indices(&self) -> &[usize] {
        &self.retained_indices
    }

    /// Number of timestamps supplied before normalization.
    #[must_use]
    pub const fn source_count(&self) -> usize {
        self.source_count
    }

    /// Number of non-finite timestamps removed during normalization.
    #[must_use]
    pub fn discarded_count(&self) -> usize {
        self.source_count - self.modified_julian_days.len()
    }

    /// Consume the axis into MJD values and retained source positions.
    #[must_use]
    pub fn into_parts(self) -> (Vec<f64>, Vec<usize>) {
        (self.modified_julian_days, self.retained_indices)
    }
}

/// Normalize numeric days from a declared epoch into solver-ready MJD values.
///
/// Matching Python `UTide`'s input boundary, NaN, infinity, and values that
/// overflow during epoch conversion are discarded. The retained source indices
/// allow callers to remove the corresponding observations. `RUTide` additionally
/// requires the retained timestamps to be strictly increasing; it never sorts
/// observations implicitly.
///
/// # Errors
///
/// Returns [`AnalysisError::EmptyTime`] if no finite timestamp remains, or
/// [`AnalysisError::NonIncreasingTime`] with the original source position of a
/// duplicate or decreasing timestamp.
pub fn normalize_numeric_time(
    time_days: &[f64],
    epoch: TimeEpoch,
) -> Result<NormalizedTimeAxis, AnalysisError> {
    let epoch_mjd = epoch.modified_julian_day();
    let mut modified_julian_days = Vec::with_capacity(time_days.len());
    let mut retained_indices = Vec::with_capacity(time_days.len());
    for (index, value) in time_days.iter().copied().enumerate() {
        let modified_julian_day = value + epoch_mjd;
        if !value.is_finite() || !modified_julian_day.is_finite() {
            continue;
        }
        if modified_julian_days
            .last()
            .is_some_and(|previous| modified_julian_day <= *previous)
        {
            return Err(AnalysisError::NonIncreasingTime { index });
        }
        modified_julian_days.push(modified_julian_day);
        retained_indices.push(index);
    }
    if modified_julian_days.is_empty() {
        return Err(AnalysisError::EmptyTime);
    }
    Ok(NormalizedTimeAxis {
        modified_julian_days,
        retained_indices,
        source_count: time_days.len(),
    })
}

/// Convert a Rust [`SystemTime`] instant to a Modified Julian Day.
#[must_use]
pub fn system_time_to_modified_julian_day(time: SystemTime) -> f64 {
    let unix_days = match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs_f64() / 86_400.0,
        Err(error) => -error.duration().as_secs_f64() / 86_400.0,
    };
    unix_days + MODIFIED_JULIAN_DAY_AT_UNIX_EPOCH
}

const fn is_leap_year(year: i32) -> bool {
    year.rem_euclid(4) == 0 && (year.rem_euclid(100) != 0 || year.rem_euclid(400) == 0)
}

const fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 31,
    }
}

fn days_from_unix_epoch(year: i32, month: u8, day: u8) -> i64 {
    let mut year = i64::from(year);
    let month = i64::from(month);
    year -= i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let shifted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use super::{
        GregorianDateTime, TimeEpoch, normalize_numeric_time, system_time_to_modified_julian_day,
    };
    use crate::AnalysisError;

    #[test]
    fn matches_pinned_python_utide_numeric_epochs() {
        for (epoch, input) in [
            (TimeEpoch::ModifiedJulian, 58_113.25),
            (TimeEpoch::Unix, 17_526.25),
            (TimeEpoch::PythonGregorian, 736_689.25),
            (TimeEpoch::Matlab, 737_055.25),
        ] {
            let axis = normalize_numeric_time(&[input], epoch).expect("valid numeric epoch");
            assert_eq!(axis.modified_julian_days(), [58_113.25]);
        }
    }

    #[test]
    fn gregorian_datetime_matches_python_millisecond_conversion() {
        let mjd_epoch = GregorianDateTime::new(1858, 11, 17, 0, 0, 0, 0).expect("valid MJD epoch");
        assert!(mjd_epoch.modified_julian_day().abs() < f64::EPSILON);
        let leap =
            GregorianDateTime::new(2000, 2, 29, 12, 34, 56, 789).expect("valid leap-day timestamp");
        assert!((leap.modified_julian_day() - 51_603.524_268_391_2).abs() < 1e-11);

        let custom = normalize_numeric_time(&[0.0, 1.25], TimeEpoch::Gregorian(mjd_epoch))
            .expect("valid custom epoch");
        assert_eq!(custom.modified_julian_days(), [0.0, 1.25]);
    }

    #[test]
    fn normalization_discards_non_finite_values_and_preserves_source_positions() {
        let axis =
            normalize_numeric_time(&[0.0, f64::NAN, 0.5, f64::INFINITY, 1.0], TimeEpoch::Unix)
                .expect("valid retained time axis");
        assert_eq!(axis.modified_julian_days(), [40_587.0, 40_587.5, 40_588.0]);
        assert_eq!(axis.retained_indices(), [0, 2, 4]);
        assert_eq!(axis.source_count(), 5);
        assert_eq!(axis.discarded_count(), 2);
    }

    #[test]
    fn normalization_rejects_unsafe_order_and_an_empty_retained_axis() {
        assert_eq!(
            normalize_numeric_time(&[1.0, f64::NAN, 1.0], TimeEpoch::ModifiedJulian),
            Err(AnalysisError::NonIncreasingTime { index: 2 })
        );
        assert_eq!(
            normalize_numeric_time(&[f64::NAN, f64::INFINITY], TimeEpoch::ModifiedJulian),
            Err(AnalysisError::EmptyTime)
        );
    }

    #[test]
    fn validates_civil_components_and_converts_system_time() {
        assert!(matches!(
            GregorianDateTime::new(2001, 2, 29, 0, 0, 0, 0),
            Err(AnalysisError::InvalidGregorianDateTime { component: "day" })
        ));
        assert!((system_time_to_modified_julian_day(UNIX_EPOCH) - 40_587.0).abs() < f64::EPSILON);
        assert!(
            (system_time_to_modified_julian_day(UNIX_EPOCH - Duration::from_hours(24)) - 40_586.0)
                .abs()
                < f64::EPSILON
        );
    }
}
