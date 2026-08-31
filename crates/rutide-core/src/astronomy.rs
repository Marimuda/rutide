//! Astronomical arguments used by the pinned Python `UTide` oracle.

const GREGORIAN_ORDINAL_AT_MJD_ZERO: f64 = 678_576.0;
const ASTRONOMY_EPOCH_ORDINAL: f64 = 693_595.5;

// Rows are mean lunar longitude, mean solar longitude, lunar perigee,
// negative ascending-node longitude, and solar perigee. Coefficients are
// degrees in the polynomial used by UTide's `ut_astron`.
const COEFFICIENTS: [[f64; 4]; 5] = [
    [270.434_164, 13.176_396_526_8, -0.000_085_0, 0.000_000_039],
    [279.696_678, 0.985_647_335_4, 0.000_022_67, 0.0],
    [334.329_556, 0.111_404_080_3, -0.000_773_9, -0.000_000_26],
    [-259.183_275, 0.052_953_922_2, -0.000_155_7, -0.000_000_050],
    [281.220_844, 0.000_047_068_4, 0.000_033_9, 0.000_000_070],
];

#[derive(Clone, Copy, Debug)]
pub(crate) struct Astronomy {
    pub(crate) cycles: [f64; 6],
    pub(crate) cycles_per_day: [f64; 6],
}

pub(crate) fn at_modified_julian_day(modified_julian_day: f64) -> Astronomy {
    let ordinal_day = modified_julian_day + GREGORIAN_ORDINAL_AT_MJD_ZERO;
    let days = ordinal_day - ASTRONOMY_EPOCH_ORDINAL;
    let scaled_days = days / 10_000.0;
    let arguments = [1.0, days, scaled_days * scaled_days, scaled_days.powi(3)];
    let derivative_arguments = [
        0.0,
        1.0,
        2.0e-4 * scaled_days,
        3.0e-4 * scaled_days * scaled_days,
    ];

    let mut cycles = [0.0; 6];
    let mut cycles_per_day = [0.0; 6];
    for (index, coefficients) in COEFFICIENTS.iter().enumerate() {
        cycles[index + 1] = dot4(*coefficients, arguments) / 360.0 % 1.0;
        cycles_per_day[index + 1] = dot4(*coefficients, derivative_arguments) / 360.0;
    }
    cycles[0] = ordinal_day.rem_euclid(1.0) + cycles[2] - cycles[1];
    cycles_per_day[0] = 1.0 + cycles_per_day[2] - cycles_per_day[1];

    Astronomy {
        cycles,
        cycles_per_day,
    }
}

fn dot4(left: [f64; 4], right: [f64; 4]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2] + left[3] * right[3]
}

#[cfg(test)]
mod tests {
    use super::at_modified_julian_day;

    #[test]
    fn matches_python_utide_astronomy_at_fixture_reference_time() {
        let astronomy = at_modified_julian_day(58_128.5);
        let expected_cycles = [
            0.717_382_277_047_221_4,
            0.588_084_614_771_332_8,
            0.805_466_891_818_554_1,
            0.268_982_363_263_740_25,
            0.621_123_498_909_238_2,
            0.786_807_086_823_151_9,
        ];
        let expected_derivatives = [
            0.966_136_808_058_926_3,
            0.036_601_101_260_367_03,
            0.002_737_909_319_293_390_5,
            0.000_309_453_921_137_102_8,
            0.000_147_093_854_666_155_88,
            0.000_000_130_827_828_230_652_8,
        ];
        for (actual, expected) in astronomy.cycles.into_iter().zip(expected_cycles) {
            assert!((actual - expected).abs() < 2e-15, "{actual} != {expected}");
        }
        for (actual, expected) in astronomy
            .cycles_per_day
            .into_iter()
            .zip(expected_derivatives)
        {
            assert!((actual - expected).abs() < 2e-15, "{actual} != {expected}");
        }
    }
}
