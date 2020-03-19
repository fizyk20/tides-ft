use chrono::prelude::{DateTime, NaiveDateTime, Utc};
use num::Complex;
use rayon::prelude::{ParallelIterator, ParallelSlice};
use serde::Deserialize;

use std::env;
use std::path::Path;

/// A single record in a CSV file downloaded from https://tidesandcurrents.noaa.gov/
#[derive(Deserialize, Debug, Clone)]
struct WaterState {
    #[serde(rename = "Date")]
    date: String,
    #[serde(rename = "Time (GMT)")]
    time: String,
    #[serde(rename = "Predicted (m)")]
    predicted: f64,
    #[serde(rename = "Verified (m)")]
    verified: f64,
}

impl WaterState {
    /// Converts the date and time fields into a `DateTime` type
    fn datetime(&self) -> DateTime<Utc> {
        DateTime::<Utc>::from_utc(
            NaiveDateTime::parse_from_str(
                &format!("{} {}", self.date, self.time),
                "%Y/%m/%d %H:%M",
            )
            .unwrap(),
            Utc,
        )
    }
}

/// A single data point about the water level
/// `time` kept in seconds since the first data point
struct DataPoint {
    time: f64,
    water_level: f64,
}

/// A single point of the Fourier transform of the data
/// Frequency stored in Hz
struct FtPoint {
    freq: f64,
    amplitude: f64,
}

/// A dataset representing the variability of the water level over some period of time
struct DataSet(Vec<DataPoint>);

impl DataSet {
    /// Reads the dataset from a CSV file
    fn get_data<P: AsRef<Path>>(path: P) -> Self {
        let mut rdr = csv::Reader::from_path(path).unwrap();
        let mut result = vec![];
        let mut records_iter = rdr.deserialize();
        let first_record: WaterState = records_iter.next().unwrap().unwrap();
        let first_datetime = first_record.datetime();
        result.push(DataPoint {
            time: 0.0,
            water_level: first_record.verified,
        });

        for record in records_iter {
            let record = record.unwrap();
            let data_point = DataPoint {
                time: (record.datetime() - first_datetime).num_seconds() as f64,
                water_level: record.verified,
            };
            result.push(data_point);
        }

        Self(result)
    }

    /// Calculates the length of the period covered by the data
    fn period_length(&self) -> f64 {
        self.0.last().unwrap().time - self.0.first().unwrap().time
    }

    /// Calculates the integral over the covered period of the water level variability function
    /// multiplied by a harmonic function with frequency `freq`
    /// (basically: calculates a single point of the Fourier transform of the dataset)
    fn integrate_freq(&self, freq: f64) -> Complex<f64> {
        let neg_i = -Complex::<f64>::i();

        // Fourier transform of a function f(x), F(k), is the integral of f(x)*exp(-2*pi*i*k*x),
        // where k - the argument of the transformed function - is the frequency.
        // We'll denote c = -2*pi*i*k, so a single point F(k) will be the integral of f(x)*exp(c*x)
        let c = 2.0 * std::f64::consts::PI * neg_i * freq;

        let result: Complex<f64> = self
            .0
            // We will calculate the integral over the whole dataset by summing integrals over
            // intervals between subsequent pairs of data points.
            // Note: the calculation is parallelized here thanks to Rayon.
            .par_windows(2)
            .map(|points| {
                // the point marking the start of the interval
                let DataPoint {
                    time: time1,
                    water_level: level1,
                } = points[0];
                // the point marking the end of the interval
                let DataPoint {
                    time: time2,
                    water_level: level2,
                } = points[1];
                let a = (level2 - level1) / (time2 - time1);
                let b = level1 - a * time1;
                // We assume that the water level function is linear between the two data points,
                // i.e. it is equal to a*x + b for a and b calculated above.
                // Then, we integrate (a*x + b) * exp(c*x) over the interval (time1, time2).
                // The integral of (a*x + b) * exp(c*x) is (a*x + b - a/c)/c * exp(c*x).
                let int_f = |x: f64| (a * x + b - a / c) / c * (c * x).exp();
                int_f(time2) - int_f(time1)
            })
            .sum();

        // Let's normalize the result by dividing by the total length of the period over which we
        // were integrating.
        result / self.period_length()
    }
}

/// Calculates the Fourier transform of the given dataset.
/// The result will cover the range of frequencies starting at `start_freq`, ending at `end_freq`
/// and have a data point every `step`.
fn fourier(data: &DataSet, start_freq: f64, end_freq: f64, step: f64) -> Vec<FtPoint> {
    let mut result = vec![];

    let mut current_freq = start_freq;
    while current_freq <= end_freq {
        // We'll only be interested in the amplitude, which is the norm of the complex value of the
        // Fourier transform.
        let amplitude = data.integrate_freq(current_freq).norm();
        result.push(FtPoint {
            freq: current_freq,
            amplitude,
        });
        current_freq += step;
    }

    result
}

fn main() {
    // Call the script as:
    // ./tides-ft path-to-csv-file [start_freq [end_freq [step]]]
    let file_path = env::args_os().nth(1).unwrap();
    let start_freq: f64 = env::args_os()
        .nth(2)
        .and_then(|arg| arg.into_string().ok())
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(0.0); // default start_freq: 0 Hz
    let end_freq: f64 = env::args_os()
        .nth(3)
        .and_then(|arg| arg.into_string().ok())
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(5.0 / 86400.0); // default end_freq: 5/86400 Hz, or 5/day
    let step: f64 = env::args_os()
        .nth(4)
        .and_then(|arg| arg.into_string().ok())
        .and_then(|arg| arg.parse().ok())
        // default step such that there will be 30000 data points between start and end freqs
        .unwrap_or(5.0 / 86400.0 / 30000.0);
    let data = DataSet::get_data(file_path);

    let fourier = fourier(&data, start_freq, end_freq, step);

    // print the results to stdout
    for point in fourier.into_iter() {
        println!("{} {}", point.freq * 86400.0, point.amplitude);
    }
}
