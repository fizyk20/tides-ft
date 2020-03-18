use chrono::prelude::{DateTime, NaiveDateTime, Utc};
use num::Complex;
use serde::Deserialize;

use std::env;
use std::path::Path;

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

struct DataPoint {
    time: f64,
    water_level: f64,
}

struct FtPoint {
    freq: f64,
    amplitude: f64,
}

struct DataSet(Vec<DataPoint>);

impl DataSet {
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

    fn time_interval(&self) -> f64 {
        self.0.last().unwrap().time - self.0.first().unwrap().time
    }

    fn integrate_freq(&self, freq: f64) -> Complex<f64> {
        let neg_i = -Complex::<f64>::i();

        let c = 2.0 * std::f64::consts::PI * neg_i * freq;

        let result: Complex<f64> = self
            .0
            .windows(2)
            .map(|points| {
                let DataPoint {
                    time: time1,
                    water_level: level1,
                } = points[0];
                let DataPoint {
                    time: time2,
                    water_level: level2,
                } = points[1];
                let a = (level2 - level1) / (time2 - time1);
                let b = level1 - a * time1;
                let int_f = |x: f64| (a * x + b - a / c) / c * (c * x).exp();
                int_f(time2) - int_f(time1)
            })
            .sum();

        result / self.time_interval()
    }
}

fn fourier(data: &DataSet, start_freq: f64, end_freq: f64, step: f64) -> Vec<FtPoint> {
    let mut result = vec![];

    let mut current_freq = start_freq;
    while current_freq <= end_freq {
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
    let file_path = env::args_os().nth(1).unwrap();
    let start_freq: f64 = env::args_os()
        .nth(2)
        .and_then(|arg| arg.into_string().ok())
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(0.0);
    let end_freq: f64 = env::args_os()
        .nth(3)
        .and_then(|arg| arg.into_string().ok())
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(5.0 / 86400.0);
    let step: f64 = env::args_os()
        .nth(4)
        .and_then(|arg| arg.into_string().ok())
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(5.0 / 86400.0 / 30000.0);
    let data = DataSet::get_data(file_path);

    let fourier = fourier(&data, start_freq, end_freq, step);

    for point in fourier.into_iter() {
        println!("{} {}", point.freq * 86400.0, point.amplitude);
    }
}
