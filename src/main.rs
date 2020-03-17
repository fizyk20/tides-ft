use chrono::prelude::{DateTime, FixedOffset};
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
    #[allow(unused)]
    fn datetime(&self) -> DateTime<FixedOffset> {
        DateTime::parse_from_str(&format!("{} {}", self.date, self.time), "%Y/%m/%d %H:%M").unwrap()
    }
}

fn get_data<P: AsRef<Path>>(path: P) -> Vec<WaterState> {
    let mut rdr = csv::Reader::from_path(path).unwrap();
    let mut result = vec![];
    for record in rdr.deserialize() {
        result.push(record.unwrap());
    }
    result
}

fn fft(data: &[WaterState]) -> Vec<Complex<f64>> {
    let mut result = vec![];
    let len: f64 = data.len() as f64;
    let neg_i = -Complex::<f64>::i();
    for k in 0..data.len() {
        let kf = k as f64;
        let mut sum: Complex<f64> = Complex::new(0.0, 0.0);
        for n in 0..data.len() {
            let nf = n as f64;
            sum += data[n].verified * (2.0 * 3.1415926535897 * neg_i * kf * nf / len).exp();
        }
        result.push(sum / len);
    }
    result
}

fn main() {
    let file_path = env::args_os().nth(1).unwrap();
    let records = get_data(file_path);
    let freq_step = 24.0 / (records.len() as f64);
    let fft = fft(&records).into_iter().map(|x| x.norm());

    for (i, amp) in fft.enumerate() {
        println!("{} {}", (i as f64) * freq_step, amp);
    }
}
