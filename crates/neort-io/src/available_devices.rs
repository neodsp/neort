#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Driver {
    pub name: String,
    pub input_devices: Vec<String>,
    pub output_devices: Vec<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct AvailableDevices {
    pub drivers: Vec<Driver>,
}

impl AvailableDevices {
    pub fn driver(&self, name: &str) -> Option<&Driver> {
        self.drivers.iter().find(|d| d.name.contains(name))
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct AvailableSettings {
    pub num_input_ch: usize,
    pub num_output_ch: usize,
    pub sample_rates: Vec<usize>,
    pub num_frames: Vec<usize>,
    pub default_sample_rate: usize,
    pub default_num_frames: usize,
}
