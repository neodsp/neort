#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DeviceConfig {
    pub driver: String,
    pub input_device: String,
    pub output_device: String,
    pub num_input_ch: u16,
    pub num_output_ch: u16,
    pub sample_rate: f64,
    pub num_frames: usize,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ProcessConfig {
    pub num_input_channels: u16,
    pub num_output_channels: u16,
    pub num_frames: usize,
    pub sample_rate: f64,
}
