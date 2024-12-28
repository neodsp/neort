#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeviceConfig {
    pub driver: String,
    pub input_device: String,
    pub output_device: String,
    pub num_input_ch: usize,
    pub num_output_ch: usize,
    pub sample_rate: usize,
    pub num_frames: usize,
}
