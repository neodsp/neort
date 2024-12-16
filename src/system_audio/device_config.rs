#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DeviceConfig {
    pub driver: String,
    pub input_device: String,
    pub output_device: String,
    pub sample_rate: u32,
    pub num_input_ch: u16,
    pub num_output_ch: u16,
    pub num_frames: usize,
}
