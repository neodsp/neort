use event_listener::{Event, Listener};
use neort_blocks::BlockHeap;
use neort_dsp::{
    adapter::resamplers::Resamplers,
    ringbuffer::shared::{create_shared_ringbuffer, RbConsumer, RbProducer},
};
use nih_plug::prelude::*;
use std::{
    sync::{atomic::AtomicBool, Arc},
    thread::{spawn, JoinHandle},
};

// This is a shortened version of the gain example with most comments removed, check out
// https://github.com/robbert-vdh/nih-plug/blob/master/plugins/examples/gain/src/lib.rs to get
// started

struct NeortExamplePlugin {
    params: Arc<NeortExamplePluginParams>,
    block: BlockHeap<f32>,
    thread: Option<JoinHandle<()>>,
    input_prod: RbProducer<f32>,
    output_cons: RbConsumer<f32>,
    event: Arc<Event>,
    terminate_flag: Arc<AtomicBool>,
}

#[derive(Params)]
struct NeortExamplePluginParams {
    /// The parameter's ID is used to identify the parameter in the wrappred plugin API. As long as
    /// these IDs remain constant, you can rename and reorder these fields as you wish. The
    /// parameters are exposed to the host in the same order they were defined. In this case, this
    /// gain parameter is stored as linear gain while the values are displayed in decibels.
    #[id = "gain"]
    pub gain: FloatParam,
}

impl Default for NeortExamplePlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(NeortExamplePluginParams::default()),
            block: BlockHeap::default(),
            thread: None,
            input_prod: RbProducer::default(),
            output_cons: RbConsumer::default(),
            event: Arc::new(Event::new()),
            terminate_flag: Arc::new(AtomicBool::new(false)),
            // adapter: AsyncAdapter::default(),
        }
    }
}

impl Default for NeortExamplePluginParams {
    fn default() -> Self {
        Self {
            // This gain is stored as linear gain. NIH-plug comes with useful conversion functions
            // to treat these kinds of parameters as if we were dealing with decibels. Storing this
            // as decibels is easier to work with, but requires a conversion for every sample.
            gain: FloatParam::new(
                "Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    // This makes the range appear as if it was linear when displaying the values as
                    // decibels
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            // Because the gain parameter is stored as linear gain instead of storing the value as
            // decibels, we need logarithmic smoothing
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            // There are many predefined formatters we can use here. If the gain was stored as
            // decibels instead of as a linear gain value, we could have also used the
            // `.with_step_size(0.1)` function to get internal rounding.
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
        }
    }
}

impl Drop for NeortExamplePlugin {
    fn drop(&mut self) {
        self.terminate_flag
            .store(true, std::sync::atomic::Ordering::SeqCst);
        // make background thread continue if it is in the state
        self.event.notify(usize::MAX);
    }
}

impl Plugin for NeortExamplePlugin {
    const NAME: &'static str = "neort Example Plugin";
    const VENDOR: &'static str = "neodsp";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "info@neodsp.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // The first audio IO layout is used as the default. The other layouts may be selected either
    // explicitly or automatically by the host or the user depending on the plugin API/backend.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),

        aux_input_ports: &[],
        aux_output_ports: &[],

        // Individual ports and the layout as a whole can be named here. By default these names
        // are generated as needed. This layout will be called 'Stereo', while a layout with
        // only one input and output channel would be called 'Mono'.
        names: PortNames::const_default(),
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    // If the plugin can send or receive SysEx messages, it can define a type to wrap around those
    // messages here. The type implements the `SysExMessage` trait, which allows conversion to and
    // from plain byte buffers.
    type SysExMessage = ();
    // More advanced plugins can use this to run expensive background tasks. See the field's
    // documentation for more information. `()` means that the plugin does not have any background
    // tasks.
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // Resize buffers and perform other potentially expensive initialization operations here.
        // The `reset()` function is always called right after this function. You can remove this
        // function if you do not need it.
        self.block = BlockHeap::new(
            audio_io_layout.main_input_channels.unwrap().get() as usize,
            buffer_config.max_buffer_size as usize,
        );

        let (input_prod, mut input_cons) = create_shared_ringbuffer::<f32>(2, 10_000, 1000);

        let (mut output_prod, output_cons) = create_shared_ringbuffer::<f32>(2, 10_000, 1000);

        self.input_prod = input_prod;
        self.output_cons = output_cons;

        let mut block = BlockHeap::<f32>::new(2, 128);

        let params = self.params.clone();

        let mut resamplers =
            Resamplers::<f32>::new(2, buffer_config.sample_rate as usize, 44100, 128);

        let event = self.event.clone();
        let terminate_flag = self.terminate_flag.clone();

        self.thread = Some(spawn(move || loop {
            // check if enough data is present, otherwise continue
            while input_cons.num_frames_stored() >= resamplers.input_frames_next() {
                // pull data from audio thread
                assert!(input_cons.pop_block(resamplers.input_block()));
                resamplers.process_input(block.view_mut());

                // simulated user process
                for channel in block.channels_mut() {
                    channel.iter_mut().for_each(|s| *s *= params.gain.value());
                }

                resamplers.process_output(block.view());
                // push data into audio thread
                assert!(output_prod.push_block(resamplers.output_block()));
            }

            // if not enough data is there to do the processing, wait for new data to arrive in audio thread
            let listener = event.listen();
            listener.wait();

            // make this thrad end, if the program is shutting down
            // make sure that the listener gets notified after setting the terminate flag to true!
            if terminate_flag.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
        }));

        true
    }

    fn reset(&mut self) {
        // Reset buffers and envelopes here. This can be called from the audio thread and may not
        // allocate. You can remove this function if you do not need it.
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let num_channels = buffer.channels();
        let num_frames = buffer.samples();
        self.block
            .copy_from_planar_data_limited(buffer.as_slice(), num_channels, num_frames);

        // push data into ringbuffer for background thread
        assert!(self.input_prod.push_block(self.block.view()));
        // notify background thread that new data is available
        self.event.notify(usize::MAX);

        if self.output_cons.num_frames_stored() >= buffer.samples() {
            assert!(self.output_cons.pop_block(self.block.view_mut()));
        } else {
            // clearing to make it noticable that there were no samples
            // here something more elegant could be done in the future
            nih_log!("MISSED PACKAGE!");
            self.block.clear();
        }

        self.block
            .copy_into_planar_data_limited(buffer.as_slice(), num_channels, num_frames);

        ProcessStatus::Normal
    }
}

impl ClapPlugin for NeortExamplePlugin {
    const CLAP_ID: &'static str = "com.neodsp.example";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("An example for using neort in a plugin.");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    // Don't forget to change these features
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo];
}

impl Vst3Plugin for NeortExamplePlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"NeortExamplePlug";

    // And also don't forget to change these categories
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
}

nih_export_clap!(NeortExamplePlugin);
nih_export_vst3!(NeortExamplePlugin);
