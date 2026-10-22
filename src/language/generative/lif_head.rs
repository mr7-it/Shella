use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::{GenerativeError, Result, surrogate::SurrogateSpike};

pub(crate) const ACTION_COUNT: usize = 3;
const LIF_NEURONS: usize = 16;
const MEMBRANE_DECAY: f64 = 0.85;
const THRESHOLD: f64 = 1.0;
const SURROGATE_SLOPE: f64 = 5.0;

/// Recurrent LIF readout over Transformer hidden states. Forward spikes are binary;
/// only backward propagation uses the fast-sigmoid surrogate derivative.
#[derive(Debug)]
pub(crate) struct DifferentiableLifHead {
    input_projection: Linear,
    recurrent_projection: Linear,
    action_projection: Linear,
}

impl DifferentiableLifHead {
    pub(crate) fn new(hidden_size: usize, builder: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            input_projection: candle_nn::linear(hidden_size, LIF_NEURONS, builder.pp("input"))?,
            recurrent_projection: candle_nn::linear(
                LIF_NEURONS,
                LIF_NEURONS,
                builder.pp("recurrent"),
            )?,
            action_projection: candle_nn::linear(LIF_NEURONS, ACTION_COUNT, builder.pp("actions"))?,
        })
    }

    pub(crate) fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let (batch, sequence, _) = hidden.dims3()?;
        if batch == 0 || sequence == 0 {
            return Err(GenerativeError::InvalidInput {
                parameter: "lif_hidden_states",
                reason: "batch dan urutan harus tidak kosong",
            });
        }
        let currents = self.input_projection.forward(hidden)?;
        let mut membrane = Tensor::zeros((batch, LIF_NEURONS), DType::F32, &Device::Cpu)?;
        let mut previous_spikes = Tensor::zeros((batch, LIF_NEURONS), DType::F32, &Device::Cpu)?;
        let mut spike_counts = Tensor::zeros((batch, LIF_NEURONS), DType::F32, &Device::Cpu)?;
        for step in 0..sequence {
            let input_current = currents.narrow(1, step, 1)?.squeeze(1)?;
            let recurrent_current = self.recurrent_projection.forward(&previous_spikes)?;
            let pre_spike = membrane
                .affine(MEMBRANE_DECAY, 0.0)?
                .broadcast_add(&input_current)?
                .broadcast_add(&recurrent_current)?;
            let threshold_distance = pre_spike.affine(1.0, -THRESHOLD)?;
            previous_spikes = threshold_distance.apply_op1(SurrogateSpike::new(SURROGATE_SLOPE))?;
            let retained_potential = previous_spikes.affine(-1.0, 1.0)?;
            membrane = pre_spike.broadcast_mul(&retained_potential)?;
            spike_counts = spike_counts.broadcast_add(&previous_spikes)?;
        }
        let firing_rate = spike_counts.affine(1.0 / sequence as f64, 0.0)?;
        Ok(self.action_projection.forward(&firing_rate)?)
    }
}
