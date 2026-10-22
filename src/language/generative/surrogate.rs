use candle_core::{
    CpuStorage, CustomOp1, Error, Layout, Result as CandleResult, Shape, Tensor,
    backend::BackendStorage, cpu_backend,
};

trait SurrogateFloat: Copy + PartialOrd {
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}

impl SurrogateFloat for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl SurrogateFloat for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(value: f64) -> Self {
        value
    }
}

fn binary_spike<T: SurrogateFloat>(value: T) -> T {
    T::from_f64(if value.to_f64() >= 0.0 { 1.0 } else { 0.0 })
}

fn fast_sigmoid_derivative<T: SurrogateFloat>(value: T, slope: f64) -> T {
    let denominator = 1.0 + slope * value.to_f64().abs();
    T::from_f64(slope / (denominator * denominator))
}

/// Spike biner pada forward dengan fast-sigmoid surrogate pada backward.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SurrogateSpike {
    slope: f64,
}

impl SurrogateSpike {
    pub(crate) fn new(slope: f64) -> Self {
        Self { slope }
    }
}

impl CustomOp1 for SurrogateSpike {
    fn name(&self) -> &'static str {
        "shella-surrogate-spike"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        let output = candle_core::map_dtype!(
            "shella-surrogate-spike",
            storage,
            |values| cpu_backend::unary_map(values, layout, binary_spike),
            (F32, F64)
        );
        Ok((output, layout.shape().clone()))
    }

    fn bwd(
        &self,
        input: &Tensor,
        _output: &Tensor,
        output_gradient: &Tensor,
    ) -> CandleResult<Option<Tensor>> {
        let derivative = input.apply_op1(SurrogateDerivative { slope: self.slope })?;
        Ok(Some(output_gradient.mul(&derivative)?))
    }
}

#[derive(Debug, Clone, Copy)]
struct SurrogateDerivative {
    slope: f64,
}

impl CustomOp1 for SurrogateDerivative {
    fn name(&self) -> &'static str {
        "shella-fast-sigmoid-surrogate-derivative"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        let slope = self.slope;
        let output = candle_core::map_dtype!(
            "shella-fast-sigmoid-surrogate-derivative",
            storage,
            |values| cpu_backend::unary_map(values, layout, |value| {
                fast_sigmoid_derivative(value, slope)
            }),
            (F32, F64)
        );
        Ok((output, layout.shape().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn forward_is_binary_but_backward_is_nonzero_near_threshold() {
        let input = Var::new(&[-0.2_f32, 0.0, 0.2], &Device::Cpu).unwrap();
        let spikes = input.apply_op1(SurrogateSpike::new(5.0)).unwrap();
        assert_eq!(spikes.to_vec1::<f32>().unwrap(), vec![0.0, 1.0, 1.0]);
        let gradients = spikes.sum_all().unwrap().backward().unwrap();
        let gradient = gradients.get(&input).unwrap().to_vec1::<f32>().unwrap();
        assert!(
            gradient
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
        assert!(gradient[1] > gradient[0]);
        assert!(gradient[1] > gradient[2]);
    }
}
