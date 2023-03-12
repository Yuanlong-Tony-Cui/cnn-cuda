// This is the skeleton for the CUDA implementation

use crate::cnn::*;
use rustacuda::function::BlockSize;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        // Create: context, module, stream.
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let _context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let module = Module::load_from_string(&ptx)?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;

        Ok(CudaContext {
            conv_layer: DeviceBox::new(&cnn.conv_layer).unwrap(),
            output_layer: DeviceBox::new(&cnn.output_layer).unwrap(),
            module,
            stream,
            _context
        })
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        // Delegate the work to the GPU by launching <kernel.cu> and
        // use `threadIdx.x`, `blockIdx.x`, and `blockDim.x` to replace the loops:
        // 1. Convolution Layer:
        let mut input_matrix = DeviceBox::new(input).unwrap();
        let mut layer1_output_db = DeviceBuffer::from_slice(&[[[0.0f64; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE])?;
        let mut output_layer_output_db = DeviceBuffer::from_slice(&[0.0f64; OUT_LAYER_SIZE])?;
        unsafe {
            // There will be (20*20*10) matrix multiplications that need to be done.
            // Each grid has _ blocks and each block has _ threads.
            let module = &self.module;
            let stream = &self.stream;
            let result = launch!(module.filter_w_conv_layer<<<100, 256, 0, stream>>>(
                input_matrix.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                layer1_output_db.as_device_ptr()
            ));

            let result = launch!(module.filter_w_relu_layer<<<20, 256, 0, stream>>>(
                layer1_output_db.as_device_ptr()
            ));

            let result = launch!(module.filter_w_output_layer<<<20, 256, 0, stream>>>(
                layer1_output_db.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                output_layer_output_db.as_device_ptr()
            ));
            result?;
        }
        // let mut layer1_output = [[[0.0f64; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE];
        // layer1_output_db.copy_to(&mut layer1_output);
        // println!("layer1_output[0][0][0]: {}", layer1_output[0][0][0]);


        // // 2. ReLU Layer:
        // unsafe {
        //     let module = &self.module;
        //     let stream = &self.stream;
        //     let result = launch!(module.filter_w_relu_layer<<<20, 256, 0, stream>>>(
        //         layer1_output_db.as_device_ptr()
        //     ));
        //     result?;
        // }

        // // 3. Output Layer:
        // unsafe {
        //     let module = &self.module;
        //     let stream = &self.stream;
        //     let result = launch!(module.filter_w_output_layer<<<20, 256, 0, stream>>>(
        //         layer1_output_db.as_device_ptr(),
        //         self.output_layer.as_device_ptr(),
        //         output_layer_output_db.as_device_ptr()
        //     ));
        //     result?;
        // }
        self.stream.synchronize()?;
        let mut output_layer_output = [0.0f64; OUT_LAYER_SIZE];
        output_layer_output_db.copy_to(&mut output_layer_output);

        Ok(OutputVec(output_layer_output))
    }
}
