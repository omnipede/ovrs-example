use std::fs;
use openvino::{Blob, Core, Layout, Precision, TensorDesc};

fn main() {
    // Init Openvino core
    println!("OpenVINO version: {}", openvino::version());
    let mut core = Core::new(None).expect("to instantiate the OpenVINO library");
    let mut network = core.read_network_from_file(
        "./public/alexnet/FP32/alexnet.xml",
        "./public/alexnet/FP32/alexnet.bin"
    ).unwrap();
    let input_name = &network.get_input_name(0).unwrap();
    assert_eq!(input_name, "data");
    network.set_input_layout(input_name, Layout::NHWC).unwrap();
    let output_name = &network.get_output_name(0).unwrap();
    assert_eq!(output_name, "prob");

    // Load the network
    let mut executable_network = core.load_network(&network, "CPU").unwrap();
    let mut infer_request = executable_network.create_infer_request().unwrap();

    // Read the image
    let tensor_data = fs::read("./tensor-1x3x227x227-f32.bgr").unwrap();
    let tensor_desc = TensorDesc::new(Layout::NHWC, &[1, 3, 227, 227], Precision::FP32);
    let blob = Blob::new(&tensor_desc, &tensor_data).unwrap();

    // Execute inference
    infer_request.set_blob(input_name, &blob).unwrap();
    infer_request.infer().unwrap();
    let mut results = infer_request.get_blob(output_name).unwrap();
    let buffer = unsafe {
        results.buffer_mut_as_type::<f32>().unwrap().to_vec()
    };

    // Print results
    struct Result {
        id: usize,
        prob: f32
    }

    let mut results: Vec<Result> = buffer.iter().enumerate()
        .map(|(c, p)| Result { id: c, prob: *p })
        .collect();

    results.sort_by(|r1, r2| {
        r2.prob.partial_cmp(&r1.prob).unwrap()
    });

    println!("-------\t-----------");
    println!("classid\tprobability");
    println!("-------\t-----------");
    for result in &results[0..10] {
        println!("{}\t{}", result.id, result.prob)
    }
}
