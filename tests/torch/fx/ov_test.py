from openvino.runtime import Core, Model, Tensor, opset11 as ops
import numpy as np
import torch

core = Core()
model = core.read_model("./model_cache/model/model1.xml")


def fake_quantize(input_data_values, input_low, input_high, output_low, output_high, levels=256):
    input_shape = input_data_values.shape
    input_data = ops.parameter(input_shape, dtype=np.float32, name="input")  # Define input tensor
    
    input_low_const = ops.constant(np.array(input_low, dtype=np.float32), name="input_low")
    input_high_const = ops.constant(np.array(input_high, dtype=np.float32), name="input_high")
    output_low_const = ops.constant(np.array(output_low, dtype=np.float32), name="output_low")
    output_high_const = ops.constant(np.array(output_high, dtype=np.float32), name="output_high")
    
    fake_quantize_op = ops.fake_quantize(input_data, input_low_const, input_high_const, output_low_const, output_high_const, levels)
    model = Model([fake_quantize_op], [input_data], "FakeQuantizeModel")
    core = Core()
    compiled_model = core.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()
    infer_request.set_input_tensor(0, input_data_values)
    infer_request.infer()
    output_tensor = infer_request.get_output_tensor(0)
    output_data = np.array(output_tensor) 
    
    return output_data

def convert_and_multiply_operation(tensor1_values, tensor2_values):
    
    tensor1 = ops.constant(tensor1_values, dtype=np.float32)
    tensor2 = ops.constant(tensor2_values, dtype=np.float32)
    
    multiply_op = ops.multiply(tensor1, tensor2)
    model = Model([multiply_op], [], "ElementwiseMultiplyModel")
    core = Core()
    compiled_model = core.compile_model(model, "CPU")
    output_tensor = compiled_model()
    output_data = np.array(output_tensor[0].data)
    return output_data


fake_quantize_node = None
for node in model.get_ops():
    # print(node.get_friendly_name())
    if node.get_friendly_name() == "mul/aten.mul.Tensor/Multiply":
        fake_quantize_node = node
        # print(f"Found FakeQuantize node: {node.get_friendly_name()}")

if fake_quantize_node:
    abc = None
    num_inputs = fake_quantize_node.get_input_size()
    # print(f"The FakeQuantize node has {num_inputs} inputs:")
    input_values = [] 
    for i in range(num_inputs):
        input_value = fake_quantize_node.input_value(i)  # Get the input value (source output)
        input_node = input_value.get_node()  # Get the node that produces this input
        if(input_node.get_friendly_name() == "mul/aten.mul.Tensor/Convert"):
            inp = input_node.input_value(0).get_node()
            input_node = inp
        input_tensor = input_node.get_data()
        
        # Convert OpenVINO tensor to NumPy array for manipulation
        # tensor_data = torch.from_numpy(input_tensor)
        input_values.append(input_tensor)
        # print(f"Input {i}: Tensor shape: {input_tensor.shape}")

    # input_data = input_values[0]
    # input_low = input_values[1]
    # input_high = input_values[2]
    # output_low = input_values[3]
    # output_high = input_values[4] 
    # quantized_output = fake_quantize(input_data, input_low, input_high, output_low, output_high)
    # torch.save(quantized_output, 'saved_tensor')
    # # print("Quantized Output:", quantized_output)
    # print(quantized_output.shape)

    weight = input_values[0]
    scale = input_values[1]

    dequantized_value = convert_and_multiply_operation(scale, weight)
    print(dequantized_value)

else:
    print("FakeQuantize node not found in the model.")




