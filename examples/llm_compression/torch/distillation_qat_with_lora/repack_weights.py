from optimum.intel.openvino import OVModelForCausalLM

from nncf.quantization.quantize_model import repack_weights

model = OVModelForCausalLM.from_pretrained("output/last")
model.model = repack_weights(model.model)

model.save_pretrained("output/last_repacked")
