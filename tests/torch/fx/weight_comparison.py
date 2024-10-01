import torch

ov_weight = torch.load('OV_FP_weight')
fx_weight = torch.load('FP_Weight')
ov_int8 = torch.load('OV_int8_values')
fx_int8 = torch.load('FX_int_8')
# ov_fq_weight = torch.load('OV_FQ_output')
# fx_fq_weight = torch.load('')
torch.set_printoptions(precision=8, sci_mode=False)
ov_pre_transform_fq_weight = torch.load('OV_FQ_output')
ov_transform_fq_weight = torch.load('OV_FQ_output_with_transformation')
fx_fq_weight = torch.load('FX_FQ_output')
# ov_int8 = ov_int8 - 128
ov_int8 = ov_int8.to(torch.int8)
print(torch.min(ov_int8)) 
print(torch.min(fx_int8))
print(torch.all(ov_int8 == fx_int8))

unequal_mask = torch.ne(ov_int8, fx_int8)

# Step 2: Get the values from both tensors where they are unequal
different_values_tensor1 = ov_int8[unequal_mask]
different_values_tensor2 = fx_int8[unequal_mask]

# Print the results
print("Values in tensor1 that are different:", different_values_tensor1)
print("Values in tensor2 that are different:", different_values_tensor2)
# print(ov_transform_fq_weight[-1][-1])
# print(ov_pre_transform_fq_weight[-1][-1])
# diff = torch.sqrt(torch.mean((fx_fq_weight - ov_transform_fq_weight) ** 2))

# print(diff.item())
# print(torch.all(fx_fq_weight == ov_transform_fq_weight))