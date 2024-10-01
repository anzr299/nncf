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

print(ov_int8[-1][-1], fx_int8[-1][-1])
print(torch.all(ov_int8 == fx_int8))
# print(ov_transform_fq_weight[-1][-1])
# print(ov_pre_transform_fq_weight[-1][-1])
# diff = torch.sqrt(torch.mean((fx_fq_weight - ov_transform_fq_weight) ** 2))

# print(diff.item())
# print(torch.all(fx_fq_weight == ov_transform_fq_weight))