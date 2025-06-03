import argparse
from sd_utils import export_models, save_fx_pipeline, collect_calibration_data, init_pipeline
import nncf
import numpy as np
from diffusers import StableDiffusionXLPipeline
from nncf.quantization.range_estimator import RangeEstimatorParametersSet
import pickle
from pathlib import Path

parser = argparse.ArgumentParser(description="Quantize the Stable Diffusion 3 Transformer and save it to a specified location.")
parser.add_argument("--save_path", type=str, required=False, default="sd3-fx/", help="Path to save the quantized pipeline.")
args = parser.parse_args()

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
)

models_dict, configs_dict = export_models(pipe)

calibration_dataset = collect_calibration_data(pipe)

models_dict['text_encoder'] = nncf.compress_weights(models_dict['text_encoder'])
models_dict['text_encoder_2'] = nncf.compress_weights(models_dict['text_encoder_2'])
models_dict['vae_encoder'] = nncf.compress_weights(models_dict['vae_encoder'])
models_dict['vae_decoder'] = nncf.compress_weights(models_dict['vae_decoder'])

transformer = models_dict["transformer"]
quantized_transformer = nncf.quantize(
    transformer,
    calibration_dataset=nncf.Dataset(calibration_dataset),
    subset_size=len(calibration_dataset),
    model_type=nncf.ModelType.TRANSFORMER,
    advanced_parameters=nncf.AdvancedQuantizationParameters(
        smooth_quant_alpha=0.7,
        disable_bias_correction=True,
        weights_range_estimator_params=RangeEstimatorParametersSet.MINMAX,
        activations_range_estimator_params=RangeEstimatorParametersSet.MINMAX,
    )
)
models_dict["transformer"] = quantized_transformer

fx_pipe = init_pipeline(models_dict, configs_dict)
save_fx_pipeline(fx_pipe, args.save_path)

print(f"Files for Quantized SD3 is downloaded to '{args.save_path}'")
