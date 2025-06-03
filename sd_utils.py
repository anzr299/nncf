import torch
import nncf
import numpy as np
import json
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin, ConfigMixin, StableDiffusion3Pipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from transformers.models.clip import CLIPTextModelWithProjection
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models import UNet2DConditionModel
import datasets
import pickle 
import os
from pathlib import Path
from transformers.models.clip.configuration_clip import CLIPTextConfig

# This function takes in the models of a SD3 pipeline in the torch fx representation and returns an SD3 pipeline with wrapped models.
def init_pipeline(models_dict, configs_dict, sd_xl=False, model_id="stabilityai/stable-diffusion-3-medium-diffusers"):
    wrapped_models = {}
    vae = AutoencoderKL(**configs_dict['vae'])
    vae.encoder = models_dict['vae_encoder']
    vae.decoder = models_dict['vae_decoder']
    def wrap_model(pipe_model, base_class, config):
        class ModelWrapper(base_class):
            def __init__(self, model, config):
                cls_name = base_class.__name__
                if isinstance(config, dict):
                    super().__init__(**config)
                else:
                    super().__init__(config)

                modules_to_delete = [name for name in self._modules.keys()]
                for name in modules_to_delete:
                    del self._modules[name]

                if cls_name == "AutoencoderKL":
                    self.encoder = model.encoder
                    self.decoder = model.decoder
                else:
                    self.model = model

            def forward(self, *args, **kwargs):
                kwargs.pop("joint_attention_kwargs", None)
                kwargs.pop("return_dict", None)
                return self.model(*args, **kwargs)
        return ModelWrapper(pipe_model, config)
    if(sd_xl):
        wrapped_models["transfounetrmer"] = wrap_model(models_dict["unet"], UNet2DConditionModel, configs_dict["unet"])
    else:
        wrapped_models["transformer"] = wrap_model(models_dict["transformer"], SD3Transformer2DModel, configs_dict["transformer"])
    wrapped_models["vae"] = wrap_model(vae, base_class=AutoencoderKL, config=configs_dict["vae"])
    wrapped_models["text_encoder"] = wrap_model(models_dict["text_encoder"], CLIPTextModelWithProjection, configs_dict["text_encoder"])
    wrapped_models["text_encoder_2"] = wrap_model(models_dict["text_encoder_2"], CLIPTextModelWithProjection, configs_dict["text_encoder_2"])

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, text_encoder_3=None, tokenizer_3=None, **wrapped_models)
    return pipe

def get_model_inputs(sd_xl=False):
    text_encoder_input = torch.ones((1, 77), dtype=torch.long)
    text_encoder_kwargs = {}
    text_encoder_kwargs["output_hidden_states"] = True

    vae_encoder_input = torch.ones((1, 3, 64, 64))
    vae_decoder_input = torch.ones((1, 16, 64, 64))
    unet_kwargs = {}
    unet_args = ()
    if(sd_xl):
        unet_args += (torch.ones((2, 16, 64, 64)), ) # Sample
        unet_args += (torch.from_numpy(np.array([1, 2], dtype=np.float32)), ) # Timestep
        unet_kwargs["encoder_hidden_states"] = torch.ones((2, 77, 2048))
        unet_kwargs["timestep_cond"] = None # Default
        unet_kwargs["cross_attention_kwargs"] = None # Default
        unet_kwargs["added_cond_kwargs"] = {'text_embeds': torch.ones([2, 1280]), 'time_ids': torch.ones([2, 6])}
    else:
        unet_kwargs["hidden_states"] = torch.ones((2, 16, 64, 64))
        unet_kwargs["timestep"] = torch.from_numpy(np.array([1, 2], dtype=np.float32))
        unet_kwargs["encoder_hidden_states"] = torch.ones((2, 154, 4096))
        unet_kwargs["pooled_projections"] = torch.ones((2, 2048))
    return text_encoder_input, text_encoder_kwargs, vae_encoder_input, vae_decoder_input, unet_kwargs, unet_args

def export_models(pipe, return_graphmodule=True):
    is_sdxl = not isinstance(pipe, StableDiffusion3Pipeline)
    configs_dict = {}
    configs_dict["text_encoder"] = pipe.text_encoder.config 
    configs_dict["text_encoder_2"] = pipe.text_encoder_2.config
    configs_dict["transformer"] = pipe.unet.config if is_sdxl else pipe.transformer.config
    configs_dict["vae"] = pipe.vae.config
    text_encoder_input, text_encoder_kwargs, vae_encoder_input, vae_decoder_input, unet_kwargs, unet_args = get_model_inputs(is_sdxl)

    with torch.no_grad():
        with disable_patching():
            text_encoder = torch.export.export_for_training(
                pipe.text_encoder,
                args=(text_encoder_input,),
                kwargs=(text_encoder_kwargs),
            )
            text_encoder_2 = torch.export.export_for_training(
                pipe.text_encoder_2,
                args=(text_encoder_input,),
                kwargs=(text_encoder_kwargs),
            )
            vae_decoder = torch.export.export_for_training(pipe.vae.decoder, args=(vae_decoder_input,))
            vae_encoder = torch.export.export_for_training(pipe.vae.encoder, args=(vae_encoder_input,))
            transformer = torch.export.export_for_training(pipe.unet, args=(unet_args), kwargs=(unet_kwargs)) if is_sdxl else torch.export.export_for_training(pipe.transformer, args=(), kwargs=(unet_kwargs))
    models_dict = {}
    models_dict["transformer"] = transformer
    models_dict["vae_encoder"] = vae_encoder
    models_dict["vae_decoder"] = vae_decoder
    models_dict["text_encoder"] = text_encoder
    models_dict["text_encoder_2"] = text_encoder_2

    models_dict = {key:value.module() if return_graphmodule else value for key, value in models_dict.items()}

    return models_dict, configs_dict


def load_fx_model(path, model_name):
    model_folder = model_name
    config_file_name = model_name
    model_config = {}
    if model_name.startswith('vae'):
        model_folder = "vae"
        config_file_name = "vae"
    elif model_name.startswith('text_encoder'):
        config_file_name = "config"

    path = f'{path}/{model_folder}'
    path = path.replace('//', '/')

    if(not Path(path).exists()):
        raise Exception(f'Path {path} does not exist!')
    
    model = torch.export.load(f'{path}/{model_name}.pt2')
    model = model.module()
    if(model_name.startswith('text_encoder')):
        model_config = CLIPTextConfig.from_json_file(f'{path}/{config_file_name}.json')
    else:
        with open(f'{path}/{config_file_name}.json', 'r') as f:
            model_config = json.load(f)
    return model, model_config

def load_fx_pipeline(path: str):
    models_dict, configs_dict = {}, {}
    models = ["transformer", "vae_encoder", "vae_decoder", "text_encoder", "text_encoder_2"]
    model_to_config = {model:model if not model.startswith("vae") else "vae" for model in models}
    for model, config_name in model_to_config.items():
        models_dict[model], configs_dict[config_name] = load_fx_model(path, model)
    pipe = init_pipeline(models_dict, configs_dict)
    return pipe

def save_fx_model(model, path, model_name, model_config):
    if not isinstance(model, torch.export.ExportedProgram):
        raise Exception('Please Make Sure Model to Save is of type torch.export.ExportedProgram')
    model_folder = model_name
    model_folder = "vae" if model_name.startswith('vae') else model_folder
    path = f'{path}/{model_folder}'
    path = path.replace('//', '/')
    
    if(not Path(path).exists()):
        os.makedirs(path)

    torch.export.save(model, f'{path}/{model_name}.pt2')

    model_name = "vae" if model_name.startswith('vae') else model_name
    if(isinstance(model_config, CLIPTextConfig)):
        model_config.save_pretrained(f'{path}/')
    else:
        with open(f'{path}/{model_name}.json', 'w') as f:
            json.dump(dict(model_config), f)

def save_fx_pipeline(pipeline, path: str):
    models_dict, configs_dict = export_models(pipeline, return_graphmodule=False)
    save_fx_model(models_dict['transformer'], path, "transformer", configs_dict['transformer'])
    save_fx_model(models_dict['text_encoder'], path, "text_encoder", configs_dict['text_encoder'])
    save_fx_model(models_dict['text_encoder_2'], path, "text_encoder_2", configs_dict['text_encoder_2'])
    save_fx_model(models_dict['vae_encoder'], path, "vae_encoder", configs_dict['vae'])
    save_fx_model(models_dict['vae_decoder'], path, "vae_decoder", configs_dict['vae'])

def disable_progress_bar(pipeline, disable=True):
    if not hasattr(pipeline, "_progress_bar_config"):
        pipeline._progress_bar_config = {"disable": disable}
    else:
        pipeline._progress_bar_config["disable"] = disable


class UNetWrapper(SD3Transformer2DModel):
    def __init__(self, transformer, config):
        super().__init__(**config)
        self.transformer = transformer
        self.captured_args = []

    def forward(self, *args, **kwargs):
        kwargs.pop("joint_attention_kwargs", None)
        kwargs.pop("return_dict", None)
        self.captured_args.append((*args, *tuple(kwargs.values())))
        return self.transformer(*args, **kwargs)


def collect_calibration_data(
    pipe, calibration_dataset_size: int=224, num_inference_steps: int=28, save_dataset: bool=True
):
    if(Path('calibration_dataset').exists()): 
        with open('calibration_dataset', 'rb') as f:
            calibration_dataset = pickle.load(f)
        return calibration_dataset
    original_unet = pipe.transformer
    calibration_data = []
    disable_progress_bar(pipe)

    dataset = datasets.load_dataset(
        "google-research-datasets/conceptual_captions",
        split="train",
        trust_remote_code=True,
    ).shuffle(seed=42)
    if(isinstance(pipe, StableDiffusion3Pipeline)):
        transformer_config = dict(pipe.transformer.config)
        wrapped_unet = UNetWrapper(pipe.transformer, transformer_config)
        pipe.transformer = wrapped_unet
    else:
        transformer_config = dict(pipe.unet.config)
        wrapped_unet = UNetWrapper(pipe.unet, transformer_config)
        pipe.unet = wrapped_unet
    # Run inference for data collection
    pbar = calibration_dataset_size
    for i, batch in enumerate(dataset):
        prompt = batch["caption"]
        if len(prompt) > pipe.tokenizer.model_max_length:
            continue
        # Run the pipeline
        pipe(prompt, num_inference_steps=num_inference_steps, height=512, width=512)
        calibration_data.extend(wrapped_unet.captured_args)
        wrapped_unet.captured_args = []
        pbar = (len(calibration_data) - pbar)
        if pbar >= calibration_dataset_size:
            break

    disable_progress_bar(pipe, disable=False)
    pipe.transformer = original_unet

    if(save_dataset):
        with open('calibration_data', 'wb') as f:
            pickle.dump(calibration_data, f)
        
    return calibration_data
