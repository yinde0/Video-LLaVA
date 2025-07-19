import argparse
import os

import torch

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_VIDEO_TOKEN, DEFAULT_SPATIO_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.serve.utils import load_image, image_ext, video_ext
from videollava.utils import disable_torch_init
from videollava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from videollava.model.multimodal_encoder.languagebind import get_spatio_temporal_features_torch

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer





def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                     args.load_8bit, args.load_4bit,
                                                                     device=args.device, cache_dir=args.cache_dir)
    image_processor, video_processor = processor['image'], processor['video']

    # Add <spatio> token to tokenizer and model
    new_tokens = [DEFAULT_SPATIO_TOKEN]
    num_new_tokens = tokenizer.add_tokens(new_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    tensor = []
    special_token = []
    args.file = args.file if isinstance(args.file, list) else [args.file]
    for file in args.file:
        if os.path.splitext(file)[-1].lower() in image_ext:
            file = image_processor.preprocess(file, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)
            special_token += [DEFAULT_IMAGE_TOKEN]
        elif os.path.splitext(file)[-1].lower() in video_ext:
            file = video_processor(file, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)
            special_token += [DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames
        else:
            raise ValueError(f'Support video of {video_ext} and image of {image_ext}, but found {os.path.splitext(file)[-1].lower()}')
        print(file.shape)
        tensor.append(file)

    video_spatio_temporal_features = None
    if len(tensor) > 0 and tensor[0].dim() == 4:  # likely a video: (C, T, H, W)
        video_tensor = tensor[0].unsqueeze(0) if tensor[0].dim() == 4 else tensor[0]
        print(f"[DEBUG] CLI Video tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")
        video_tower = model.get_video_tower()
        with torch.no_grad():
            video_features = video_tower(video_tensor)
            print(f"[DEBUG] CLI Raw video features shape: {video_features.shape}, dtype: {video_features.dtype}")
            if isinstance(video_features, (list, tuple)):
                for i, feat in enumerate(video_features):
                    print(f"[DEBUG] CLI video_features[{i}] shape: {feat.shape}, dtype: {feat.dtype}")
            video_spatio_temporal_features = get_spatio_temporal_features_torch(video_features)
            print(f"[DEBUG] CLI Spatiotemporal features shape: {video_spatio_temporal_features.shape}, dtype: {video_spatio_temporal_features.dtype}")

    num_spatio_tokens = getattr(args, 'num_spatio_tokens', 4)  # default 4
    num_image_tokens = getattr(args, 'num_image_tokens', 8)    # default 8

    # Example prompt construction
    prompt_prefix = (DEFAULT_IMAGE_TOKEN * num_image_tokens) + (DEFAULT_SPATIO_TOKEN * num_spatio_tokens)


    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if file is not None:
            # first message
            if getattr(model.config, "mm_use_im_start_end", False):
                inp = ''.join([DEFAULT_IM_START_TOKEN + i + DEFAULT_IM_END_TOKEN for i in special_token]) + (DEFAULT_SPATIO_TOKEN * num_spatio_tokens) + '\n' + inp
            else:
                inp = ''.join(special_token) + (DEFAULT_SPATIO_TOKEN * num_spatio_tokens) + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            file = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Before model.generate, check sequence length
        max_seq_len = getattr(model.config, 'max_position_embeddings', 2048)
        if input_ids is not None and input_ids.shape[-1] > max_seq_len:
            print(f"[WARNING] Truncating input_ids from {input_ids.shape[-1]} to {max_seq_len}")
            input_ids = input_ids[..., :max_seq_len]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,  # video as fake images
                video_spatio_temporal_features=video_spatio_temporal_features,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LanguageBind/Video-LLaVA-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--file", nargs='+', type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
