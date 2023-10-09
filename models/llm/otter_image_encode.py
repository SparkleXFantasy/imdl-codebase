import torch
from otter_ai import OtterForConditionalGeneration
from tqdm import tqdm, trange

precision = {}
precision["torch_dtype"] = torch.bfloat16
model = OtterForConditionalGeneration.from_pretrained("/home/aya/workspace/hub/weights/gpt/OTTER-Image-MPT7B", device_map='cuda', **precision)

answer_token_id = 0
eos_token_id = 0


for i in trange(100):
    
    input = torch.rand((1, 3, 224, 224)).to('cuda').to(torch.bfloat16)


    prompt = '<image>User: What is in the image? GPT:<answer>'
    model.freeze_weights()

    lang_x = model.text_tokenizer(
                [
                    prompt
                ],
                return_tensors="pt",
            )
            
            
    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]
            
    """
        stage 1: vision feature    
    """
    vision_feat = model.encode_vision(input)    # [B, 1, 64, 1024]


    ## Swin Feature to Vision Feat here

    ## QFormer or direct projection

    ## aligned_feat = proj(swin_feat)

    ## loss_feat = L1Loss(vision_feat, swin_feat)

    model.set_condition_vision(vision_feat)

    output = model.lang_encoder.generate(
        input_ids=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        eos_token_id=model.text_tokenizer.encode("<|endofchunk|>")[-1],
        max_new_tokens=512,
        num_beams=4,
        no_repeat_ngram_size=3,
    )

    model.lang_encoder.clear_conditioned_layers()
    response = model.text_tokenizer.decode(output[0])
    parsed_output = (
                response
                .split("<answer>")[-1]
                .lstrip()
                .rstrip()
                .split("<|endofchunk|>")[0]
                .lstrip()
                .rstrip()
                .lstrip('"')
                .rstrip('"')
            )

    parsed_token_ids = model.text_tokenizer(
        [
            parsed_output
        ],
        return_tensors='pt'
    )
    response_x_input_ids = parsed_token_ids["input_ids"].to('cuda')

    embedding = model.lang_encoder.get_output_embeddings()
    output_embedding = embedding(response_x_input_ids)
    print(parsed_output)
    # response_x = model.text_tokenizer(
    #             [
    #                 parsed_output
    #             ],
    #             return_tensors="pt",
    #         )

    # response_x_input_ids = response_x["input_ids"].to('cuda')
    # response_x_attention_mask = response_x["attention_mask"].to('cuda')

    # lang_encoder = model.lang_encoder.transformer

    # response_embedding = lang_encoder(
    #     input_ids=response_x_input_ids,
    #     past_key_values=None,
    #     attention_mask=response_x_attention_mask,
    #     prefix_mask=None,
    #     sequence_id=None,
    #     return_dict=None,
    #     output_attentions=None,
    #     output_hidden_states=None,
    #     use_cache=None,
    #     )

    # print(response_embedding.shape)