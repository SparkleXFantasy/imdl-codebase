import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from .llm.otter.modeling_otter import OtterForConditionalGeneration


class LLMBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.precision = {}
        self.precision["torch_dtype"] = torch.bfloat16

        self.model = OtterForConditionalGeneration.from_pretrained("/home/aya/workspace/hub/weights/gpt/OTTER-Image-MPT7B", device_map='cuda', **self.precision)

        self.model.text_tokenizer.padding_side = "left"
        self.tokenizer = self.model.text_tokenizer
        self.output_embedding = self.model.lang_encoder.get_output_embeddings()
        self.pad_max_length = 512
        self.model.freeze_weights()
        
    def freeze_weights(self):
        # Freeze all parameters in self.model
        for param in self.parameters():
            param.requires_grad = False
    
    def get_formatted_prompt_hist(self, prompts: list, answers: list) -> str:
        formatted_prompt = "<image>"
        try:
            assert(len(prompts) == len(answers) + 1)
        except:
            print(f'Unmatched length of prompts and answers. Expected len(prompts) == len(answers) + 1. Got {len(prompts)}, {len(answers)}')
        for i in range(len(prompts) - 1):
            formatted_prompt += f'User: {prompts[i]} GPT:<answer> {answers[i]}<|endofchunk|>'
        formatted_prompt += f'User: {prompts[-1]} GPT:<answer>'
        return formatted_prompt

    def encode_vision(self, vision_input):
        vision_input = vision_input.to(self.model.device).to(self.precision["torch_dtype"])
        vision_feat = self.model.encode_vision(vision_input)    # [B, 1, 64, 1024]
        return vision_feat
    
    def pad_max_length_embedding(self, embedding: torch.Tensor, pad_max_length: int=64):
        """
        Pad the embedding from [B, C, D] to [B, MAX_LENGTH, D]. Alignment on the right side.
        Args:
            embedding (torch.Tensor): [B, C, D]
            pad_max_length (int): max length
        """
        return F.pad(embedding, (0, 0, 0, pad_max_length - embedding.shape[1]), 'constant', 0)
        
        
    def forward(self, vision_feat, prompts, output_parsed_response=False):
        """
        Forward the image and prompts for text generation.

        Args:
            vision_feat (torch.Tensor): [B, 1, 64, 1024]
            prompts (str): the list of input prompts
            output_parsed_response (bool): set True to output the parsed response from LLM.
        
        Output:
            answers: the list of text embedding of the answers.
            response_embeddings: the text embedding of responses.
        """
        answers = list()
        response_embeddings = None
        
        B = vision_feat.shape[0]
        
        for b in range(B):
            single_responses = list()
            sequential_output_embeddings = None
            
            single_vision_feat = vision_feat[b, :, : ,:].unsqueeze(0)
            self.model.set_condition_vision(single_vision_feat)
            
            for i in range(len(prompts)):
                lang_x = self.tokenizer(
                    [
                        self.get_formatted_prompt_hist(prompts[:i + 1], single_responses),
                    ],
                    return_tensors="pt",
                )
                lang_x_input_ids = lang_x["input_ids"]
                lang_x_attention_mask = lang_x["attention_mask"]
                
                outputs = self.model.lang_encoder.generate(
                    input_ids=lang_x_input_ids.to(self.model.device),
                    attention_mask=lang_x_attention_mask.to(self.model.device),
                    eos_token_id=self.tokenizer.encode("<|endofchunk|>")[-1],
                    pad_token_id=self.tokenizer.encode("<|endofchunk|>")[-1],
                    max_new_tokens=512,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                )
                response = self.tokenizer.decode(outputs[0])
                parsed_response = (
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
                single_responses.append(parsed_response)
                parsed_response_token_ids = self.tokenizer(
                    [
                        parsed_response
                    ],
                    return_tensors='pt'
                )
                parsed_response_input_ids = parsed_response_token_ids["input_ids"].to(self.model.device)
                parsed_output_embedding = self.output_embedding(parsed_response_input_ids)
                padded_embedding = self.pad_max_length_embedding(parsed_output_embedding)
                sequential_output_embeddings = padded_embedding if sequential_output_embeddings is None else torch.cat([sequential_output_embeddings, padded_embedding], dim=0)
            answers.append(single_responses)
            response_embeddings = sequential_output_embeddings.unsqueeze(0) if response_embeddings is None else torch.cat([response_embeddings, sequential_output_embeddings.unsqueeze(0)], dim=0)
            self.model.lang_encoder.clear_conditioned_layers()
            
        if output_parsed_response:
            return response_embeddings, answers
        else:
            return response_embeddings, None