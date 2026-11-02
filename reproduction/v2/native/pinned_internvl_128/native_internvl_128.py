"""Pinned official InternVL native chat adapter; technical admission only.

Dynamic image tiling follows OpenGVLab's fixed-revision model-card example.
Checkpoint license Apache-2.0; upstream custom implementation headers retain MIT.
"""
from __future__ import annotations
import hashlib
import sys
import time
from pathlib import Path

from internvl_trace import install_internvl_generation_trace_capture as install_generation_trace_capture


def official_tiles(image):
    width,height=image.size
    ratios=set((i,j) for n in range(1,13) for i in range(1,n+1) for j in range(1,n+1) if 1<=i*j<=12)
    ratios=sorted(ratios,key=lambda x:x[0]*x[1])
    best,difference=(1,1),float('inf')
    for ratio in ratios:
        delta=abs(width/height-ratio[0]/ratio[1])
        if delta<difference:
            difference,best=delta,ratio
        elif delta==difference and width*height>0.5*448*448*ratio[0]*ratio[1]:
            best=ratio
    resized=image.resize((448*best[0],448*best[1]))
    tiles=[resized.crop(((i%best[0])*448,(i//best[0])*448,(i%best[0]+1)*448,(i//best[0]+1)*448)) for i in range(best[0]*best[1])]
    if len(tiles)!=1:
        tiles.append(image.resize((448,448)))
    return tiles


def make_packet(question,images):
    images=[image.convert('RGB') for image in images]
    tiles=[official_tiles(image) for image in images]
    return {'question':''.join(f'Image-{i+1}: <image>\n' for i in range(len(images)))+question,
            'tiles':tiles,'num_patches_list':[len(group) for group in tiles],
            'pixel_values':None,
            'image_pixel_sha256':[hashlib.sha256(image.tobytes()).hexdigest() for image in images]}


class GenerateCounter:
    def __init__(self,synchronize=lambda:None):
        self.counts={'invocations':0,'successful_returns':0,'errors':0}
        self.synchronize=synchronize
        self.last_seconds=0.0

    def call(self,function,*args,**kwargs):
        self.synchronize()
        started=time.perf_counter()
        self.counts['invocations']+=1
        try:
            result=function(*args,**kwargs)
        except Exception:
            self.counts['errors']+=1
            raise
        else:
            self.counts['successful_returns']+=1
            return result
        finally:
            self.synchronize()
            self.last_seconds=time.perf_counter()-started


class CapturingTokenizer:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        self.last_text=None

    def __getattr__(self,name):
        return getattr(self.tokenizer,name)

    def __call__(self,text,**kwargs):
        self.last_text=text
        return self.tokenizer(text,**kwargs)


class NativeInternAdapter:
    def __init__(self,path,max_new_tokens=128):
        if max_new_tokens != 128:
            raise ValueError('technical_protocol_128_v1 requires max_new_tokens=128')
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel,AutoTokenizer
        self.torch=torch
        self.transform=T.Compose([T.Lambda(lambda image:image.convert('RGB') if image.mode!='RGB' else image),
                                 T.Resize((448,448),interpolation=InterpolationMode.BICUBIC),T.ToTensor(),
                                 T.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225))])
        self.model=AutoModel.from_pretrained(str(path),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,
                                            use_flash_attn=False,trust_remote_code=True,device_map='auto',local_files_only=True).eval()
        self.tokenizer=CapturingTokenizer(AutoTokenizer.from_pretrained(str(path),trust_remote_code=True,use_fast=False,local_files_only=True))
        self.max_new_tokens=max_new_tokens
        generation_counts={'invocations':0,'successful_returns':0,'errors':0}
        tokenizer_eos=getattr(self.tokenizer,'eos_token_id',None)
        self.counter=install_generation_trace_capture(
            self.model.language_model,
            counts=generation_counts,
            required_max_new_tokens=self.max_new_tokens,
            tokenizer_eos_token_ids=tokenizer_eos,
            synchronize=torch.cuda.synchronize,
        )
        self.last_trace={}
        self.vision_started=0
        self.model.vision_model.register_forward_pre_hook(self._vision_pre,with_kwargs=True)
        self.model.vision_model.register_forward_hook(self._vision_post,with_kwargs=True)

    @property
    def counts(self):
        return self.counter.counts

    def tensor_sha(self,tensor):
        return hashlib.sha256(tensor.detach().contiguous().view(self.torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def _vision_pre(self,module,args,kwargs):
        values=kwargs.get('pixel_values',args[0] if args else None)
        self.last_trace['vision_forward_count']=self.last_trace.get('vision_forward_count',0)+1
        self.last_trace['vision_actual_input_shape']=list(values.shape)
        self.last_trace['vision_actual_input_sha256']=self.tensor_sha(values)
        self.torch.cuda.synchronize()
        self.vision_started=time.perf_counter()

    def _vision_post(self,module,args,kwargs,output):
        self.torch.cuda.synchronize()
        self.last_trace['vision_forward_seconds']=time.perf_counter()-self.vision_started

    def generate(self,row,images):
        self.last_trace={}
        trace=self.last_trace
        self.tokenizer.last_text=None
        started=time.perf_counter()
        packet=make_packet(row['user_prompt'],images)
        device=next(self.model.parameters()).device
        if images:
            groups=[self.torch.stack([self.transform(tile) for tile in tiles]).to(device=device,dtype=self.torch.bfloat16) for tiles in packet['tiles']]
            values=self.torch.cat(groups,dim=0)
        else:
            values=None
        trace.update({'received_image_count':len(images),'image_pixel_sha256':packet['image_pixel_sha256'],
                      'num_patches_list':packet['num_patches_list'],'native_pixel_values_is_none':values is None,
                      'native_pixel_values_shape':list(values.shape) if values is not None else None,
                      'expected_vision_input_sha256':self.tensor_sha(values) if values is not None else None,
                      'processor_seconds':time.perf_counter()-started,'vision_forward_count':0})
        self.model.system_message=row['system_prompt']
        self.torch.cuda.synchronize()
        started=time.perf_counter()
        with self.torch.inference_mode():
            raw=self.model.chat(self.tokenizer,values,packet['question'],
                                {'max_new_tokens':self.max_new_tokens,'do_sample':False},history=None,return_history=False,
                                num_patches_list=packet['num_patches_list'],verbose=False)
        self.torch.cuda.synchronize()
        trace['native_chat_seconds']=time.perf_counter()-started
        trace['language_generate_seconds']=self.counter.last_seconds
        trace['generation_token_trace']=dict(self.counter.last_trace)
        if not trace['generation_token_trace']:
            raise RuntimeError('native chat returned without a traceable low-level generate result')
        prompt=self.tokenizer.last_text or ''
        trace.update({'system_text_in_native_prompt':row['system_prompt'] in prompt,
                      'user_text_in_native_prompt':row['user_prompt'] in prompt,
                      'image_block_count':prompt.count('<img>'),
                      'image_context_token_count':prompt.count('<IMG_CONTEXT>'),
                      'native_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
        images_ok=(trace['vision_forward_count']==1 and trace.get('vision_actual_input_sha256')==trace['expected_vision_input_sha256']
                   and trace['image_block_count']==len(images)
                   and trace['image_context_token_count']==sum(packet['num_patches_list'])*self.model.num_image_token) if images else (
                       trace['vision_forward_count']==0 and trace['native_pixel_values_is_none'] and trace['image_block_count']==0 and trace['image_context_token_count']==0)
        trace['ingestion_verified']=bool(images_ok and trace['system_text_in_native_prompt'] and trace['user_text_in_native_prompt'])
        if not trace['ingestion_verified']:
            raise RuntimeError('native image ingestion or instruction text mismatch')
        return raw.strip(),trace
