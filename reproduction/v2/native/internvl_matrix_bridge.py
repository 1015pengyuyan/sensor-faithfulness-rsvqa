"""Project verified native observations into the unchanged shared runner schema."""
from internvl_native_128_bridge import InternVLNative128Adapter


class MatrixInternVL128Adapter:
    def __init__(self, native):
        self.native = native
        self.model = native.model
        self.max_new_tokens = native.max_new_tokens

    @classmethod
    def from_pretrained(cls, model_path, max_new_tokens=128):
        return cls(InternVLNative128Adapter.from_pretrained(model_path, max_new_tokens=max_new_tokens))

    @property
    def counts(self):
        return self.native.counts

    def generate(self, row, images):
        raw, original = self.native.generate(row, images)
        trace = dict(original)
        if trace.get('ingestion_verified') is not True:
            raise RuntimeError('native InternVL ingestion verification did not pass')
        no_image = (not images and trace.get('native_pixel_values_is_none') is True
                    and trace.get('vision_forward_count') == 0
                    and trace.get('image_block_count') == 0
                    and trace.get('image_context_token_count') == 0)
        if not images and not no_image:
            raise RuntimeError('native InternVL no-image evidence is incomplete')
        trace['processor_images_argument_omitted'] = no_image
        trace['visual_tensor_keys'] = ['pixel_values'] if images else []
        trace['image_count'] = len(images)
        trace['shared_schema_projection'] = 'native verified tensor/marker observations; raw trace retained'
        return raw, trace
