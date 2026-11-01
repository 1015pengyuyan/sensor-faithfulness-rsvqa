import importlib.util
import pathlib
import unittest

P = pathlib.Path(__file__).resolve().parents[1] / 'reproduction/v2/public_io.py'
spec = importlib.util.spec_from_file_location('public_io', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PublicExportTests(unittest.TestCase):
    def test_prediction_preserves_response(self):
        r = dict(call_id='main:x', fixture_id='x', raw_output='No. yes',
                 parsed_answer=None, parser_status='ambiguous', strict_json_valid=False,
                 model_path='/home/private/model', environment={'password': 'secret'})
        out = m.prediction(r)
        self.assertEqual(out['raw_output'], r['raw_output'])
        self.assertIsNone(out['parsed_answer'])
        self.assertNotIn('model_path', out)
        self.assertNotIn('environment', out)

    def test_reject_unknown_machine_path_in_response(self):
        with self.assertRaises(ValueError):
            m.assert_public({'raw_output': '/home/private/file'})

    def test_prose_escape_is_not_a_windows_drive(self):
        m.assert_public({'raw_output': 'The answer is:\\n{"answer":"yes"}'})

    def test_fixture_preserves_prompt_and_order(self):
        r = dict(fixture_id='x', system_prompt='verbatim', user_prompt='question',
                 image_refs=[{'slot': 1, 'relative_path': 's2/x.png', 'sha256': 'a'},
                             {'slot': 2, 'relative_path': 's1/y.png', 'sha256': 'b'}])
        out = m.fixture(r)
        self.assertEqual(out['system_prompt'], 'verbatim')
        self.assertEqual(out['image_refs'], r['image_refs'])

    def test_image_order_and_no_image(self):
        refs = [{'sha256': 'a'}, {'sha256': 'b'}]
        m.check_images(refs, [{'verified_sha256': 'a'}, {'verified_sha256': 'b'}])
        with self.assertRaises(ValueError):
            m.check_images(refs, [{'verified_sha256': 'b'}, {'verified_sha256': 'a'}])
        m.check_images([], [], [])
        with self.assertRaises(ValueError):
            m.check_images([], [], ['pixel_values'])

    def test_missing_and_extra_images_rejected(self):
        with self.assertRaises(ValueError):
            m.check_images([{'sha256': 'a'}], [])
        with self.assertRaises(ValueError):
            m.check_images([], [{'verified_sha256': 'a'}])

    def test_native_support_imports_without_model_loading(self):
        import importlib
        import sys
        native = P.parent / 'native'
        sys.path.insert(0, str(native))
        for name in ['qwen_technical_adapter', 'qwen_frozen_runner', 'native_earthmind_128',
                     'native_mantis_128', 'native_qwen25vl7b_128', 'idefics2_native_128',
                     'llava_native_128', 'qwen3_native_128_bridge', 'internvl_matrix_bridge']:
            importlib.import_module(name)
        module = importlib.import_module('internvl_native_128_bridge')
        module.load_pinned_implementation()  # hash verification and import only; not from_pretrained


if __name__ == '__main__':
    unittest.main()
