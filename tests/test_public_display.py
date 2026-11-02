import io
import pathlib
import sys
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'reproduction/v2'))
try:
    from public_display import apply_public_display
except ImportError:
    apply_public_display=None


class PublicDisplayTests(unittest.TestCase):
    def test_open_font_is_used_after_an_arial_style_was_loaded(self):
        self.assertIsNotNone(apply_public_display,'A verified public-font policy is required')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        plt.rcParams['font.family']=['Arial']
        report=apply_public_display()
        fig,ax=plt.subplots()
        line,=ax.plot([1,2],[0.25,0.75])
        text=ax.set_title('Fixed values',fontweight='bold')
        fig.canvas.draw()
        resolved=font_manager.findfont(text.get_fontproperties(),fallback_to_default=False)
        self.assertIn('DejaVuSans',pathlib.Path(resolved).name)
        self.assertEqual(list(line.get_ydata()),[0.25,0.75])
        buffer=io.StringIO();fig.savefig(buffer,format='svg')
        self.assertNotIn('Arial',buffer.getvalue())
        self.assertEqual(report['display_version'],'public-display-v2-dejavu')
        plt.close(fig)

    def test_missing_fixed_font_is_rejected_without_substitution(self):
        self.assertIsNotNone(apply_public_display,'A verified public-font policy is required')
        with self.assertRaises(FileNotFoundError):
            apply_public_display(font_root=pathlib.Path('/nonexistent_public_font_bundle'))


if __name__=='__main__':unittest.main()
