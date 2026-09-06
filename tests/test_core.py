import unittest

import numpy as np

from src.absolute_pe import add_positional_encoding, sinusoidal_pe
from src.extract import random_qk, select_head
from src.rope import (
    apply_rope,
    attention_scores,
    l2_norms,
    pair_frequencies,
    pair_xy,
    rotate_pair,
    row_cosine,
)


class RopeTests(unittest.TestCase):
    def test_even_dim_required(self):
        with self.assertRaises(ValueError):
            apply_rope(np.ones((4, 5)))

    def test_position_zero_is_identity(self):
        x = np.random.default_rng(0).standard_normal((8, 16))
        y = apply_rope(x, base=10000.0, style="interleaved")
        np.testing.assert_allclose(y[0], x[0], atol=1e-12)

    def test_preserves_l2(self):
        x = np.random.default_rng(1).standard_normal((12, 32))
        for style in ("interleaved", "llama"):
            y = apply_rope(x, style=style)
            np.testing.assert_allclose(l2_norms(x), l2_norms(y), atol=1e-10)

    def test_matches_scratch_even_odd(self):
        x = np.random.default_rng(2).standard_normal((6, 10))
        seq, dim = x.shape
        positions = np.arange(seq)[:, None]
        pair_indices = np.arange(0, dim, 2)
        inv_freq = 1 / (10000 ** (pair_indices / dim))
        angles = positions * inv_freq
        cos, sin = np.cos(angles), np.sin(angles)
        expected = np.empty_like(x)
        expected[:, 0::2] = x[:, 0::2] * cos - x[:, 1::2] * sin
        expected[:, 1::2] = x[:, 0::2] * sin + x[:, 1::2] * cos
        np.testing.assert_allclose(apply_rope(x, style="interleaved"), expected)

    def test_relative_angle_depends_on_offset(self):
        omega = pair_frequencies(8, base=10000.0)[0]
        x = np.zeros((5, 8))
        x[:, 0] = 1.0
        y = apply_rope(x, style="interleaved")
        xm, ym = pair_xy(y, 3, 0)
        xn, yn = pair_xy(y, 1, 0)
        a_m = np.arctan2(ym, xm)
        a_n = np.arctan2(yn, xn)
        self.assertAlmostEqual(a_m - a_n, (3 - 1) * omega, places=10)

    def test_rotate_pair_and_batched_heads(self):
        even, odd = np.array([1.0]), np.array([0.0])
        re, ro = rotate_pair(even, odd, np.array([np.pi / 2]))
        np.testing.assert_allclose(re, 0.0, atol=1e-12)
        np.testing.assert_allclose(ro, 1.0, atol=1e-12)
        x = np.random.default_rng(3).standard_normal((4, 7, 16))
        y = apply_rope(x, style="llama")
        self.assertEqual(y.shape, x.shape)
        np.testing.assert_allclose(l2_norms(x), l2_norms(y), atol=1e-10)

    def test_cosine_identity_at_zero(self):
        x = np.random.default_rng(4).standard_normal((5, 12))
        y = apply_rope(x)
        np.testing.assert_allclose(row_cosine(x, y)[0], 1.0, atol=1e-10)

    def test_attention_scores_shape(self):
        q = np.ones((3, 4))
        k = np.ones((3, 4))
        s = attention_scores(q, k)
        self.assertEqual(s.shape, (3, 3))


class AbsolutePeTests(unittest.TestCase):
    def test_matches_loop_formula(self):
        seq, dim, base = 4, 5, 10000.0
        pe = sinusoidal_pe(seq, dim, base=base)
        for k in range(seq):
            for i in range(dim):
                if i % 2 == 0:
                    expected = np.sin(k / base ** (i / dim))
                else:
                    expected = np.cos(k / base ** ((i - 1) / dim))
                self.assertAlmostEqual(pe[k, i], expected, places=12)

    def test_add_changes_norm(self):
        emb = np.random.default_rng(0).standard_normal((6, 8))
        pe, combined = add_positional_encoding(emb)
        self.assertEqual(pe.shape, emb.shape)
        self.assertFalse(np.allclose(l2_norms(emb), l2_norms(combined)))


class ExtractHelpersTests(unittest.TestCase):
    def test_random_qk(self):
        data = random_qk(seq_len=8, dim=16, seed=0, base=10000.0)
        self.assertEqual(data["q_before"].shape, (8, 16))
        np.testing.assert_allclose(l2_norms(data["q_before"]), l2_norms(data["q_after"]), atol=1e-10)
        slice_q = select_head(data["q_before"], 99)
        self.assertEqual(slice_q.shape, (8, 16))
        from src.plots import bulk_before_after_delta, attention_heatmaps

        fig = bulk_before_after_delta(data["q_before"], data["q_after"])
        self.assertTrue(len(fig.data) >= 1)
        s = attention_scores(data["q_before"], data["k_before"])
        attention_heatmaps(s, s)


if __name__ == "__main__":
    unittest.main()
