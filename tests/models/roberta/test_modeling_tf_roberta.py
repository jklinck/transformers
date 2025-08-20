# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tests/models/roberta/test_modeling_tf_roberta.py
# Minimal, surgical tests to guard that TFRobertaEmbeddings reads padding_idx
# from config.pad_token_id (not hardcoded to 1), and that a model forward with
# inputs_embeds works when pad_token_id != 1.

import unittest

from transformers import is_tf_available, RobertaConfig
from transformers.testing_utils import require_tf

if is_tf_available():
    import tensorflow as tf
    from transformers.models.roberta.modeling_tf_roberta import (
        TFRobertaEmbeddings,
        TFRobertaModel,
    )


@require_tf
class TFRobertaPaddingIdxBehaviorTest(unittest.TestCase):
    def test_padding_idx_is_read_from_config(self):
        # Directly checks your change: self.padding_idx == config.pad_token_id
        cfg = RobertaConfig(
            vocab_size=99,
            pad_token_id=7,           # non-1 pad to catch regressions
            max_position_embeddings=8,
            hidden_size=4,
            type_vocab_size=2,
        )
        emb = TFRobertaEmbeddings(cfg)
        self.assertEqual(emb.padding_idx, 7)

    def test_create_position_ids_respects_pad_token_id(self):
        # Verifies position ids produced from input_ids keep pads at pad_id and
        # start non-pad tokens at pad_id + 1.
        cfg = RobertaConfig(
            vocab_size=99,
            pad_token_id=2,           # non-1 pad to ensure change is honored
            max_position_embeddings=16,
            hidden_size=8,
            type_vocab_size=2,
        )
        emb = TFRobertaEmbeddings(cfg)

        # Pads (2) at positions 0 and 3 in row 0; 0 and 1 in row 1.
        input_ids = tf.constant(
            [[2, 5, 6, 2],
             [2, 2, 7, 8]],
            dtype=tf.int32,
        )

        pos = emb.create_position_ids_from_input_ids(input_ids)

        expected = tf.constant(
            [[2, 3, 4, 2],
             [2, 2, 3, 4]],
            dtype=tf.int32,
        )
        self.assertListEqual(pos.numpy().tolist(), expected.numpy().tolist())

    def test_create_position_ids_respects_pad_token_id_zero(self):
        # Pad=0 is common; non-pad tokens should start at 1.
        cfg = RobertaConfig(
            vocab_size=99,
            pad_token_id=0,           # pad = 0
            max_position_embeddings=16,
            hidden_size=8,
            type_vocab_size=2,
        )
        emb = TFRobertaEmbeddings(cfg)

        input_ids = tf.constant(
            [[0, 5, 6, 0],
             [0, 0, 7, 8]],
            dtype=tf.int32,
        )

        pos = emb.create_position_ids_from_input_ids(input_ids)

        expected = tf.constant(
            [[0, 1, 2, 0],
             [0, 0, 1, 2]],
            dtype=tf.int32,
        )
        self.assertListEqual(pos.numpy().tolist(), expected.numpy().tolist())

    def test_create_position_ids_all_padding_rows(self):
        # Entire sequences of padding should remain equal to pad_token_id.
        cfg = RobertaConfig(
            vocab_size=99,
            pad_token_id=5,
            max_position_embeddings=16,
            hidden_size=8,
            type_vocab_size=2,
        )
        emb = TFRobertaEmbeddings(cfg)

        input_ids = tf.constant(
            [[5, 5, 5, 5],
             [5, 5, 5, 5]],
            dtype=tf.int32,
        )

        pos = emb.create_position_ids_from_input_ids(input_ids)
        expected = tf.constant(
            [[5, 5, 5, 5],
             [5, 5, 5, 5]],
            dtype=tf.int32,
        )
        self.assertListEqual(pos.numpy().tolist(), expected.numpy().tolist())

    def test_create_position_ids_from_inputs_embeds_respects_pad_token_id(self):
        # Some TF RoBERTa versions expose a helper for inputs_embeds; others
        # build these ids internally. Skip when the helper isn't present.
        cfg = RobertaConfig(
            vocab_size=99,
            pad_token_id=7,
            max_position_embeddings=32,
            hidden_size=8,
            type_vocab_size=2,
        )
        emb = TFRobertaEmbeddings(cfg)

        if not hasattr(emb, "create_position_ids_from_inputs_embeds"):
            self.skipTest("TFRobertaEmbeddings has no create_position_ids_from_inputs_embeds in this version")

        batch, seq = 2, 4
        inputs_embeds = tf.random.uniform([batch, seq, cfg.hidden_size], dtype=tf.float32)

        pos = emb.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length=0)

        expected_row = tf.range(
            start=cfg.pad_token_id + 1,
            limit=cfg.pad_token_id + 1 + seq,
            dtype=tf.int32,
        )
        expected = tf.broadcast_to(expected_row, [batch, seq])
        self.assertListEqual(pos.numpy().tolist(), expected.numpy().tolist())

    def test_create_position_ids_from_inputs_embeds_respects_past_length(self):
        cfg = RobertaConfig(
            vocab_size=99,
            pad_token_id=7,
            max_position_embeddings=64,
            hidden_size=8,
            type_vocab_size=2,
        )
        emb = TFRobertaEmbeddings(cfg)

        if not hasattr(emb, "create_position_ids_from_inputs_embeds"):
            self.skipTest("TFRobertaEmbeddings has no create_position_ids_from_inputs_embeds in this version")

        batch, seq = 2, 4
        inputs_embeds = tf.random.uniform([batch, seq, cfg.hidden_size], dtype=tf.float32)

        past = 3
        pos = emb.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length=past)

        expected_row = tf.range(
            start=cfg.pad_token_id + 1 + past,
            limit=cfg.pad_token_id + 1 + past + seq,
            dtype=tf.int32,
        )
        expected = tf.broadcast_to(expected_row, [batch, seq])
        self.assertListEqual(pos.numpy().tolist(), expected.numpy().tolist())

    def test_model_forward_inputs_embeds_smoke_non1_pad(self):
        # Smoke test that the *model* forward path with inputs_embeds works
        # when pad_token_id != 1 (covers the internal position-id path).
        cfg = RobertaConfig(
            vocab_size=99,
            pad_token_id=7,               # non-1 pad
            max_position_embeddings=32,   # must exceed pad_id + 1 + seq_len
            hidden_size=8,
            num_attention_heads=2,        # divisors of hidden_size
            intermediate_size=16,
            num_hidden_layers=1,
            type_vocab_size=2,
        )
        model = TFRobertaModel(cfg)

        batch, seq = 2, 4
        inputs_embeds = tf.random.uniform([batch, seq, cfg.hidden_size], dtype=tf.float32)
        attention_mask = tf.constant([[1, 1, 1, 0],
                                      [1, 1, 1, 1]], dtype=tf.int32)

        # Pass input_ids explicitly as None to satisfy Keras' call signature.
        outputs = model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            training=False,
        )
        self.assertEqual(tuple(outputs.last_hidden_state.shape), (batch, seq, cfg.hidden_size))


if __name__ == "__main__":
    unittest.main()
