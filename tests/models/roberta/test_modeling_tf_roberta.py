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

import unittest

from transformers import is_tf_available, RobertaConfig
from transformers.testing_utils import require_tf

if is_tf_available():
    import tensorflow as tf
    from transformers.models.roberta.modeling_tf_roberta import TFRobertaEmbeddings

@require_tf
class TFRobertaPaddingIdxBehaviorTest(unittest.TestCase):
    # ... keep your previous tests here ...

    def test_create_position_ids_from_inputs_embeds_respects_pad_token_id(self):
        # pad != 1 to catch regressions if padding_idx is ever hardcoded again
        cfg = RobertaConfig(
            vocab_size=99,
            pad_token_id=7,
            max_position_embeddings=32,
            hidden_size=8,
            type_vocab_size=2,
        )
        emb = TFRobertaEmbeddings(cfg)

        # shape: (batch=2, seq=4, hidden)
        inputs_embeds = tf.random.uniform([2, 4, cfg.hidden_size], dtype=tf.float32)

        pos = emb.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length=0)
        # When using inputs_embeds, positions start at pad_id+1 and increase by 1
        expected = tf.constant([[8, 9, 10, 11],
                                [8, 9, 10, 11]], dtype=tf.int32)

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

        inputs_embeds = tf.random.uniform([2, 4, cfg.hidden_size], dtype=tf.float32)

        pos = emb.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length=3)
        # Offset by past length: (pad_id + 1) + 3 = 11 to 14
        expected = tf.constant([[11, 12, 13, 14],
                                [11, 12, 13, 14]], dtype=tf.int32)

        self.assertListEqual(pos.numpy().tolist(), expected.numpy().tolist())
