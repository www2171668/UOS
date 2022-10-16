
"""Test for alf.algorithms.text_codec."""

import tensorflow as tf
import alf.algorithms.text_codec as text_codec


class TestTextEncodeDecodeNetwork(tf.test.TestCase):
    def test_encode_decode(self):
        vocab_size = 1000
        seq_len = 5
        embed_size = 50
        encoder_lstm_size = 100
        code_len = encoder_lstm_size
        decoder_lstm_size = 100

        encoder = text_codec.TextEncodeNetwork(vocab_size, seq_len, embed_size,
                                               encoder_lstm_size)
        decoder = text_codec.TextDecodeNetwork(vocab_size, code_len, seq_len,
                                               decoder_lstm_size)

        s0 = tf.constant([1, 3, 2])
        s1 = tf.constant([4])

        batch = [s0, s1]
        batch_size = len(batch)

        input = tf.keras.preprocessing.sequence.pad_sequences(
            batch, padding='post', maxlen=seq_len)

        encoded, _ = encoder(input)
        self.assertEqual((batch_size, encoder_lstm_size), encoded.shape)

        decoded, _ = decoder(encoded)
        self.assertEqual((batch_size, seq_len, vocab_size), decoded.shape)


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    tf.test.main()
