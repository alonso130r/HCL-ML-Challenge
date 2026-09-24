import json
import unittest

import train_qwen_qlora as training


class TrainingTests(unittest.TestCase):
    def test_early_stopping_tracks_best_and_counts_plateaus(self):
        stop = training.EarlyStopping(patience=2, min_delta=0.01)
        self.assertEqual(stop.update(0.5), (True, False))
        self.assertEqual(stop.update(0.505), (True, False))
        self.assertEqual(stop.update(0.49), (False, True))
        self.assertEqual(stop.best, 0.505)

    def test_significant_improvement_resets_patience(self):
        stop = training.EarlyStopping(patience=2, min_delta=0.01)
        stop.update(0.5)
        stop.update(0.49)
        self.assertEqual(stop.update(0.52), (True, False))
        self.assertEqual(stop.update(0.52), (False, False))

    def test_examples_have_causal_context_and_only_target_label(self):
        rows = [dict(Dialogue_ID='1', Utterance_ID=str(i), Speaker='A',
                     Utterance=f'turn {i}', Emotion=label)
                for i, label in enumerate(['joy', 'anger', 'sadness'])]
        examples = training.build_messages(list(reversed(rows)), 1)
        self.assertEqual(json.loads(examples[1]['messages'][-1]['content']), {'emotion': 'anger'})
        prompt = examples[1]['messages'][0]['content']
        self.assertIn('turn 0', prompt)
        self.assertIn('turn 1', prompt)
        self.assertNotIn('turn 2', prompt)
        self.assertNotIn('attached media', prompt)
        with self.assertRaises(ValueError):
            training.build_messages(rows + [rows[0]], 1)

    def test_config_uses_masked_lora_and_safe_memory_defaults(self):
        args = training.parse_args([])
        config = training.training_config(args)
        self.assertEqual(config['fine_tune_type'], 'lora')
        self.assertTrue(config['mask_prompt'])
        self.assertTrue(config['grad_checkpoint'])
        self.assertFalse(config['test'])
        self.assertEqual(config['batch_size'], 1)
        self.assertIn('linear_attn.in_proj_qkv', config['lora_parameters']['keys'])


if __name__ == '__main__':
    unittest.main()
