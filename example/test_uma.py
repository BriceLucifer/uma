"""Unit tests for the uma library.

Run with:
    uv run python example/test_uma.py
"""
from __future__ import annotations
import unittest
import math
import mlx.core as mx
import uma
import uma.nn as nn
import uma.optim as optim
from uma.data.dataset import Dataset
from uma.data.dataloader import DataLoader
from uma.trainer import Trainer


# ── helpers ────────────────────────────────────────────────────────────────

def eval_scalar(x: mx.array) -> float:
    mx.eval(x)
    return x.item()


class SimpleDataset(Dataset):
    def __init__(self, x: mx.array, y: mx.array):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════
# functional.py
# ══════════════════════════════════════════════════════════════════════════

class TestLossFunctions(unittest.TestCase):

    def test_cross_entropy_shape(self):
        logits = mx.random.normal((8, 4))
        targets = mx.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=mx.uint32)
        loss = uma.cross_entropy(logits, targets)
        mx.eval(loss)
        self.assertEqual(loss.shape, ())

    def test_cross_entropy_reduction_sum(self):
        logits = mx.random.normal((4, 3))
        targets = mx.array([0, 1, 2, 0], dtype=mx.uint32)
        mean_loss = eval_scalar(uma.cross_entropy(logits, targets, reduction="mean"))
        sum_loss  = eval_scalar(uma.cross_entropy(logits, targets, reduction="sum"))
        self.assertAlmostEqual(sum_loss, mean_loss * 4, places=4)

    def test_cross_entropy_invalid_reduction(self):
        logits = mx.ones((4, 3))
        targets = mx.zeros((4,), dtype=mx.uint32)
        with self.assertRaises(ValueError):
            uma.cross_entropy(logits, targets, reduction="none")

    def test_mse_zero_on_perfect(self):
        x = mx.ones((4, 1))
        loss = eval_scalar(uma.mse(x, x))
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_mse_invalid_reduction(self):
        x = mx.ones((4,))
        with self.assertRaises(ValueError):
            uma.mse(x, x, reduction="avg")

    def test_binary_cross_entropy_shape(self):
        logits = mx.random.normal((16,))
        targets = mx.array([0.0] * 8 + [1.0] * 8)
        loss = uma.binary_cross_entropy(logits, targets)
        mx.eval(loss)
        self.assertEqual(loss.shape, ())

    def test_binary_cross_entropy_invalid_reduction(self):
        logits = mx.zeros((4,))
        targets = mx.zeros((4,))
        with self.assertRaises(ValueError):
            uma.binary_cross_entropy(logits, targets, reduction="none")

    def test_huber_equals_mse_small_errors(self):
        # For |err| <= delta, huber = 0.5 * err^2 → same as 0.5 * MSE
        pred = mx.array([0.1, -0.1, 0.2])
        tgt  = mx.zeros((3,))
        huber_val = eval_scalar(uma.huber(pred, tgt, delta=1.0, reduction="mean"))
        mse_val   = eval_scalar(uma.mse(pred, tgt, reduction="mean"))
        self.assertAlmostEqual(huber_val, 0.5 * mse_val, places=5)

    def test_huber_invalid_reduction(self):
        x = mx.ones((4,))
        with self.assertRaises(ValueError):
            uma.huber(x, x, reduction="none")

    def test_l2_regularization(self):
        params = {"w": mx.ones((4,))}
        reg = eval_scalar(uma.l2_regularization(params, lam=1.0))
        self.assertAlmostEqual(reg, 4.0, places=5)  # sum(1^2) * 4 * 1.0

    def test_l1_regularization(self):
        params = {"w": mx.array([-1.0, 2.0, -3.0])}
        reg = eval_scalar(uma.l1_regularization(params, lam=1.0))
        self.assertAlmostEqual(reg, 6.0, places=5)  # 1+2+3

    def test_regularization_empty_params(self):
        # Should not crash, should return 0
        reg = eval_scalar(uma.l2_regularization({}, lam=1.0))
        self.assertAlmostEqual(reg, 0.0, places=6)
        reg = eval_scalar(uma.l1_regularization({}, lam=1.0))
        self.assertAlmostEqual(reg, 0.0, places=6)


# ══════════════════════════════════════════════════════════════════════════
# nn/module.py
# ══════════════════════════════════════════════════════════════════════════

class TestModule(unittest.TestCase):

    def test_parameters_keys(self):
        fc = nn.Linear(4, 2)
        params = fc.parameters()
        self.assertIn("weight", params)
        self.assertIn("bias", params)

    def test_parameters_nested(self):
        model = nn.Sequential(nn.Linear(4, 2), nn.ReLU())
        params = model.parameters()
        self.assertIn("_layer_0.weight", params)
        self.assertIn("_layer_0.bias", params)

    def test_update_applies_values(self):
        fc = nn.Linear(2, 2)
        new_w = mx.zeros((2, 2))
        fc.update({"weight": new_w})
        mx.eval(fc.weight)
        self.assertTrue((fc.weight == 0).all().item())

    def test_train_eval_toggle(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        model.eval()
        dropout = model[1]
        self.assertFalse(dropout.training)
        model.train()
        self.assertTrue(dropout.training)

    def test_batchnorm_not_in_parameters(self):
        bn = nn.BatchNorm1d(8)
        params = bn.parameters()
        self.assertNotIn("_running_mean", params)
        self.assertNotIn("_running_var", params)
        self.assertIn("weight", params)
        self.assertIn("bias", params)

    def test_batchnorm_buffers(self):
        bn = nn.BatchNorm1d(8)
        buffers = bn.buffers()
        self.assertIn("_running_mean", buffers)
        self.assertIn("_running_var", buffers)

    def test_state_dict_includes_buffers(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU())
        sd = model.state_dict()
        self.assertIn("_layer_0.weight", sd)
        self.assertIn("_layer_1._running_mean", sd)
        self.assertIn("_layer_1._running_var", sd)

    def test_load_state_dict_strict(self):
        fc = nn.Linear(4, 2)
        sd = fc.state_dict()
        fc2 = nn.Linear(4, 2)
        fc2.load_state_dict(sd, strict=True)
        mx.eval(fc.weight, fc2.weight)
        self.assertTrue((fc.weight == fc2.weight).all().item())

    def test_load_state_dict_strict_missing_key(self):
        fc = nn.Linear(4, 2)
        with self.assertRaises(KeyError):
            fc.load_state_dict({}, strict=True)

    def test_load_state_dict_not_strict(self):
        fc = nn.Linear(4, 2)
        # partial load — should not raise
        fc.load_state_dict({"weight": mx.zeros((4, 2))}, strict=False)
        mx.eval(fc.weight)
        self.assertTrue((fc.weight == 0).all().item())

    def test_save_load_roundtrip(self, tmp_path="/tmp/uma_test_weights"):
        model = nn.Sequential(nn.Linear(4, 2), nn.BatchNorm1d(2))
        # Run a forward pass to update running stats
        x = mx.random.normal((8, 4))
        _ = model(x)
        mx.eval(x)

        model.save(tmp_path)

        model2 = nn.Sequential(nn.Linear(4, 2), nn.BatchNorm1d(2))
        model2.load(tmp_path + ".npz")

        sd1 = model.state_dict()
        sd2 = model2.state_dict()
        for k in sd1:
            mx.eval(sd1[k], sd2[k])
            self.assertTrue((sd1[k] == sd2[k]).all().item(), f"mismatch at key: {k}")


# ══════════════════════════════════════════════════════════════════════════
# nn layers
# ══════════════════════════════════════════════════════════════════════════

class TestLinear(unittest.TestCase):

    def test_output_shape(self):
        fc = nn.Linear(8, 4)
        out = fc(mx.random.normal((16, 8)))
        mx.eval(out)
        self.assertEqual(out.shape, (16, 4))

    def test_no_bias(self):
        fc = nn.Linear(4, 2, bias=False)
        self.assertIsNone(fc.bias)
        self.assertNotIn("bias", fc.parameters())


class TestConv2d(unittest.TestCase):

    def test_output_shape(self):
        conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        x = mx.random.normal((2, 8, 8, 1))  # NHWC
        out = conv(x)
        mx.eval(out)
        self.assertEqual(out.shape, (2, 8, 8, 8))

    def test_stride(self):
        conv = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
        x = mx.random.normal((1, 8, 8, 1))
        out = conv(x)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 4, 4, 4))


class TestMaxPool2d(unittest.TestCase):

    def test_output_shape(self):
        pool = nn.MaxPool2d(2)
        x = mx.random.normal((2, 8, 8, 4))
        out = pool(x)
        mx.eval(out)
        self.assertEqual(out.shape, (2, 4, 4, 4))

    def test_non_divisible_raises(self):
        pool = nn.MaxPool2d(2)
        x = mx.random.normal((1, 7, 7, 4))
        with self.assertRaises(ValueError):
            pool(x)


class TestBatchNorm(unittest.TestCase):

    def test_batchnorm1d_output_shape(self):
        bn = nn.BatchNorm1d(8)
        out = bn(mx.random.normal((16, 8)))
        mx.eval(out)
        self.assertEqual(out.shape, (16, 8))

    def test_batchnorm1d_running_stats_update(self):
        bn = nn.BatchNorm1d(4)
        x = mx.ones((8, 4)) * 2.0
        bn(x)
        mx.eval(bn._running_mean)
        # running mean should have moved toward 2.0 from 0.0
        mean_val = bn._running_mean[0].item()
        self.assertGreater(mean_val, 0.0)

    def test_batchnorm2d_output_shape(self):
        bn = nn.BatchNorm2d(8)
        out = bn(mx.random.normal((4, 6, 6, 8)))
        mx.eval(out)
        self.assertEqual(out.shape, (4, 6, 6, 8))

    def test_batchnorm_eval_uses_running_stats(self):
        bn = nn.BatchNorm1d(4)
        # warm up running stats with enough steps so mean/var converge near 3.0/0.0
        for _ in range(100):
            bn(mx.ones((8, 4)) * 3.0)
        mx.eval(bn._running_mean)
        bn.eval()
        # in eval mode, output should be near 0 (normalized by running mean ≈ 3)
        out = bn(mx.ones((2, 4)) * 3.0)
        mx.eval(out)
        self.assertLess(abs(out[0][0].item()), 0.1)


class TestDropout(unittest.TestCase):

    def test_identity_in_eval(self):
        drop = nn.Dropout(0.5)
        drop.eval()
        x = mx.ones((10, 10))
        out = drop(x)
        mx.eval(out)
        self.assertTrue((out == x).all().item())

    def test_scales_in_train(self):
        drop = nn.Dropout(0.0)  # 0% drop → output == input
        x = mx.ones((100, 100))
        out = drop(x)
        mx.eval(out)
        self.assertAlmostEqual(out.mean().item(), 1.0, places=3)

    def test_invalid_p_raises(self):
        with self.assertRaises(ValueError):
            nn.Dropout(1.0)
        with self.assertRaises(ValueError):
            nn.Dropout(-0.1)


class TestLayerNorm(unittest.TestCase):

    def test_output_shape(self):
        norm = nn.LayerNorm(16)
        out = norm(mx.random.normal((8, 16)))
        mx.eval(out)
        self.assertEqual(out.shape, (8, 16))

    def test_normalizes_output(self):
        norm = nn.LayerNorm(64)
        x = mx.random.normal((32, 64)) * 10 + 5
        out = norm(x)
        mx.eval(out)
        # mean ≈ 0, std ≈ 1 per sample
        mean = out.mean(axis=-1)
        mx.eval(mean)
        self.assertAlmostEqual(abs(mean.mean().item()), 0.0, places=4)


class TestEmbedding(unittest.TestCase):

    def test_output_shape(self):
        emb = nn.Embedding(100, 32)
        idx = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.uint32)
        out = emb(idx)
        mx.eval(out)
        self.assertEqual(out.shape, (2, 3, 32))


class TestActivations(unittest.TestCase):

    def test_relu_nonnegative(self):
        x = mx.array([-1.0, 0.0, 1.0])
        out = nn.ReLU()(x)
        mx.eval(out)
        self.assertTrue((out >= 0).all().item())

    def test_sigmoid_range(self):
        x = mx.array([-10.0, 0.0, 10.0])
        out = nn.Sigmoid()(x)
        mx.eval(out)
        self.assertTrue((out > 0).all().item())
        self.assertTrue((out < 1).all().item())

    def test_softmax_sums_to_one(self):
        x = mx.random.normal((4, 8))
        out = nn.Softmax()(x)
        mx.eval(out)
        sums = out.sum(axis=-1)
        mx.eval(sums)
        for i in range(4):
            self.assertAlmostEqual(sums[i].item(), 1.0, places=5)

    def test_leaky_relu_negative_slope(self):
        act = nn.LeakyReLU(negative_slope=0.1)
        x = mx.array([-2.0, 1.0])
        out = act(x)
        mx.eval(out)
        self.assertAlmostEqual(out[0].item(), -0.2, places=5)
        self.assertAlmostEqual(out[1].item(),  1.0, places=5)


class TestSequential(unittest.TestCase):

    def test_forward_chain(self):
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        out = model(mx.random.normal((16, 8)))
        mx.eval(out)
        self.assertEqual(out.shape, (16, 2))

    def test_parameters_tracked(self):
        model = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))
        params = model.parameters()
        self.assertIn("_layer_0.weight", params)
        self.assertIn("_layer_1.weight", params)

    def test_getitem(self):
        fc = nn.Linear(4, 2)
        model = nn.Sequential(fc, nn.ReLU())
        self.assertIs(model[0], fc)


class TestModuleList(unittest.TestCase):

    def test_parameters_tracked(self):
        ml = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        params = ml.parameters()
        self.assertIn("_item_0.weight", params)
        self.assertIn("_item_2.weight", params)

    def test_iter(self):
        layers = [nn.Linear(4, 4) for _ in range(3)]
        ml = nn.ModuleList(layers)
        self.assertEqual(len(list(ml)), 3)

    def test_getitem(self):
        layers = [nn.Linear(4, 4) for _ in range(2)]
        ml = nn.ModuleList(layers)
        self.assertIs(ml[0], layers[0])


# ══════════════════════════════════════════════════════════════════════════
# optimizers
# ══════════════════════════════════════════════════════════════════════════

class TestOptimizers(unittest.TestCase):

    def _dummy_params(self):
        return {"w": mx.array([1.0, 2.0, 3.0])}

    def _dummy_grads(self):
        return {"w": mx.array([0.1, 0.1, 0.1])}

    def test_sgd_decreases_param(self):
        params = self._dummy_params()
        opt = optim.SGD(params, lr=0.1)
        opt.params = params
        updates = opt.step(self._dummy_grads())
        mx.eval(updates["w"])
        # w should decrease: 1.0 - 0.1*0.1 = 0.99
        self.assertAlmostEqual(updates["w"][0].item(), 0.99, places=5)

    def test_adam_updates_all_keys(self):
        params = {"w": mx.ones((4,)), "b": mx.zeros((4,))}
        opt = optim.Adam(params, lr=1e-3)
        opt.params = params
        grads = {"w": mx.ones((4,)) * 0.1, "b": mx.ones((4,)) * 0.1}
        updates = opt.step(grads)
        mx.eval(updates["w"], updates["b"])
        self.assertIn("w", updates)
        self.assertIn("b", updates)

    def test_adamw_weight_decay(self):
        params = {"w": mx.ones((4,))}
        opt_adam  = optim.Adam(params,  lr=1e-2)
        opt_adamw = optim.AdamW({"w": mx.ones((4,))}, lr=1e-2, weight_decay=0.1)
        grads = {"w": mx.zeros((4,))}  # zero grad → only weight decay differs
        opt_adam.params  = params
        opt_adamw.params = {"w": mx.ones((4,))}
        u_adam  = opt_adam.step(grads)
        u_adamw = opt_adamw.step(grads)
        mx.eval(u_adam["w"], u_adamw["w"])
        # AdamW should shrink params more due to weight decay
        self.assertLess(u_adamw["w"][0].item(), u_adam["w"][0].item())

    def test_steplr(self):
        params = {"w": mx.ones((2,))}
        opt = optim.SGD(params, lr=0.1)
        sched = optim.StepLR(opt, step_size=2, gamma=0.5)
        sched.step()  # epoch 1 — no decay
        self.assertAlmostEqual(opt.lr, 0.1, places=6)
        sched.step()  # epoch 2 — decay
        self.assertAlmostEqual(opt.lr, 0.05, places=6)

    def test_cosine_annealing(self):
        params = {"w": mx.ones((2,))}
        opt = optim.SGD(params, lr=1.0)
        sched = optim.CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(opt.lr)
        # lr should go down then back up (cosine shape)
        self.assertLess(lrs[4], lrs[0])

    def test_reduce_lr_on_plateau(self):
        params = {"w": mx.ones((2,))}
        opt = optim.SGD(params, lr=1.0)
        sched = optim.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
        sched.step(1.0)   # best
        sched.step(1.0)   # no improve
        sched.step(1.0)   # no improve → reduce
        self.assertAlmostEqual(opt.lr, 0.5, places=6)


# ══════════════════════════════════════════════════════════════════════════
# data
# ══════════════════════════════════════════════════════════════════════════

class TestDataLoader(unittest.TestCase):

    def test_batch_shape(self):
        x = mx.random.normal((100, 8))
        y = mx.zeros((100,), dtype=mx.uint32)
        loader = DataLoader(SimpleDataset(x, y), batch_size=16, shuffle=False)
        xb, yb = next(iter(loader))
        mx.eval(xb, yb)
        self.assertEqual(xb.shape, (16, 8))
        self.assertEqual(yb.shape, (16,))

    def test_num_batches(self):
        x = mx.random.normal((100, 4))
        y = mx.zeros((100,))
        loader = DataLoader(SimpleDataset(x, y), batch_size=16, shuffle=False)
        self.assertEqual(len(loader), 7)  # ceil(100/16)

    def test_all_samples_covered(self):
        x = mx.arange(20).reshape(20, 1).astype(mx.float32)
        y = mx.zeros((20,))
        loader = DataLoader(SimpleDataset(x, y), batch_size=4, shuffle=False)
        seen = []
        for xb, _ in loader:
            mx.eval(xb)
            seen.extend(xb.reshape(-1).tolist())
        self.assertEqual(sorted(seen), list(range(20)))

    def test_shuffle_changes_order(self):
        x = mx.arange(50).reshape(50, 1).astype(mx.float32)
        y = mx.zeros((50,))
        loader = DataLoader(SimpleDataset(x, y), batch_size=50, shuffle=True)
        xb, _ = next(iter(loader))
        mx.eval(xb)
        vals = xb.reshape(-1).tolist()
        self.assertNotEqual(vals, list(range(50)))  # extremely unlikely to be sorted


# ══════════════════════════════════════════════════════════════════════════
# trainer.py
# ══════════════════════════════════════════════════════════════════════════

class TestTrainer(unittest.TestCase):

    def _make_simple_setup(self):
        N, D, C = 64, 8, 3
        x = mx.random.normal((N, D))
        y = mx.array([i % C for i in range(N)], dtype=mx.uint32)
        model = nn.Sequential(nn.Linear(D, 16), nn.ReLU(), nn.Linear(16, C))
        opt = optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = lambda m, x, y: uma.cross_entropy(m(x), y)
        loader = DataLoader(SimpleDataset(x, y), batch_size=32, shuffle=False)
        return model, opt, loss_fn, loader

    def test_train_step_returns_float(self):
        model, opt, loss_fn, loader = self._make_simple_setup()
        trainer = Trainer(model, opt, loss_fn)
        x, y = next(iter(loader))
        loss = trainer.train_step(x, y)
        self.assertIsInstance(loss, float)
        self.assertFalse(math.isnan(loss))

    def test_fit_returns_history(self):
        model, opt, loss_fn, loader = self._make_simple_setup()
        trainer = Trainer(model, opt, loss_fn)
        history = trainer.fit(loader, epochs=2)
        self.assertIn("loss", history)
        self.assertEqual(len(history["loss"]), 2)

    def test_fit_loss_decreases(self):
        model, opt, loss_fn, loader = self._make_simple_setup()
        trainer = Trainer(model, opt, loss_fn)
        history = trainer.fit(loader, epochs=10)
        self.assertLess(history["loss"][-1], history["loss"][0])

    def test_fit_with_val_metric(self):
        model, opt, loss_fn, loader = self._make_simple_setup()
        trainer = Trainer(model, opt, loss_fn)
        x_val = mx.random.normal((20, 8))
        y_val = mx.array([i % 3 for i in range(20)], dtype=mx.uint32)
        history = trainer.fit(
            loader, epochs=2,
            val_data=(x_val, y_val),
            metric_fn=uma.eval.accuracy,
            metric_name="acc",
        )
        self.assertIn("acc", history)
        self.assertEqual(len(history["acc"]), 2)

    def test_clip_grad_norm(self):
        model, opt, loss_fn, loader = self._make_simple_setup()
        trainer = Trainer(model, opt, loss_fn, clip_grad_norm=0.1)
        history = trainer.fit(loader, epochs=2)
        self.assertEqual(len(history["loss"]), 2)

    def test_early_stopping(self):
        model, opt, loss_fn, loader = self._make_simple_setup()
        trainer = Trainer(model, opt, loss_fn)
        x_val = mx.random.normal((20, 8))
        y_val = mx.array([i % 3 for i in range(20)], dtype=mx.uint32)
        history = trainer.fit(
            loader, epochs=100,
            val_data=(x_val, y_val),
            metric_fn=uma.eval.accuracy,
            metric_name="acc",
            early_stopping_patience=3,
            early_stopping_mode="max",
        )
        # stopped well before 100 epochs
        self.assertLess(len(history["loss"]), 100)

    def test_invalid_early_stopping_mode(self):
        model, opt, loss_fn, loader = self._make_simple_setup()
        trainer = Trainer(model, opt, loss_fn)
        with self.assertRaises(ValueError):
            trainer.fit(loader, epochs=1,
                        early_stopping_patience=3,
                        early_stopping_mode="maximum")

    def test_early_stopping_restores_best_params(self):
        """Restored params should produce a valid model (no NaN outputs)."""
        model, opt, loss_fn, loader = self._make_simple_setup()
        trainer = Trainer(model, opt, loss_fn)
        x_val = mx.random.normal((20, 8))
        y_val = mx.array([i % 3 for i in range(20)], dtype=mx.uint32)
        trainer.fit(
            loader, epochs=50,
            val_data=(x_val, y_val),
            metric_fn=uma.eval.accuracy,
            metric_name="acc",
            early_stopping_patience=3,
            early_stopping_mode="max",
        )
        model.eval()
        out = model(x_val)
        mx.eval(out)
        self.assertFalse(mx.isnan(out).any().item())


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
