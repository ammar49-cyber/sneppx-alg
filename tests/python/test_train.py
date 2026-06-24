import numpy as np
from arix_algo import Model, Trainer


def test_train_step():
    model = Model({'input_dim': 8, 'output_dim': 8})
    trainer = Trainer(model, {'learning_rate': 0.01, 'batch_size': 4})
    x = np.random.randn(1, 4, 8).astype(np.float32)
    y = np.random.randn(1, 4, 8).astype(np.float32)
    loss = trainer.train_step(x, y)
    assert loss >= 0, f"Loss should be non-negative, got {loss}"
    assert np.isfinite(loss), f"Loss should be finite, got {loss}"
    print(f"  test_train_step loss={loss:.6f} PASS")


if __name__ == '__main__':
    test_train_step()
    print("All train tests passed.")
