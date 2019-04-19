import colorama

from common.train_eval import train
from common.utils import lossfun_one_batch

colorama.init()


def main():
    static_params = dict(
        n_samples=25,
        num_epochs=100,
        batch_size=64,
        out_dim=512,
        crop_size=224,
        normalize_output=True,
        distance_type='euclidean',
        dataset='car196',
        l2_weight_decay=1e-4,
        alpha=10.0,
        learning_rate=1e-4,
        n_classes=10
    )
    train(lossfun_one_batch, static_params, '/disk-main/car196')


if __name__ == '__main__':
    main()
