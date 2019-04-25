import colorama

from common.train_eval import train
from common.utils import lossfun_one_batch_adversarial

colorama.init()


def main():
    static_params = dict(
        n_samples=5,
        num_epochs=100,
        batch_size=64,
        out_dim=512,
        crop_size=224,
        normalize_output=True,
        normalize_hidden=True,
        dataset='car196',
        l2_weight_decay=1e-4,
        alpha=1,
        learning_rate=1e-3,
        n_classes=10,
        model_save_path="/disk-main/tmp/models",
        epsilon=1e-2
    )
    # train(lossfun_one_batch, static_params, '/disk-main/car196', "/disk-main/logs/")
    data_path = '/disk-main/car196'
    log_path = "/disk-main/tmp/logs/"

    train(lossfun_one_batch_adversarial, static_params, data_path, log_path)


if __name__ == '__main__':
    main()
