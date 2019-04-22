import colorama

from common.train_eval import train
from common.utils import lossfun_one_batch

colorama.init()


def main():
    static_params = dict(
        n_samples=5,
        num_epochs=100,
        batch_size=64,
        out_dim=512,
        crop_size=224,
        normalize_output=True,
        distance_type='euclidean',
        dataset='car196',
        l2_weight_decay=1e-4,
        alpha=1,
        learning_rate=1e-3,
        n_classes=98,
        neg_gen_epoch=0,
        model_save_path="/mnt/tmp/data/models"
    )
    # train(lossfun_one_batch, static_params, '/disk-main/car196', "/disk-main/logs/")
    data_path = '/mnt/tmp/data/car196'
    log_path = "/mnt/tmp/data/logs/"
    train(lossfun_one_batch, static_params, data_path, log_path)


if __name__ == '__main__':
    main()
