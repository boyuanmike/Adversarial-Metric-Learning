import colorama

# from common.train_eval import train
# from common.utils import lossfun_one_batch, lossfun_one_batch_baseline
from common.train_eval_2gen import train
from common.utils_2gen import lossfun_one_batch,lossfun_one_batch_retain

colorama.init()
path = '/disk-main/'
# ds = 'CUB_200_2011'
ds = 'car196'

def main():
    static_params = dict(
        n_samples=5,
        num_epochs=100,
        batch_size=64,
        out_dim=512,
        crop_size=224,
        normalize_output=True,
        normalize_hidden=False,
        distance_type='euclidean',
        dataset='car196',
        l2_weight_decay=1e-4,
        alpha=1,
        learning_rate=1e-3,
        n_classes=10,
        neg_gen_epoch=0,
        model_save_path=path + "models"
    )


    data_path = path + ds

    log_path = "logs/"+ ds +'_retain/'

    train(lossfun_one_batch, static_params, data_path, log_path)

    # train(lossfun_one_batch_baseline, static_params, data_path, log_path)



if __name__ == '__main__':
    main()
