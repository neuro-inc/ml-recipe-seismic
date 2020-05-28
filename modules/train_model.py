from model1 import uResNet34
from model_utils import get_scheduler
from data_gen1 import SliceIterator, primary_transform
from const import (
    model_log_dir, dumps_dir, model_dir,
    slices_dir, wells, crossval_dict, norm_dict_path
)
import numpy as np
import re
import pickle
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras import backend as K


def get_train_test_split(slices_dir, crossval_dict):
    all_slices = list(slices_dir.glob('*.pkl'))
    cv_dataset = []
    for d in crossval_dict:
        cv_dataset.append(
            {
                'train':  [f for f in all_slices if re.match('.+_.+_(.+)', f.stem)[1] in d['train']],
                'test':  [f for f in all_slices if re.match('.+_.+_(.+)', f.stem)[1] in d['test']]
            }
        )
    return cv_dataset

# Definitions of global constants and train/test splits.
cv_dataset = get_train_test_split(slices_dir, crossval_dict)


# Model class shows the decoder type of our hourglass-shaped architecture.
model_class = uResNet34
image_size = (480, 512)

# Training parameters: learning rate scheduler, logger, number of epochs for training, and batch size.
lr_steps = {0: 1e-4, 100: 5e-5}
scheduler = get_scheduler(lr_steps)
csv_logger = CSVLogger(model_log_dir / r'training-sz{}x{}.log'.format(*image_size), append=True)

nb_epoch = 50
batch_size = 1

# Reading normalization parameters for different carotage types.
with open(norm_dict_path, 'rb') as f:
    norm_dict = pickle.load(f)


def train(c_types, model_weights, norm, train_slices, test_slices, suff):
    """Main function for training the model"""
    K.clear_session()
    model_checkpoint = dumps_dir / r'{}.{}.sz{}x{}.{}.{{epoch:02d}}-{{val_masked_correlation:.2f}}.hdf5'. \
        format(model_class.__name__, '-'.join(c_types), *image_size, suff)
    model_checkpoint = str(model_checkpoint)

    train_gen = SliceIterator(train_slices, c_types, image_size, transform_fun=primary_transform, norm=norm, aug=True,
                              batch_size=batch_size, shuffle=True, verbose=False, output_ids=False, blur=False)
    test_gen = SliceIterator(test_slices, c_types, image_size, transform_fun=primary_transform, norm=norm, aug=False,
                             batch_size=batch_size, shuffle=False, verbose=False, output_ids=False)
    callbacks = [ModelCheckpoint(model_checkpoint, monitor='val_masked_correlation', mode='max', save_best_only=True),
                 LearningRateScheduler(scheduler),
                 csv_logger]
    model = model_class(input_size=image_size, weights=model_weights, n_carotage=len(c_types))
    model.fit_generator(
        train_gen,
        steps_per_epoch=int(np.ceil(len(train_slices) / batch_size)),
        epochs=nb_epoch,
        validation_data=test_gen,
        validation_steps=int(np.ceil(len(test_slices) / batch_size)),
        workers=4,
        callbacks=callbacks
    )


if __name__ == '__main__':

    # перечень каротажей и инициирующих весов для тренировки; None - с нуля
    pretrain = {
        'Gamma_Ray': None,
        'Porosity': None,
        'Density': None,
        'P_Impedance': None,
    }

    nfolds = len(cv_dataset)
    for fold in range(nfolds):  # номер фолда
        print(f'fold {fold + 1} of {nfolds}')
        train_slices = cv_dataset[fold]['train']
        test_slices = cv_dataset[fold]['test']
        suff = f'smtd_{fold}'

        # For each carotage type, read model weights and train the model
        for k, v in pretrain.items():
            print(f'Carotage {k}, {v}')
            c_types = [k]
            norm = [(norm_dict[c]['mean'], norm_dict[c]['std']) for c in ['seismic'] + c_types]
            model_weights = model_dir / v if v else None
            train(c_types, model_weights, norm, train_slices, test_slices, suff)
