"""
Main model training for WyCryst+ Framework
"""

from data import *
from featurizer import *
from utils import *
from augmentation import wyckoff_augment
from model import  MPVAE
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau,CSVLogger,LearningRateScheduler,EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib
import warnings
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
warnings.filterwarnings("ignore")

module_dir = 'PATH_TO_DATA'
module_dir2 = 'PATH_TO_TEMP_FILES'

def main():

    # read ternary compound data into dataframe
    print('---------Loading Input Data---------------')
    wyckoff_multiplicity_array, wyckoff_DoF_array = wyckoff_para_loader()
    df_clean = get_input_df()
    scaler_y1 = MinMaxScaler()
    df_clean['formation_energy_per_atom'] = scaler_y1.fit_transform(df_clean[['formation_energy_per_atom']])

    train_df, test_df = train_test_split_bysg(df_clean)    

    # if augment 
    train_df = wyckoff_augment(train_df)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    Crystal_train, sg_train = wyckoff_represent(train_df, 3, 20)
    Crystal_test, sg_test = wyckoff_represent(test_df, 3, 20)
    
    X_train = np.stack(Crystal_train, axis=0)
    X_test = np.stack(Crystal_test, axis=0)
    X2_train = np.stack(sg_train, axis=0)
    X2_test = np.stack(sg_test, axis=0)
    
    y_train = train_df[['formation_energy_per_atom'] + ['ind']].values
    y_test = test_df[['formation_energy_per_atom'] + ['ind']].values

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X2_train = X2_train.astype('float32')
    X2_test = X2_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    
    # print MPVAE input and output shape
    print('---------Printing Input Shape of Source Dataset---------------')
    print('Wyckoff array size:', X_train.shape, X_test.shape, '\nSpace Group array size:', X2_train.shape, X2_test.shape
            , '\nTarget Property array size:', y_train.shape, y_test.shape) 

    # building and training MPVAE
    print('---------Building MPVAE for Pre-training---------------')
    sup_dim = 2
    VAE, encoder, decoder, regression, vae_loss, loss_recon, loss_sg, loss_KL, loss_prop, loss_formula = MPVAE(X_train, y_train, sup_dim)  

    # Callbacks
    # CSV = CSVLogger('temp_files/Wyckoff_log.csv')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=6, min_lr=2e-5)

    VAE.add_loss(vae_loss)
    VAE.add_metric(loss_KL, name='kl_loss', aggregation='mean')
    VAE.add_metric(loss_prop, name='prop_loss', aggregation='mean')
    VAE.add_metric(loss_recon, name='recon_loss', aggregation='mean')
    VAE.add_metric(loss_sg, name='sg_loss', aggregation='mean')
    VAE.add_metric(loss_formula, name='wyckoff_formula_loss', aggregation='mean')
    VAE.add_metric(vae_loss, name='total_loss', aggregation='mean')
    VAE.compile(optimizer=RMSprop(learning_rate=2e-4))

    print('---------Pre-training MPVAE---------------')
    VAE.fit(x=[X_train, X2_train, y_train[:, :sup_dim]], shuffle=True,
            batch_size=512, epochs=55, callbacks=[reduce_lr],
            validation_data=([X_test, X2_test, y_test[:, :sup_dim]], None)) 

    # Save the pre-trained model
    print('---------Saving Pre-trained Model---------------')
    print('trained model weights saved to temp_files/temp_model')
    VAE.save_weights(join(module_dir2, '/vae_models/PreTrained_VAE_weights.h5'))

    # Transfer learning
    print('---------Starting Transfer Learning---------------')
    df_clean = df_clean[df_clean['band_gap'] > 0]
    train_df2 = train_df[train_df['band_gap'] > 0]
    test_df2 = test_df[test_df['band_gap'] > 0]

    scaler_y2 = MinMaxScaler()
    df_clean['band_gap'] = scaler_y2.fit_transform(df_clean[['band_gap']])
    train_df2['band_gap'] = scaler_y2.fit_transform(train_df2[['band_gap']])
    test_df2['band_gap'] = scaler_y2.fit_transform(test_df2[['band_gap']])

    train_df2 = train_df2.reset_index(drop=True)
    test_df2 = test_df2.reset_index(drop=True)

    Crystal_train2, sg_train2 = wyckoff_represent(train_df2, 3, 20)
    Crystal_test2, sg_test2 = wyckoff_represent(test_df2, 3, 20)
    
    X_train2 = np.stack(Crystal_train2, axis=0)
    X_test2 = np.stack(Crystal_test2, axis=0)
    X2_train2 = np.stack(sg_train2, axis=0)
    X2_test2 = np.stack(sg_test2, axis=0)

    y_train2 = train_df2[['formation_energy_per_atom'] + ['band_gap'] + ['ind']].values
    y_test2 = test_df2[['formation_energy_per_atom'] + ['band_gap'] + ['ind']].values

    X_train2 = X_train2.astype('float32')
    X_test2 = X_test2.astype('float32')
    X2_train2 = X2_train2.astype('float32')
    X2_test2 = X2_test2.astype('float32')
    y_train2 = y_train2.astype('float32')
    y_test2 = y_test2.astype('float32')

    # print MPVAE input and output shape
    print('---------Printing Input Shape of Target Dataset---------------')
    print('Wyckoff array size:', X_train2.shape, X_test2.shape, '\nSpace Group array size:', X2_train2.shape, X2_test2.shape
            , '\nTarget Property array size:', y_train2.shape, y_test2.shape) 

    # building and fine-tuning MPVAE
    print('---------Building MPVAE for Fine-tuning---------------')
    VAE, encoder, decoder, regression, vae_loss, loss_recon, loss_sg, loss_KL, loss_prop, loss_formula = MPVAE(X_train2, y_train2, sup_dim=sup_dim, multi=True)

    # Add loss and metrics
    VAE.add_loss(vae_loss)
    VAE.add_metric(loss_KL, name='kl_loss', aggregation='mean')
    VAE.add_metric(loss_prop, name='prop_loss', aggregation='mean')  # include multi-property in loss function
    VAE.add_metric(loss_recon, name='recon_loss', aggregation='mean')
    VAE.add_metric(loss_sg, name='sg_loss', aggregation='mean')
    VAE.add_metric(loss_formula, name='wyckoff_formula_loss', aggregation='mean')
    VAE.add_metric(vae_loss, name='total_loss', aggregation='mean')

    # Load pre-trained weights to the new model structure
    VAE.load_weights(join(module_dir2, '/vae_models/PreTrained_VAE_weights.h5'), by_name=True, skip_mismatch=False)

    # Function to set BatchNormalization layers to inference mode
    def set_batchnorm_inference_mode(model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    # Set BatchNormalization layers in encoder and decoder to inference mode
    set_batchnorm_inference_mode(encoder)
    set_batchnorm_inference_mode(decoder)

    # Freeze only some layers of the encoder
    for layer in encoder.layers[:len(encoder.layers) // 2]:  # freeze the first half of the layers
        layer.trainable = False

    # Validate that specific layers are frozen
    for layer in encoder.layers[:len(encoder.layers) // 2]:  
        assert not layer.trainable, f"Layer {layer.name} is not frozen as expected"
        print(f"Layer {layer.name} is frozen")

    # Validate that BatchNormalization layers are in inference mode (frozen)
    for layer in encoder.layers + decoder.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert not layer.trainable, f"BatchNormalization Layer {layer.name} is not frozen as expected"
            print(f"BatchNormalization Layer {layer.name} is frozen")

    # Reduce learning rate callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=6, min_lr=2e-5)

    # Compile the updated model
    VAE.compile(optimizer=RMSprop(learning_rate=2e-4))

    # Fine-tune the model with the new dataset
    print('---------Fine-tuning the Pre-trained MPVAE---------------')
    VAE.fit(
        x=[X_train2, X2_train2, y_train2[:, :sup_dim]], 
        shuffle=True,
        batch_size=512, 
        epochs=55, 
        callbacks=[reduce_lr],
        validation_data=([X_test2, X2_test2, y_test2[:, :sup_dim]], None)
    ) 

    # Print model performance
    print('---------Printing Result---------------')
    vae_x, vae_sg = VAE.predict([X_test2, X2_test2, y_test2[:, :sup_dim]])

    # Property MAE
    y_result = regression.predict([X_test2, X2_test2])

    # formation energy MAE
    y_result1 = scaler_y1.inverse_transform(y_result[:, :1])
    y_test_p1 = scaler_y1.inverse_transform(y_test2[:, :1])
    MAE_result1 = MAE(y_test_p1, y_result1)
    print('property-learning branch MAE (Ef)', MAE_result1, 'eV/atom')

    # band gap MAE
    y_result2 = scaler_y2.inverse_transform(y_result[:, 1:])
    y_test_p2 = scaler_y2.inverse_transform(y_test2[:, 1:2])
    MAE_result2 = MAE(y_test_p2, y_result2)
    print('property-learning branch MAE (Eg)', MAE_result2, 'eV')

    # Element ACC
    Element = joblib.load(join(module_dir, 'element.pkl'))
    E_v = to_categorical(np.arange(0, len(Element), 1))
    accu = []
    vae_ele = []
    X_ele = [] 
    for i in range(num_ele):
        ele_v = np.argmax(vae_x[:, 0:len(E_v), i], axis=1)
        ele_t = np.argmax(X_test2[:, 0:len(E_v), i], axis=1)
        vae_ele.append(ele_v)
        X_ele.append(ele_t) 
        accu1 = 100 * round(metrics.accuracy_score(ele_v, ele_t), 3)
        accu.append(accu1)
    print('Element accuracy %', accu)

    # Wyckoff ACC
    X_test1 = X_test2
    wyckoff_test = []
    for i in range(vae_x.shape[0]):
        wyckoff_test.append(np.all(np.around(vae_x[i][198:224], decimals=0) == X_test1[i][198:224]))
    wyckoff_test = np.array(wyckoff_test)
    print('Wyckoff Accuracy %', np.mean(wyckoff_test) * 100)

    # SG ACC
    sg_v1 = np.argmax(vae_sg, axis=1)
    sg_t = np.argmax(X2_test2, axis=1)
    accu1 = 100 * round(metrics.accuracy_score(sg_v1, sg_t), 3)
    print('SG Accuracy %', accu1)

    # Save model
    print('---------Saving Final Model---------------')
    print('trained model weights saved to temp_files/temp_model')
    VAE.save_weights(join(module_dir2, '/vae_models/TL_VAE_weights.h5'))
    encoder.save_weights(join(module_dir2, '/vae_models/TL_encoder.h5'))
    decoder.save_weights(join(module_dir2, '/vae_models/TL_decoder.h5'))
    regression.save_weights(join(module_dir2, '/vae_models/TL_regression.h5'))
    joblib.dump(scaler_y1, os.path.join(module_dir2, "/vae_models/tl_scaler_y1.joblib"))
    joblib.dump(scaler_y2, os.path.join(module_dir2, "/vae_models/tl_scaler_y2.joblib"))

    print('---------End---------------')


if __name__ == '__main__':
    main()

