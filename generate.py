"""
Generator for WyCryst+ Framework
"""

from data import *
from featurizer import *
from utils import *
from model import MPVAE
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau,CSVLogger,LearningRateScheduler,EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import re
from os.path import abspath, dirname, join
from tqdm import tqdm
import itertools
import warnings
import time
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
warnings.filterwarnings("ignore")

module_dir = 'PATH_TO_DATA'
module_dir2 = 'PATH_TO_TEMP_FILES'

def main():

    print('---------Building Input Data---------------')
    wyckoff_multiplicity_array, wyckoff_DoF_array = wyckoff_para_loader()
    df = get_input_df()
    df = df[df['band_gap'] > 0]

    # Set your generation target (reference materials)  
    # (e.g.) Target: Ef < -0.5 & Eg ~= 1.5
    df_target_pro = df[df['formation_energy_per_atom'] < -0.5]
    df_target_pro = df[(df['band_gap'] > 1.45) & (df['band_gap'] < 1.55)]
    df_target_pro.rename(columns={'spacegroup.crystal_system':'spacegroup_crystal_system'
                                        ,'spacegroup.number':'spacegroup_number'}, inplace=True)
    df_target_pro = df_target_pro.reset_index()

    st = time.time()
    target_C,target_sg = wyckoff_represent(df_target_pro,3,20)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    a = np.stack(target_C,axis=0)
    a.shape
    X = a
    X2 = np.stack(target_sg,axis=0)
    X2=X2[:,:,0]

    print(X.shape,X2.shape)

    Y = df_target_pro[['formation_energy_per_atom']+['band_gap']+['index']].values

    # print reference size
    print('---------Printing Reference Size---------------')
    print('Wyckoff array size:',X.shape, '\nSpace Group array size:', X2.shape
            , '\nTarget Property array size:', Y.shape)

    # loading trained model 
    print('---------Loading Trained MPVAE---------------')
    sup_dim = 2
    VAE, encoder, decoder, regression, vae_loss, loss_recon, loss_sg, loss_KL, loss_prop, loss_formula = MPVAE(X_train2, y_train2, sup_dim=sup_dim, multi=True)

    # Add loss and metrics
    VAE.add_loss(vae_loss)
    VAE.add_metric(loss_KL, name='kl_loss', aggregation='mean')
    VAE.add_metric(loss_prop, name='prop_loss', aggregation='mean')  # include multi-property in loss function
    VAE.add_metric(loss_recon, name='recon_loss', aggregation='mean')
    VAE.add_metric(loss_sg, name='sg_loss', aggregation='mean')
    VAE.add_metric(loss_formula, name='wyckoff_formula_loss', aggregation='mean')
    VAE.add_metric(vae_loss, name='total_loss', aggregation='mean')

    encoder.load_weights(os.path.join(module_dir2,'vae_models/TL_encoder.h5'))
    decoder.load_weights(os.path.join(module_dir2,'vae_models/TL_decoder.h5'))
    regression.load_weights(os.path.join(module_dir2,'vae_models/TL_regression.h5'))

    scaler_y1 = joblib.load(os.path.join(module_dir2,"vae_models/tl_scaler_y1.joblib"))
    scaler_y2 = joblib.load(os.path.join(module_dir2,"vae_models/tl_scaler_y2.joblib"))

    # sampling from latent space
    print('---------Sampling Result---------------')
    # Sampling the latent space and perform inverse design

    # Set number of purturbing instances around each compound
    Nperturb = 20

    # Set local purturbation (Lp) scale
    Lp_scale = 1.4

    #set random state
    np.random.seed(124123)

    # Sample (Lp)
    sample_latent = encoder.predict([X,X2])
    print(sample_latent.shape)
    samples = sample_latent
    samples = np.tile(samples, (Nperturb, 1))
    gaussian_noise = np.random.normal(0, 1, samples.shape)
    samples = samples + gaussian_noise * Lp_scale
    wyckoff_designs = decoder.predict(samples, verbose=1)
    print(samples.shape)

    sample_x = wyckoff_designs[0]
    sample_x[sample_x<0.1] =0
    sample_sg = wyckoff_designs[1]
    sample_sg[sample_sg<0.1] =0
    print(sample_x.shape)
    print(sample_sg.shape)

    Element = joblib.load(os.path.join(module_dir,'element.pkl'))
    E_v = to_categorical(np.arange(0, len(Element), 1))
    sample_ele = []
    for i in range(num_ele):
        ele_v = np.argmax(sample_x[:,0:len(E_v),i],axis=1)
        sample_ele.append(ele_v)

    sg_s = np.argmax(sample_sg,axis=1)

    with open(os.path.join(module_dir, "wyckoff-position-multiplicities.json")) as file:
        wyckoff_multiplicity_dict = json.load(file)
    with open(os.path.join(module_dir, "wyckoff-position-params.json")) as file:
        param_dict = json.load(file)

    # create dataframe for sampled data
    df_sample = pd.DataFrame(columns=['reconstructed_formula','reconstructed_ratio','reconstructed_wyckoff', 'reconstructed_sg',
                                       'reconstructed_DoF','str_wyckoff'])

    for i in tqdm(range(sample_x.shape[0])):
        # calculate reconstructed ratio and DoF
        recon_ratio=np.matmul(np.reshape(wyckoff_multiplicity_array[sg_s[i]+1],(1,26)),np.round(sample_x[i][198:],decimals=0))
        recon_ratio = np.reshape(recon_ratio,(3,))

        recon_DoF = np.matmul(np.reshape(wyckoff_DoF_array[sg_s[i]+1],(1,26)),np.round(sample_x[i][198:],decimals=0))
        recon_DoF = np.sum(np.reshape(recon_DoF,(3,)))

        # reconstruct formula
        formula =''
        elements = []
        for j in range(3):
            formula +=Element[sample_ele[j][i]]
            elements.append(Element[sample_ele[j][i]])
            formula +=str(int(recon_ratio[j]))
        recon_sg = sg_s[i]+1

        # check if compound is valid
        if np.any(recon_ratio==0):
            continue

        recon_wyckoff_dic = get_reconstructed_wyckoff(np.round(sample_x[i,198:,:]),elements,sg_s[i]+1)
        all_wyckoff = list(itertools.chain(recon_wyckoff_dic[elements[0]],recon_wyckoff_dic[elements[1]],recon_wyckoff_dic[elements[2]]))
        common_wyckoff = [l for l in all_wyckoff if all_wyckoff.count(l) > 1]
        DoF_check = True
        for k in common_wyckoff:
            if param_dict[str(recon_sg)][re.sub(r'[^a-zA-Z]', '', k)]==0:
                DoF_check=False
                break
        if DoF_check==False:
            continue

        # add info to dataframe
        df_sample.loc[i, 'reconstructed_sg'] = recon_sg
        df_sample.loc[i, 'reconstructed_formula']=formula
        df_sample.at[i, 'reconstructed_ratio']=recon_ratio.tolist()
        df_sample['reconstructed_DoF'].at[i]=recon_DoF
        df_sample['reconstructed_wyckoff'].at[i] = recon_wyckoff_dic
        df_sample['str_wyckoff'].at[i]=str(df_sample.loc[i, 'reconstructed_wyckoff'])
        df_sample.at[i, 'oxid_test']= neutral_test(formula)

    df_sample['predicted_formation_energy']=get_reconstructed_property(df_sample, regression, scaler_y1)[:,0]
    df_sample['predicted_band_gap']=get_reconstructed_property(df_sample, regression, scaler_y2)[:,1]
    df_sample['predicted_SC']=get_reconstructed_SC(df_sample)
    df_sample['reduced_formula'] = [Composition(i).reduced_formula for i in df_sample['reconstructed_formula']]

    # saving novel & charge-neutral sampled wyckoff genes with synthesizability score > 0.5
    print('---------Saving Result---------------')
    df_sample1 = df_sample.loc[-df_sample['reduced_formula'].isin(df_all['pretty_formula'].to_list())]
    df_sample1 = df_sample1[(df_sample1['predicted_SC']>0.5)&(df_sample1['oxid_test']==True)].groupby(by='reduced_formula',group_keys=False).first().sort_values(by=['reconstructed_sg','reconstructed_DoF','predicted_formation_energy'],ascending = [False,True, True])
    file_path = os.path.join(module_dir2, "sampled_wyckoff_genes.csv")
    df_sample1.to_csv(file_path, index=True) 
    print('---------End---------------')

if __name__ == "__main__":
    main()





