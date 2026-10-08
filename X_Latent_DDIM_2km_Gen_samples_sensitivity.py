# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 13:31:16 2025

@author: 59278
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import FUNC_read_data as read
import FUNC_analyse_data as analyse
import tensorflow as tf
import X_Latent_DDIM_2km_UNet_final as DDIM_UNet
import X_VAE_model_2km_new as VAE
import copy

#设置扩散步长
batchs=32
timestep=500
denoise_step=5
Model = 'IR' #'IR','Full'
#%%
test_files=['2020_Himawari_cloudsat_128_cloud_SAZ']
data_path='/public/home/Xiongqq/Data_Train'
save_name=f'CRRLDM-{Model}'
save_path=f'/public/home/Xiongqq/CRR-LDM_result/CRRLDM-{Model}_test_2020'
#%%
tar_name=['cloud_scenario','Radar_Reflectivity']
if Model == 'IR':
    var_name=[#'albedo_01','albedo_02','albedo_03','albedo_04','albedo_05','albedo_06','tbb_07','SOZ',
              'tbb_08','tbb_09','tbb_10','tbb_11','tbb_12','tbb_13','tbb_14',
              'tbb_15','tbb_16']
    var_combin=[#('albedo_01','albedo_02','albedo_03'),('albedo_04','albedo_05','albedo_06'),'tbb_07','SOZ',
              ('tbb_08','tbb_09','tbb_10'),('tbb_13','tbb_14','tbb_15')]
    DDIM_weight='/public/home/Xiongqq/weight_use/tf214_v2/ddim_ir_tf214.h5'
else:
    var_name=['albedo_01','albedo_02','albedo_03','albedo_04','albedo_05','albedo_06',
              'tbb_07','tbb_08','tbb_09','tbb_10','tbb_11','tbb_12','tbb_13','tbb_14',
              'tbb_15','tbb_16','SOZ']
    var_combin=[('albedo_01','albedo_02','albedo_03'),('albedo_04','albedo_05','albedo_06'),
              ('tbb_08','tbb_09','tbb_10'),('tbb_13','tbb_14','tbb_15')]
    DDIM_weight='/public/home/Xiongqq/weight_use/tf214_v2/ddim_full_tf214.h5'
encoder_weight="/public/home/Xiongqq/weight_use/tf214_v2/encoder_tf214.h5"
decoder_weight="/public/home/Xiongqq/weight_use/tf214_v2/decoder_tf214.h5"

'''
#%%
test_files= ['2020_Himawari_cloudsat_128_cloud_SAZ']
data_path = "/mnt/d/data/Train_data"
save_name = "GEN_samples_2020"
save_path = "/mnt/d/result"
DDIM_weight="/mnt/d/weights/Latent_DDIM3/Lat_DDIM_ema_time3_epoch19_loss0.382.h5"
encoder_weight="/mnt/d/weights/VAE_long128_new3/encoder_time1_epochs17_loss88.029.h5"
decoder_weight="/mnt/d/weights/VAE_long128_new3/decoder_time1_epochs17_loss88.029.h5"
'''
'''
#%%
test_files= ['2020_Himawari_cloudsat_128_cloud_SAZ']
data_path = "/work/home/acmh4zm9q3/Data_Train"
save_name = "Full_GEN_samples_2020"
save_path = "/work/home/acmh4zm9q3/Model_test"
DDIM_weight="/work/home/acmh4zm9q3/LDM_Out/weight/Latent_DDIM_final/Lat_DDIM_ema_time5_epoch29_loss0.312.weights.h5"
encoder_weight="/work/home/acmh4zm9q3/VAE_out/weight/kl_0001/encoder_time4_epochs29_loss416.884.weights.h5"
decoder_weight="/work/home/acmh4zm9q3/VAE_out/weight/kl_0001/decoder_time4_epochs29_loss416.884.weights.h5"
'''

#%%
def make_dataset(data_path,data_files,var_name,random=None,trans=False):
    dicts = {key: [] for key in var_name}
    for file_name in data_files:
        file_path = data_path+'/'+file_name+'.nc'
        data_set = read.read_varible(file_path,var_name,random=random,trans=trans) 
        for key, value in data_set.items():
            dicts[key].append(value)
    data = {key: np.concatenate(value) for key, value in dicts.items()}
    print(f'样本数量{len(data[var_name[0]])}')
    return data

'''
def generate_data(inputs,timestep,denoise_step):
    skip_timestep=timestep//denoise_step
    variable = tf.convert_to_tensor(inputs, dtype=tf.float32)
    noise = tf.random.normal(shape=(len(variable),32, 32, 1), dtype=tf.float32)
    samples = tf.clip_by_value(noise ,-3, 3)
    #samples = tf.zeros((len(variable), 32, 32, 1), dtype=tf.float32)
    progbar = tf.keras.utils.Progbar(timestep)
    for t in range(timestep-1,0,-skip_timestep):
        tk=t
        ts=t-skip_timestep
        if ts<0:
            ts=0
        print(tk,ts)
        t_k = tf.cast(tf.fill(variable.shape[0], tk), dtype=tf.int32)
        t_s = tf.cast(tf.fill(variable.shape[0], ts), dtype=tf.int32)
        pred_noise = network.predict([samples,t_k,variable], verbose=1, batch_size=batchs)
        samples = Diffusion.DDIM_denoise(pred_noise,samples,t_k,t_s,clip_denoised=True)
        progbar.update(timestep - t) 
    return samples
'''

@tf.function(reduce_retracing=True)
def generate_one_batch(variable):
    batch_size = tf.shape(variable)[0]
    samples = tf.random.normal((batch_size,32,32,1), dtype=tf.float32)
    samples = tf.clip_by_value(samples,-3.0,3.0)
    skip_timestep = timestep//denoise_step

    for t in range(timestep-1,0,-skip_timestep):
        ts = max(t-skip_timestep,0)
        t_k = tf.fill([batch_size],tf.cast(t,tf.int32))
        t_s = tf.fill([batch_size],tf.cast(ts,tf.int32))
        pred_noise = network([samples,t_k,variable],training=False)
        samples = Diffusion.DDIM_denoise(pred_noise,samples,t_k,t_s,clip_denoised=True)

    result = decoder(samples,training=False)
    return result

def generate_data(inputs,batch_size):
    inputs = tf.convert_to_tensor(inputs,dtype=tf.float32)
    total_samples = int(inputs.shape[0])
    total_batch = int(np.ceil(total_samples/batch_size))
    progbar = tf.keras.utils.Progbar(total_batch)
    results = []

    for i in range(total_batch):
        start = i*batch_size
        end = min(start+batch_size,total_samples)
        variable = inputs[start:end]
        result = generate_one_batch(variable)
        results.append(result)
        progbar.update(i+1)

    return tf.concat(results,axis=0)

def test_method(data,var_name,method=None):
    if var_name is None:
        return data
    if isinstance(var_name, str):
        var_name = [var_name]
    for name in var_name:
        if method== 'random':
            idx = np.random.permutation(data[name].shape[0])
            data[name] = data[name][idx]
        elif method== 'zeros':
            data[name][:]=np.nan
        elif method== 'noise':
            noise = np.random.uniform(0, 1, data[name].shape)
            value = data[name]
            min_v = value.min()
            data[name]=(value-min_v)*noise + min_v
        else:
            data
    return data

def sensitivity(data,var_name,alpha):
    if var_name is None:
        return data
    if isinstance(var_name, str):
        var_name = [var_name]
    for name in var_name:
        std = np.std(data[name])
        data[name] = data[name] + alpha * std
    return data

#%%
"""初始化VAE，加载权重"""
encoder=VAE.Encoder(64, 128)
decoder=VAE.Decoder(32, 32)
encoder.load_weights(encoder_weight)
decoder.load_weights(decoder_weight)
print('VAE加载完成')
#%%
"""初始化DDIM，加载权重"""
Diffusion = DDIM_UNet.GaussianDiffusion(timesteps=timestep,clip_min=-3.0,clip_max=3.0,)
network = DDIM_UNet.U_Net(32,64,len(var_name))
network.load_weights(DDIM_weight)
print('DDIM加载完成')
#%%
"""读取数据集"""
#data_path="D:/data/Train"
#test_files=["2020_Himawari_cloudsat_128_cloud_SAZ"]
"""VAE编码"""
test_target = make_dataset(data_path,test_files,tar_name,trans=True)
target_data = read.scale_Reflect(test_target['Radar_Reflectivity'])
mean,log_var,latent = encoder.predict(target_data, verbose=1, batch_size=batchs)
test_target['Tar_Latent']=latent[...,0]
test_target['Tar_mean']=mean[...,0]
test_target['Tar_logvar']=log_var[...,0]
del mean,log_var,latent
print('Radar_Reflectivity编码完成')

"""VAE编码"""
test_input=make_dataset(data_path,test_files,var_name)
miss_var=var_name[:]
#miss_var[-1]=None
#alpha_list = np.arange(-2, 2.5, 0.5)
#for alpha in alpha_list: #miss_var+
for j in range(1,31,1): #miss_var+
    for name in miss_var+var_combin:#[None]:
        data = copy.deepcopy(test_target)
        input_data=copy.deepcopy(test_input)
        #input_data[name][:]=np.nan
        #input_data = sensitivity(input_data,name,alpha)
        input_data = test_method(input_data,name,method='random')
        if isinstance(name, (list, tuple, set)):
            name = f'z_{name[0]}~{name[-1].split("_")[-1]}'
        elif name is None:
            name = 'original'
        final_save = f'{save_path}_{name}'
        input_data  = read.scale_varible(input_data,var_name)
        print(name,'***********************************************')
        #%%
        """DDIM生成"""
        for i in range(1):
            Gen_result = generate_data(input_data, batch_size = batchs)
            data['Gen_result'] = read.rescale_gen(Gen_result)[...,0]
            del Gen_result
            result_rmse = analyse.Count_remse(data['Gen_result'],data['Radar_Reflectivity'])
            result_mean,result_var=analyse.mean_var(result_rmse)
            print(f'result_mean:{result_mean}  result_var:{result_var}')
            del result_mean,result_var,result_rmse 
            #%%
            """保存数据"""
            #final_name = f'{save_name}_{name}_{i}'
            #final_name = f'{save_name}_{name}_{alpha}alpha'
            final_name = f'{save_name}_{name}_random_{j}'
            read.save_data_nc(data,final_save,final_name)
            print(f'数据已保存{final_name}')



