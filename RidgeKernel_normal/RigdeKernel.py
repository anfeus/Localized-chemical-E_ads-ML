#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:16:22 2022

@author: andres
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import model_selection
from sklearn.model_selection import KFold, cross_val_score, cross_validate
import shap
from xgboost import plot_importance
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler

from catboost import CatBoostRegressor
import warnings

def plot_correlation(corr_name, column_name, plot_name):
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_name, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(column_name),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(column_name, rotation='vertical',fontsize=6)
    ax.set_yticklabels(column_name,fontsize=6)
    plt.tight_layout()
    plt.show()
    ax.figure.savefig(str(plot_name + '.png'), dpi=600, transparent=True)
    
def shap_beeswarm_plot(shap_values_mol, color_plot_shap, plot_name):
    shap.plots.beeswarm(shap_values_mol, show=False, color=plt.get_cmap(color_plot_shap))
    plt.xlim(-3,3)
    plt.tight_layout()
    plt.savefig('shap_' + plot_name + '.png', dpi=600, transparent=True)
    plt.show()
    
def shap_scatter_plot(shap_values_mol, color_plot_shap, plot_name, ref_color_sca):    
    shap.plots.scatter(shap_values_mol, color=ref_color_sca, show=False, cmap=color_plot_shap)
    plt.tight_layout()
    plt.savefig('shap_scatter_' + plot_name + '.png', dpi=600, transparent=True)
    plt.show()

def cluster_scatter_plot(s2d_values, color_plot_dbscan, plot_name, feature, column_feature, column_feature_stan, size_plt, alpha_plt):    
    p_points = plt.scatter(s2d_values[:, 0], s2d_values[:, 1], c=column_feature_stan[feature], norm=plt.Normalize(), s=size_plt, alpha=alpha_plt, cmap=color_plot_dbscan)
    p_points.axes.set_xticks([])
    p_points.axes.set_yticks([])
    p_points.axes.axis('square')
    clb = plt.colorbar(p_points)
    clb.set_label(feature)
    indices_lab = np.round(np.linspace(column_feature[feature].min(), column_feature[feature].max(), num=5, endpoint=True), 2)
    indices = np.linspace(column_feature_stan[feature].min(), column_feature_stan[feature].max(), num=5, endpoint=True)
    clb.set_ticks(indices)
    clb.set_ticklabels(indices_lab)
    plt.tight_layout()
    plt.savefig('umap_dbscan_' + plot_name + '.png', dpi=600, transparent=True)
    plt.show()

def X_y_train_test(X_in, y_in):
    X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

color_plot = "viridis"
color_umad_1 = "viridis"
color_umad_2 = "tab20c"
color_umad_3 = "turbo"
model = 'KernelRidge'

data = pd.read_pickle(r'database_total_final.pkl')
data = data.drop(columns=['Group_surf', 'SpecificHeat_surf'])

data = data[['atoms_surf', 'CN', 'gCN', 'CN_max',
              'FermiEnergy_surf', 'Total_charge_surf', 's_charge_surf', 'p_charge_surf', 'd_charge_surf', 'd_center', 'WorkFunction', 'stm_surf',
              'AtomicMass_surf', 'Electronegativity_surf', 'FirstIonization_surf', 'AtomicRadius_surf', 'MeltingPoint_surf', 'BoilingPoint_surf', 'NumberofShells_surf',
              'H_ads',
              'HOMO_ads', 'Total_charge_ads', 's_charge_ads', 'p_charge_ads', 'stm_ads',
              'AtomicMass_ads', 'Electronegativity_ads', 'Group_ads', 'FirstIonization_ads', 'AtomicRadius_ads', 'MeltingPoint_ads', 'BoilingPoint_ads', 'SpecificHeat_ads', 'NumberofShells_ads',
              'Adsorption_energy']]

data = data.astype(float)

data.drop_duplicates()

data = data[data['Adsorption_energy'] <= 0] 

column_names = data.columns.values.tolist()

X = data.loc[:, data.columns != 'Adsorption_energy']
y = data['Adsorption_energy']

###Normalize with Maximum Absolute

# scaler = MaxAbsScaler()
# scaled = scaler.fit_transform(X)
# scaled_Max_X = pd.DataFrame(scaled, columns=X.columns)
# scaled_Max_Xy = pd.concat([scaled_Max_X, y], axis = 1)

# ###Normalize with Min-Max

# scaler = MinMaxScaler()
# scaled = scaler.fit_transform(X)
# scaled_MinMax_X = pd.DataFrame(scaled, columns=X.columns)
# scaled_MinMax_Xy = pd.concat([scaled_MinMax_X, y], axis = 1)

# ###Normalize with standar scaler

# scaler = StandardScaler()
# scaled = scaler.fit_transform(X)
# scaled_standar_X = pd.DataFrame(scaled, columns=X.columns)
# scaled_standar_Xy = pd.concat([scaled_standar_X, y], axis = 1)

###Normalize with Robust Scaler

scaler = RobustScaler()
scaled = scaler.fit_transform(X)
scaled_robust_X = pd.DataFrame(scaled, columns=X.columns)
scaled_robust_Xy = pd.concat([scaled_robust_X, y], axis = 1)
scaled_robust_Xy_surf = pd.concat([scaled_robust_X.iloc[:, :19], y], axis = 1)
scaled_robust_Xy_ads = pd.concat([scaled_robust_X.iloc[:, 19:], y], axis = 1)

column_names = data.columns.values.tolist()
column_names_X = scaled_robust_X.columns.values.tolist()
column_names_surf = scaled_robust_Xy_surf.values.tolist()
column_names_ads = scaled_robust_Xy_ads.values.tolist()

correlations = data.corr()
# # # # # correlations_scaled_Max = scaled_Max_Xy.corr()
# # # # # correlations_scaled_MinMax = scaled_MinMax_Xy.corr()
# # # # correlations_scaled_standar = scaled_standar_Xy.corr()
correlations_scaled_robust = scaled_robust_Xy.corr()
correlations_scaled_robust_surf = scaled_robust_Xy_surf.corr()
correlations_scaled_robust_ads = scaled_robust_Xy_ads.corr()

# plot_correlation(correlations, column_names, 'correlations')
# # # # # # plot_correlation(correlations_scaled_Max, column_names, 'correlations_scaled_Max')
# # # # # # plot_correlation(correlations_scaled_MinMax, column_names, 'correlations_scaled_MinMax')
# # # # # plot_correlation(correlations_scaled_standar, column_names, 'correlations_scaled_standar')
# plot_correlation(correlations_scaled_robust, column_names, 'correlations_scaled_robust')

# plot_correlation(correlations_scaled_robust_surf, correlations_scaled_robust_surf, 'correlations_scaled_robust_surf')
# plot_correlation(correlations_scaled_robust_ads, correlations_scaled_robust_ads, 'correlations_scaled_robust_ads')

X_train, X_test, y_train, y_test = X_y_train_test(scaled_robust_X , y)

kfold = KFold(n_splits=10, shuffle=True, random_state=7)
cv = ShuffleSplit(n_splits=50, test_size=0.3, random_state=0)
# # ['laplacian', 'rbf', 'linear', 'polynomial']
# kernel_list = ['laplacian', 'rbf', 'linear', 'polynomial']
# alpha_list = [0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.12, 0.14, 0.16, 0.3, 0.5, 0.7, 0.9, 1]

# results = []
# for ker in kernel_list:
#     for alp in alpha_list:
        
#         print(ker, alp)
#         RidgeKernel_model = KernelRidge(alpha=alp, kernel=ker, gamma=None)
#         RidgeKernel_model.fit(X_train,y_train)

#         mse_train = mean_squared_error(y_train, RidgeKernel_model.predict(X_train))
#         print("MSE Train Set Regression: %.4f" % mse_train)
#         mse = mean_squared_error(y_test, RidgeKernel_model.predict(X_test))
#         print("MSE Test Set Regression: %.4f" % mse)
        
#         results_rid = model_selection.cross_val_score(RidgeKernel_model, X_train, y_train, cv=kfold)
#         #             # print("Accuracy: Final mean:%.3f%%, Final standard deviation:(%.3f%%)" % (results_xgb.mean()*100.0, results_xgb.std()*100.0))
#         print(results_rid.mean(), results_rid.std()*100.0)
#         #             # print("Variance of kfold accuracies:",results_xgb.var())
#         print("")
        
#         results.append([ker, alp, mse_train, mse, results_rid.mean(), results_rid.std()*100.0])

# results = pd.DataFrame(results, columns=['kernel', 'alpha', 'mse_train', 'mse_test', 'Accuracy_k_folds', 'deviation_k_folds'])
# name = 'results_ridge_total_par_1.pkl'
# # results.to_pickle(name)
# best_tunning = pd.read_pickle(name)

# # chi2_kernel
RidgeKernel_model = KernelRidge(alpha=0.01, kernel='laplacian', gamma=None)
RidgeKernel_model.fit(X_train,y_train)

mse_train = mean_squared_error(y_train, RidgeKernel_model.predict(X_train))
print("MSE Train Set XGB Regression: %.4f" % mse_train)
print("RMSE Train Set XGB Regression: %.4f" % np.sqrt(mse_train)) 
mae_train = mean_absolute_error(y_train, RidgeKernel_model.predict(X_train))
print("MAE Train Set XGB Regression: %.4f" % mae_train)  
mse = mean_squared_error(y_test, RidgeKernel_model.predict(X_test))
print("MSE Test Set XGB Regression: %.4f" % mse)
print("RMSE Test Set XGB Regression: %.4f" % np.sqrt(mse))
mae = mean_absolute_error(y_test, RidgeKernel_model.predict(X_test))
print("MAE Test Set XGB Regression: %.4f" % mae)

results_RidgeKernel = model_selection.cross_val_score(RidgeKernel_model, X_train, y_train, cv=kfold)
print("Accuracy: Final mean:%.3f%%, Final standard deviation:(%.3f%%)" % (results_RidgeKernel.mean()*100.0, results_RidgeKernel.std()*100.0))
print('Accuracies from each of the number folds using kfold:',results_RidgeKernel)
print("Variance of kfold accuracies:",results_RidgeKernel.var())

filename = 'Results_' + model + '_total'

outfile = open(filename, "w")

outfile.write("MSE Train Set XGB Regression: " + str(round(mse_train,3)) + "\n")
outfile.write("RMSE Train Set XGB Regression: " + str(round(np.sqrt(mse_train),3)) + "\n")
outfile.write("MAE Train Set XGB Regression: " + str(round(mae_train,3)) + "\n")
outfile.write("MSE Test Set XGB Regression: " + str(round(mse,3)) + "\n")
outfile.write("RMSE Test Set XGB Regression: " + str(round(np.sqrt(mse),3)) + "\n")
outfile.write("MAE Test Set XGB Regression: " + str(round(mae,3)) + "\n")
outfile.write("Accuracy: Final mean: " + str(round(results_RidgeKernel.mean()*100.0,3)) + "%, Final standard deviation: " + str(round(results_RidgeKernel.std()*100.0,3)) + "%\n")
outfile.write('Accuracies from each of the number folds using kfold: ' + str(results_RidgeKernel) + "\n")
outfile.write("Variance of kfold accuracies:" + str(results_RidgeKernel.var()) + "\n")
# outfile.write("\n")

outfile.close()

# explainer = shap.KernelExplainer(RidgeKernel_model.predict, X_train)
# shap_values = explainer.shap_values(X_train)

# shap_beeswarm_plot(shap_values, color_plot, 'overall_' + model + '_total')
# shap_beeswarm_plot(shap_values[:,:19], color_plot, 'surf_' + model + '_total')
# shap_beeswarm_plot(shap_values[:,19:], color_plot, 'ads_' + model + '_total')
# shap_beeswarm_plot(shap_values[:,:4], color_plot, 'surf_geo_' + model + '_total')
# shap_beeswarm_plot(shap_values[:,4:12], color_plot, 'surf_ele_' + model + '_total')
# shap_beeswarm_plot(shap_values[:,12:19], color_plot, 'surf_at_' + model + '_total')

# shap_scatter_plot(shap_values[:,'d_charge_surf'], color_umad_3, 'd_charge_surf_' + model + '_total', shap_values)
# shap_scatter_plot(shap_values[:,'atoms_surf'], color_umad_3, 'atoms_surf_' + model + '_total', shap_values)
# shap_scatter_plot(shap_values[:,'s_charge_surf'], color_umad_3, 's_charge_surf_' + model + '_total', shap_values)
# # shap_scatter_plot(shap_values[:,'d_center'], color_umad_3, 'd_center_surf_' + model + '_total', shap_values)
# shap_scatter_plot(shap_values[:,'stm_surf'], color_umad_3, 'stm_surf_' + model + '_total', shap_values)

# from umap import UMAP

# # for plotting
# alpha = .4
# size = 15

# # compute 2D embedding of raw variable values
# X_2d = UMAP(n_components=2, n_neighbors=200, min_dist=0).fit_transform(X_train)

# # compute 2D embedding of SHAP values
# s_2d = UMAP(n_components=2, n_neighbors=200, min_dist=0).fit_transform(shap_values.values)

# p_points = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, norm=plt.Normalize(), s=size, alpha=alpha, cmap=color_umad_1)
# p_points.axes.set_xticks([])
# p_points.axes.set_yticks([])
# p_points.axes.axis('square')
# clb = plt.colorbar(p_points)
# clb.set_label('Adsorption energy [eV]')
# # clb.set_ticks([y_train.min(), y_train.max()])
# # clb.set_ticklabels(['Low', 'High']) 
# plt.tight_layout()
# plt.savefig('umap_x_raw_' + model + '_total.png', dpi=600, transparent=True)
# plt.show()



# p_points = plt.scatter(s_2d[:, 0], s_2d[:, 1], c=y_train, norm=plt.Normalize(), s=size, alpha=alpha, cmap=color_umad_1)
# p_points.axes.set_xticks([])
# p_points.axes.set_yticks([])
# p_points.axes.axis('square')
# clb = plt.colorbar(p_points)
# clb.set_label('Adsorption energy [eV]')
# # clb.set_ticks([y_train.min(), y_train.max()])
# # clb.set_ticklabels(['Low', 'High']) 
# plt.tight_layout()
# plt.savefig('umap_shap_' + model + '_total.png', dpi=600, transparent=True)
# plt.show()

# # labeling of clusters

# from sklearn.cluster import DBSCAN

# s_labels = DBSCAN(eps=0.3, min_samples=2).fit(s_2d).labels_
# s_dbscan = DBSCAN(eps=0.3, min_samples=2).fit(s_2d)

# clus_lab = []
# tem = 0
# for cluster, count in pd.Series(s_labels).value_counts().sort_index().items():    
#     clus_lab.append(['L' + str(tem), s_2d[s_labels == cluster].mean(0)[0], s_2d[s_labels == cluster].mean(0)[1]])
#     tem += 1
#     print(cluster, count, s_2d[s_labels == cluster].mean(0))

# # DBSCAN plot with labels

# p_points = plt.scatter(s_2d[:, 0], s_2d[:, 1], c=s_labels, norm=plt.Normalize(), s=size, alpha=alpha, cmap=color_umad_2)
# p_points.axes.set_xticks([])
# p_points.axes.set_yticks([])

# for i in range(len(clus_lab)):
#     p_points.axes.text(clus_lab[i][1], clus_lab[i][2], clus_lab[i][0], fontsize = 8)

# p_points.axes.axis('square')
# plt.tight_layout()
# plt.savefig('umap_dbscan_' + model + '_total.png', dpi=600, transparent=True)
# plt.show()

# # DBSCAN plot by feature

# for i in column_names_X:
#     cluster_scatter_plot(s_2d, color_umad_3, i + '_' + model + '_total', i, data, X_train, size, alpha)

lin_teo = np.arange(-10, 0.5, 0.1)
y_pred = RidgeKernel_model.predict(X_test)
y_pred_train = RidgeKernel_model.predict(X_train)
# title = "Real vs. Predicted energy value with the XGB Model for the Training and Test Sets"
plt.scatter(lin_teo, lin_teo, linestyle='dashed', color='grey', s=2, alpha=0.5)
plt.scatter(y_train, y_pred_train, color='blue', label='Train set', s=15, alpha=0.4)
plt.scatter(y_test, y_pred, color='green', label='Test set', s=15, alpha=0.4)
plt.xlabel('DFT Energy [eV]')
plt.xlim([-11, 1])
plt.ylim([-11, 1])
plt.ylabel('ML predicted Energy [eV]')
plt.axis('square')
plt.legend()
plt.title(label='ML vs DFT adsorption energy for the KRR model', fontsize=13)
plt.tight_layout()
plt.savefig('Predict_vs_Real.png', dpi=200)
plt.show()
