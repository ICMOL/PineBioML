from urllib import response

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect, HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt, csrf_protect

import pandas as pd
import numpy as np
from numpy import interp
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from matplotlib.patches import Patch

import xlwt
import xlsxwriter

from PIL import Image
from io import BytesIO


@csrf_exempt
def home(request):
    # print client ip
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    print('\nclient_ip =', ip)
    context = {}
    context["name"] = '\tGenetic Data Analysis Platform\t'
    return render(request, 'home/home.html', context)


@csrf_exempt
def check(request):
    context = {}
    context["name"] = '資料探勘第三組'
    if request.method == "POST":
        shutil.rmtree('media')
        os.mkdir('media')
        uploaded_file = request.FILES['file']
        print(uploaded_file)
        fss = FileSystemStorage()
        file = fss.save(uploaded_file.name, uploaded_file)
        # return redirect("/")
    mediafiles = os.listdir(settings.MEDIA_ROOT)
    #path = "C:\csmuProject\media"  # 資料夾目錄
    path = "media"
    files = os.listdir(path)  # 得到資料夾下的所有檔名稱
    df = pd.read_csv(path + "/" + files[0])
    print(df)
    lenRC = "[" + str(len(df)) + " rows x " + str(df.shape[1]) + " columns]"
    print(lenRC)
    context = {'lenRC': lenRC}
    # df = df.iloc[0:6, 0:10]
    # df = df.to_html()
    # context = {'df': df}
    df_head_left = df.iloc[0:5, 0:5]
    df_head_left.index = df_head_left.index + 1
    df_head_left = df_head_left.to_html(col_space=40, classes='dftable_h')
    context = {'df_head_left': df_head_left}

    df_head_right = df.iloc[0:5, df.shape[1] - 5:df.shape[1]]
    df_head_right = df_head_right.to_html(col_space=40,
                                          index=False,
                                          classes='dftable_t')
    context = {'df_head_right': df_head_right}

    df_tail_left = df.iloc[len(df) - 5:len(df), 0:5]
    df_tail_left.index = df_tail_left.index + 1
    df_tail_left = df_tail_left.to_html(col_space=40,
                                        columns=None,
                                        classes='dftable_h')
    context = {'df_tail_left': df_tail_left}

    df_tail_right = df.iloc[len(df) - 5:len(df), df.shape[1] - 5:df.shape[1]]
    df_tail_right = df_tail_right.to_html(col_space=40,
                                          columns=None,
                                          index=False,
                                          classes='dftable_t')
    context = {'df_tail_right': df_tail_right}

    return render(request, 'home/checkData.html', locals())


@csrf_exempt
def dataCheck(request):
    f1 = request.POST.get('f1')
    f2 = request.POST.get('f2')
    c1 = request.POST.get('c1')
    ordernum = []
    ordernum.append(int(f1) + 1)
    ordernum.append(int(f2) + 1)
    ordernum.append(int(c1) + 1)
    path = "media"  # 資料夾目錄
    files = os.listdir(path)  # 得到資料夾下的所有檔名稱
    df = pd.read_csv(path + "/" + files[0])
    classList = list(df[list(df)[int(c1)]].value_counts().index)
    print("dataCheck ok")

    return JsonResponse({'classList': classList, 'ordernum': ordernum})


@csrf_exempt
def featurepage(request):
    if request.method == "POST":
        print("featurepage ok")
        return render(request, 'home/fsm.html', locals())


@csrf_exempt
def selectcheck(request):
    print("selectcheck ok")

    return render(request, 'home/parameter.html', locals())


featurearr = []
modelarr = []
fsarr = []
aucarr = []
stdarr = []


@csrf_exempt
def finalcheck(request):
    # sparse request
    # it shall be down in dict
    lr1 = request.POST.get('lr1')
    lr2 = request.POST.get('lr2')
    svclinear1 = request.POST.get('svclinear1')
    svcpoly1 = request.POST.get('svcpoly1')
    svcpoly2 = request.POST.get('svcpoly2')
    svcrbf1 = request.POST.get('svcrbf1')
    rf1 = request.POST.get('rf1')
    rf2 = request.POST.get('rf2')
    xgb1 = request.POST.get('xgb1')
    xgb2 = request.POST.get('xgb2')
    ada1 = request.POST.get('ada1')
    ada2 = request.POST.get('ada2')
    gb1 = request.POST.get('gb1')
    gb2 = request.POST.get('gb2')
    dt1 = request.POST.get('dt1')
    mlp1 = request.POST.get('mlp1')
    mlp2 = request.POST.get('mlp2')
    mlp2num1 = request.POST.get('mlp2num1')
    mlp2num2 = request.POST.get('mlp2num1')
    featstart = int(request.POST.get('featstart')) - 1
    featend = int(request.POST.get('featend'))
    classnum = int(request.POST.get('class')) - 1

    cbl1 = request.POST.get('cbl1')
    cbl1 = cbl1.split(',')
    cbl2 = request.POST.get('cbl2')
    cbl2 = cbl2.split(',')

    radioselect = int(request.POST.get('radioselect'))
    t_select1 = request.POST.get('t_select1')
    t_select2 = request.POST.get('t_select2')

    # IO
    path = "media"  # 資料夾目錄
    files = os.listdir(path)  # 得到資料夾下的所有檔名稱
    df = pd.read_csv(path + "/" + files[0])
    seleted_y = df[df.columns[classnum]].isin([t_select1, t_select2])
    df = df[seleted_y]  # class為list(df)[classnum]
    df1 = pd.concat(
        [df.iloc[:, classnum:classnum + 1], df.iloc[:, featstart:featend]],
        axis=1)
    df1 = df1.dropna(axis=1)
    df1[list(df)[classnum]] = df1[list(df)[classnum]].map({
        t_select1: 0,
        t_select2: 1
    })

    # preprocessing
    scaler = MinMaxScaler()
    df2 = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns)

    cr = df2.corr().abs()
    cr1 = pd.DataFrame(cr[list(df)[classnum]])
    cr1 = cr1.T

    df3 = pd.concat([df2, cr1])
    result1 = df3.sort_values(by=[list(df)[classnum]], ascending=False, axis=1)
    df3 = result1.drop([list(df)[classnum]], axis=0)

    correl2 = df3.iloc[:, 1:].corr().abs()
    upper_tri = correl2.where(
        np.triu(np.ones(correl2.shape), k=1).astype(np.bool))
    to_drop = [
        column for column in upper_tri.columns if any(upper_tri[column] > 0.6)
    ]

    df4 = df3.drop(to_drop, axis=1)
    X = df4.iloc[:, 1:]
    y = df4.iloc[:, 0]
    newX = np.transpose(X.columns)

    # clarify models
    clf_labels = []
    all_clf = []
    seed = 1
    for i in cbl2:
        if i == '0':
            lr1 = float(lr1)
            clf0 = LogisticRegression(random_state=seed, C=lr1, solver=lr2)
            clf_labels.append('Logistic Regression')
            all_clf.append(clf0)
        elif i == '1':
            svclinear1 = float(svclinear1)
            clf1 = SVC(kernel='linear',
                       random_state=seed,
                       C=svclinear1,
                       probability=True)
            clf_labels.append('SVC-Linear')
            all_clf.append(clf1)
        elif i == '2':
            svcpoly1 = float(svcpoly1)
            svcpoly2 = int(svcpoly2)
            clf2 = SVC(kernel='poly',
                       random_state=seed,
                       C=svcpoly1,
                       degree=svcpoly2,
                       probability=True)
            clf_labels.append('SVC-Polynomial')
            all_clf.append(clf2)
        elif i == '3':
            svcrbf1 = float(svcrbf1)
            clf3 = SVC(kernel='rbf',
                       random_state=seed,
                       C=svcrbf1,
                       probability=True)
            clf_labels.append('SVC-RBF')
            all_clf.append(clf3)
        elif i == '4':
            if dt1 != 'None':
                dt1 = int(dt1)
            else:
                dt1 = None
            clf4 = DecisionTreeClassifier(random_state=seed, max_depth=dt1)
            clf_labels.append('Decision Tree Classifier')
            all_clf.append(clf4)
        elif i == '5':
            rf1 = int(rf1)
            if rf2 != 'None':
                rf2 = int(rf2)
            else:
                rf2 = None
            clf5 = RandomForestClassifier(random_state=seed,
                                          n_estimators=rf1,
                                          max_depth=rf2)
            clf_labels.append('Random Forest Classifier')
            all_clf.append(clf5)
        elif i == '6':
            xgb1 = int(xgb1)
            xgb2 = int(xgb2)
            clf6 = XGBClassifier(random_state=seed,
                                 n_estimators=xgb1,
                                 max_depth=xgb2)
            clf_labels.append('XGBClassifier')
            all_clf.append(clf6)
        elif i == '7':
            gb1 = int(gb1)
            gb2 = float(gb2)
            clf7 = GradientBoostingClassifier(random_state=seed,
                                              n_estimators=gb1,
                                              learning_rate=gb2)
            clf_labels.append('Gradient Boosting Classifier')
            all_clf.append(clf7)
        elif i == '8':
            ada1 = int(ada1)
            ada2 = float(ada2)
            clf8 = AdaBoostClassifier(random_state=seed,
                                      n_estimators=ada1,
                                      learning_rate=ada2)
            clf_labels.append('AdaBoost Classifier')
            all_clf.append(clf8)
        elif i == '9':
            mlp2num1 = int(mlp2num1)
            if mlp2 == '1':
                clf9 = MLPClassifier(random_state=seed,
                                     solver=mlp1,
                                     hidden_layer_sizes=(mlp2num1))
            else:
                mlp2num2 = int(mlp2num2)
                clf9 = MLPClassifier(random_state=seed,
                                     solver=mlp1,
                                     hidden_layer_sizes=(mlp2num1, mlp2num2))
            clf_labels.append('Multi-Layer Perceptron')
            all_clf.append(clf9)

    # print(clf_labels)
    # print(all_clf)

    cv = StratifiedKFold(n_splits=radioselect, random_state=seed, shuffle=True)
    result_arr = []
    colors_dict = {}
    fs_names = []
    all_fs = []
    # global X_LiR1, X_LoR1, X_SVC1, X_DT1, X_RF1, X_XGBoost1, X_GB1, X_AB1

    # 特徵選取
    for i in cbl1:
        if i == '0':
            # linear regression
            model = LinearRegression()
            model.fit(X, y)

            importance_abs = abs(model.coef_)
            LiR_score = pd.DataFrame(importance_abs, columns=['score'])
            LiR_score = LiR_score.T
            X_LiR = pd.DataFrame(LiR_score)
            X_LiR.columns = newX
            X_LiR = pd.concat([X, X_LiR])
            X_LiR = X_LiR.sort_values(by=['score'], ascending=False, axis=1)
            X_LiR = X_LiR.drop(['score'], axis=0)

            auc_val = []
            FS1 = "Linear Regression"
            X_LiR1 = X_LiR.iloc[:, 0:20]
            # X_LiR1 = X_LiR.iloc[:, 0:10]
            fs_names.append(FS1)
            all_fs.append(X_LiR1)

            for clf, label in zip(all_clf, clf_labels):
                # print('10-fold cross validation:\n')
                scores = cross_val_score(estimator=clf,
                                         X=X_LiR1,
                                         y=y,
                                         cv=cv,
                                         scoring='roc_auc')
                # print("ROC AUC: %0.3f (+/- %0.3f) [%s]"% (scores.mean(), scores.std(), label))

                auc_val.append([FS1, label, np.mean(scores), np.std(scores)])
            result = sorted(auc_val, key=lambda s: s[2], reverse=True)
            LiR_result2 = pd.DataFrame(
                result, columns=["FS", "Classifier", "AUC", "STD"]).round(3)
            result1 = sorted(auc_val, key=lambda s: s[2], reverse=False)
            # print(result)
            # for FS, label, auc_mean, auc_std in result:
            #     print("%s, AUC = %0.3f +/- %0.3f" % (label, auc_mean, auc_std))
            # for FS, label, auc_mean, auc_std in result1:
            # print("%s, AUC = %0.3f +/- %0.3f" %(label, auc_mean, auc_std))
            # b = plt.barh(label, auc_mean, height=0.5)
            # plt.bar_label(b,fmt='%.3f',label_type='edge',fontsize=12)
            # plt.xlabel('AUC')

            X_LiR1_T = pd.DataFrame(np.transpose(X_LiR1.columns))
            X_LiR1_T.columns = ["Name of Feature"]
            #X_LiR1_T.index = ["1", "2", "3", "4", "5"]
            X_LiR1_T.index = [
                "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                "13", "14", "15", "16", "17", "18", "19", "20"
            ]
            LiR_result2 = LiR_result2
            result_arr.append(LiR_result2)
            colors_dict['Linear Regression'] = '#207297'
        elif i == '1':
            model = LogisticRegression(random_state=seed)
            print("X")
            print(X)
            print("y")
            print(y)
            model.fit(X, y)
            importance = model.coef_[0]
            importance_abs = abs(importance)
            LoR_score = pd.DataFrame(importance_abs, columns=['score'])
            LoR_score = LoR_score.T
            X_LoR = pd.DataFrame(LoR_score)
            X_LoR.columns = newX
            X_LoR = pd.concat([X, X_LoR])
            X_LoR = X_LoR.sort_values(by=['score'], ascending=False, axis=1)
            X_LoR = X_LoR.drop(['score'], axis=0)

            auc_val = []
            FS2 = "Logistic Regression"
            X_LoR1 = X_LoR.iloc[:, 0:20]
            # X_LoR1 = X_LoR.iloc[:, 0:10]
            fs_names.append(FS2)
            all_fs.append(X_LoR1)

            for clf, label in zip(all_clf, clf_labels):
                # print('10-fold cross validation:\n')
                scores = cross_val_score(estimator=clf,
                                         X=X_LoR1,
                                         y=y,
                                         cv=cv,
                                         scoring='roc_auc')
                # print("ROC AUC: %0.3f (+/- %0.3f) [%s]"% (scores.mean(), scores.std(), label))

                auc_val.append([FS2, label, np.mean(scores), np.std(scores)])
            result = sorted(auc_val, key=lambda s: s[2], reverse=True)
            LoR_result2 = pd.DataFrame(
                result, columns=["FS", "Classifier", "AUC", "STD"]).round(3)
            result1 = sorted(auc_val, key=lambda s: s[2], reverse=False)
            # print(result)
            # for FS, label, auc_mean, auc_std in result:
            #     print("%s, AUC = %0.3f +/- %0.3f" % (label, auc_mean, auc_std))
            # for FS, label, auc_mean, auc_std in result1:
            # print("%s, AUC = %0.3f +/- %0.3f" %(label, auc_mean, auc_std))
            # b = plt.barh(label, auc_mean, height=0.5)
            # plt.bar_label(b,fmt='%.3f',label_type='edge',fontsize=12)
            # plt.xlabel('AUC')

            X_LoR1_T = pd.DataFrame(np.transpose(X_LoR1.columns))
            X_LoR1_T.columns = ["Name of Feature"]
            X_LoR1_T.index = [
                "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                "13", "14", "15", "16", "17", "18", "19", "20"
            ]
            # X_LoR1_T.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            LoR_result2 = LoR_result2
            result_arr.append(LoR_result2)
            colors_dict['Logistic Regression'] = '#BBE1B7'
        elif i == '2':
            model = SVC(kernel='linear', probability=True, random_state=seed)
            model.fit(X, y)
            importance = model.coef_[0]
            importance_abs = abs(importance)
            SVC_score = pd.DataFrame(importance_abs, columns=['score'])
            SVC_score = SVC_score.T
            X_SVC = pd.DataFrame(SVC_score)
            X_SVC.columns = newX
            X_SVC = pd.concat([X, X_SVC])
            X_SVC = X_SVC.sort_values(by=['score'], ascending=False, axis=1)
            X_SVC = X_SVC.drop(['score'], axis=0)

            auc_val = []
            FS3 = "SVC-Linear"
            X_SVC1 = X_SVC.iloc[:, 0:20]
            # X_SVC1 = X_SVC.iloc[:, 0:10]
            fs_names.append(FS3)
            all_fs.append(X_SVC1)

            for clf, label in zip(all_clf, clf_labels):
                # print('10-fold cross validation:\n')
                scores = cross_val_score(estimator=clf,
                                         X=X_SVC1,
                                         y=y,
                                         cv=cv,
                                         scoring='roc_auc')
                # print("ROC AUC: %0.3f (+/- %0.3f) [%s]"% (scores.mean(), scores.std(), label))

                auc_val.append([FS3, label, np.mean(scores), np.std(scores)])
            result = sorted(auc_val, key=lambda s: s[2], reverse=True)
            SVC_result2 = pd.DataFrame(
                result, columns=["FS", "Classifier", "AUC", "STD"]).round(3)
            result1 = sorted(auc_val, key=lambda s: s[2], reverse=False)
            # print(result)
            # for FS, label, auc_mean, auc_std in result:
            #     print("%s, AUC = %0.3f +/- %0.3f" % (label, auc_mean, auc_std))
            # for FS, label, auc_mean, auc_std in result1:
            # print("%s, AUC = %0.3f +/- %0.3f" %(label, auc_mean, auc_std))
            # b = plt.barh(label, auc_mean, height=0.5)
            # plt.bar_label(b,fmt='%.3f',label_type='edge',fontsize=12)
            # plt.xlabel('AUC')

            X_SVC1_T = pd.DataFrame(np.transpose(X_SVC1.columns))
            X_SVC1_T.columns = ["Name of Feature"]
            X_SVC1_T.index = [
                "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                "13", "14", "15", "16", "17", "18", "19", "20"
            ]
            # X_SVC1_T.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            SVC_result2 = SVC_result2
            result_arr.append(SVC_result2)
            colors_dict['SVC-Linear'] = '#EDB096'
        elif i == '3':
            model = DecisionTreeClassifier(random_state=seed)
            model.fit(X, y)
            importance_abs = abs(model.feature_importances_)
            DT_score = pd.DataFrame(importance_abs, columns=['score'])
            DT_score = DT_score.T
            X_DT = pd.DataFrame(DT_score)
            X_DT.columns = newX
            X_DT = pd.concat([X, X_DT])
            X_DT = X_DT.sort_values(by=['score'], ascending=False, axis=1)
            X_DT = X_DT.drop(['score'], axis=0)

            auc_val = []
            FS4 = "Decision Tree Classifier"
            X_DT1 = X_DT.iloc[:, 0:20]
            # X_DT1 = X_DT.iloc[:, 0:10]
            fs_names.append(FS4)
            all_fs.append(X_DT1)

            for clf, label in zip(all_clf, clf_labels):
                # print('10-fold cross validation:\n')
                scores = cross_val_score(estimator=clf,
                                         X=X_DT1,
                                         y=y,
                                         cv=cv,
                                         scoring='roc_auc')
                # print("ROC AUC: %0.3f (+/- %0.3f) [%s]"% (scores.mean(), scores.std(), label))

                auc_val.append([FS4, label, np.mean(scores), np.std(scores)])
            result = sorted(auc_val, key=lambda s: s[2], reverse=True)
            DT_result2 = pd.DataFrame(
                result, columns=["FS", "Classifier", "AUC", "STD"]).round(3)
            result1 = sorted(auc_val, key=lambda s: s[2], reverse=False)
            # print(result)
            # for FS, label, auc_mean, auc_std in result:
            #     print("%s, AUC = %0.3f +/- %0.3f" % (label, auc_mean, auc_std))
            # for FS, label, auc_mean, auc_std in result1:
            # print("%s, AUC = %0.3f +/- %0.3f" %(label, auc_mean, auc_std))
            # b = plt.barh(label, auc_mean, height=0.5)
            # plt.bar_label(b,fmt='%.3f',label_type='edge',fontsize=12)
            # plt.xlabel('AUC')

            X_DT1_T = pd.DataFrame(np.transpose(X_DT1.columns))
            X_DT1_T.columns = ["Name of Feature"]
            X_DT1_T.index = [
                "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                "13", "14", "15", "16", "17", "18", "19", "20"
            ]
            # X_DT1_T.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            DT_result2 = DT_result2
            result_arr.append(DT_result2)
            colors_dict['Decision Tree Classifier'] = '#77969A'
        elif i == '4':
            model = RandomForestClassifier(random_state=seed)
            model.fit(X, y)
            importance = model.feature_importances_
            RF_score = pd.DataFrame(importance, columns=['score'])
            RF_score = RF_score.T
            X_RF = pd.DataFrame(RF_score)
            X_RF.columns = newX
            X_RF = pd.concat([X, X_RF])
            X_RF = X_RF.sort_values(by=['score'], ascending=False, axis=1)
            X_RF = X_RF.drop(['score'], axis=0)

            auc_val = []
            FS5 = "Random Forest Classifier"
            X_RF1 = X_RF.iloc[:, 0:20]
            # X_RF1 = X_RF.iloc[:, 0:10]
            fs_names.append(FS5)
            all_fs.append(X_RF1)

            for clf, label in zip(all_clf, clf_labels):
                # print('10-fold cross validation:\n')
                scores = cross_val_score(estimator=clf,
                                         X=X_RF1,
                                         y=y,
                                         cv=cv,
                                         scoring='roc_auc')
                # print("ROC AUC: %0.3f (+/- %0.3f) [%s]"% (scores.mean(), scores.std(), label))

                auc_val.append([FS5, label, np.mean(scores), np.std(scores)])
            result = sorted(auc_val, key=lambda s: s[2], reverse=True)
            RF_result2 = pd.DataFrame(
                result, columns=["FS", "Classifier", "AUC", "STD"]).round(3)
            result1 = sorted(auc_val, key=lambda s: s[2], reverse=False)
            # print(result)
            # for FS, label, auc_mean, auc_std in result:
            #     print("%s, AUC = %0.3f +/- %0.3f" % (label, auc_mean, auc_std))
            # for FS, label, auc_mean, auc_std in result1:
            # print("%s, AUC = %0.3f +/- %0.3f" %(label, auc_mean, auc_std))
            # b = plt.barh(label, auc_mean, height=0.5)
            # plt.bar_label(b,fmt='%.3f',label_type='edge',fontsize=12)
            # plt.xlabel('AUC')

            X_RF1_T = pd.DataFrame(np.transpose(X_RF1.columns))
            X_RF1_T.columns = ["Name of Feature"]
            X_RF1_T.index = [
                "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                "13", "14", "15", "16", "17", "18", "19", "20"
            ]
            # X_RF1_T.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            RF_result2 = RF_result2
            result_arr.append(RF_result2)
            colors_dict['Random Forest Classifier'] = '#F3EC91'
        elif i == '5':
            model = XGBClassifier(random_state=seed)
            model.fit(X, y)
            importance_abs = abs(model.feature_importances_)
            XGBoost_score = pd.DataFrame(importance_abs, columns=['score'])
            XGBoost_score = XGBoost_score.T
            X_XGBoost = pd.DataFrame(XGBoost_score)
            X_XGBoost.columns = newX
            X_XGBoost = pd.concat([X, X_XGBoost])
            X_XGBoost = X_XGBoost.sort_values(by=['score'],
                                              ascending=False,
                                              axis=1)
            X_XGBoost = X_XGBoost.drop(['score'], axis=0)

            auc_val = []
            FS6 = "XGBClassifier"
            X_XGBoost1 = X_XGBoost.iloc[:, 0:20]
            # X_XGBoost1 = X_XGBoost.iloc[:, 0:10]
            fs_names.append(FS6)
            all_fs.append(X_XGBoost1)

            for clf, label in zip(all_clf, clf_labels):
                # print('10-fold cross validation:\n')
                scores = cross_val_score(estimator=clf,
                                         X=X_XGBoost1,
                                         y=y,
                                         cv=cv,
                                         scoring='roc_auc')
                # print("ROC AUC: %0.3f (+/- %0.3f) [%s]"% (scores.mean(), scores.std(), label))

                auc_val.append([FS6, label, np.mean(scores), np.std(scores)])
            result = sorted(auc_val, key=lambda s: s[2], reverse=True)
            XGBoost_result2 = pd.DataFrame(
                result, columns=["FS", "Classifier", "AUC", "STD"]).round(3)
            result1 = sorted(auc_val, key=lambda s: s[2], reverse=False)
            # print(result)
            # for FS, label, auc_mean, auc_std in result:
            #     print("%s, AUC = %0.3f +/- %0.3f" % (label, auc_mean, auc_std))
            # for FS, label, auc_mean, auc_std in result1:
            # print("%s, AUC = %0.3f +/- %0.3f" %(label, auc_mean, auc_std))
            # b = plt.barh(label, auc_mean, height=0.5)
            # plt.bar_label(b,fmt='%.3f',label_type='edge',fontsize=12)
            # plt.xlabel('AUC')

            X_XGBoost1_T = pd.DataFrame(np.transpose(X_XGBoost1.columns))
            X_XGBoost1_T.columns = ["Name of Feature"]
            X_XGBoost1_T.index = [
                "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                "13", "14", "15", "16", "17", "18", "19", "20"
            ]
            # X_XGBoost1_T.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            XGBoost_result2 = XGBoost_result2
            result_arr.append(XGBoost_result2)
            colors_dict['XGBClassifier'] = '#EDB096'
        elif i == '6':
            model = GradientBoostingClassifier(random_state=seed)
            model.fit(X, y)
            importance_abs = abs(model.feature_importances_)
            GB_score = pd.DataFrame(importance_abs, columns=['score'])
            GB_score = GB_score.T
            X_GB = pd.DataFrame(GB_score)
            X_GB.columns = newX
            X_GB = pd.concat([X, X_GB])
            X_GB = X_GB.sort_values(by=['score'], ascending=False, axis=1)
            X_GB = X_GB.drop(['score'], axis=0)

            auc_val = []
            FS7 = "Gradient Boosting Classifier"
            X_GB1 = X_GB.iloc[:, 0:20]
            # X_GB1 = X_GB.iloc[:, 0:10]
            fs_names.append(FS7)
            all_fs.append(X_GB1)

            for clf, label in zip(all_clf, clf_labels):
                # print('10-fold cross validation:\n')
                scores = cross_val_score(estimator=clf,
                                         X=X_GB1,
                                         y=y,
                                         cv=cv,
                                         scoring='roc_auc')
                # print("ROC AUC: %0.3f (+/- %0.3f) [%s]"% (scores.mean(), scores.std(), label))

                auc_val.append([FS7, label, np.mean(scores), np.std(scores)])
            result = sorted(auc_val, key=lambda s: s[2], reverse=True)
            GB_result2 = pd.DataFrame(
                result, columns=["FS", "Classifier", "AUC", "STD"]).round(3)
            result1 = sorted(auc_val, key=lambda s: s[2], reverse=False)
            # print(result)
            # for FS, label, auc_mean, auc_std in result:
            #     print("%s, AUC = %0.3f +/- %0.3f" % (label, auc_mean, auc_std))
            # for FS, label, auc_mean, auc_std in result1:
            # print("%s, AUC = %0.3f +/- %0.3f" %(label, auc_mean, auc_std))
            # b = plt.barh(label, auc_mean, height=0.5)
            # plt.bar_label(b,fmt='%.3f',label_type='edge',fontsize=12)
            # plt.xlabel('AUC')

            X_GB1_T = pd.DataFrame(np.transpose(X_GB1.columns))
            X_GB1_T.columns = ["Name of Feature"]
            X_GB1_T.index = [
                "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                "13", "14", "15", "16", "17", "18", "19", "20"
            ]
            # X_GB1_T.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            GB_result2 = GB_result2
            result_arr.append(GB_result2)
            colors_dict['Gradient Boosting Classifier'] = '#EDA1D3'
        elif i == '7':
            model = AdaBoostClassifier(random_state=seed)
            model.fit(X, y)
            importance_abs = abs(model.feature_importances_)
            AB_score = pd.DataFrame(importance_abs, columns=['score'])
            AB_score = AB_score.T
            X_AB = pd.DataFrame(AB_score)
            X_AB.columns = newX
            X_AB = pd.concat([X, X_AB])
            X_AB = X_AB.sort_values(by=['score'], ascending=False, axis=1)
            X_AB = X_AB.drop(['score'], axis=0)

            auc_val = []
            FS8 = "AdaBoost Classifier"
            X_AB1 = X_AB.iloc[:, 0:20]
            # X_AB1 = X_AB.iloc[:, 0:10]
            fs_names.append(FS8)
            all_fs.append(X_AB1)

            for clf, label in zip(all_clf, clf_labels):
                # print('10-fold cross validation:\n')
                scores = cross_val_score(estimator=clf,
                                         X=X_AB1,
                                         y=y,
                                         cv=cv,
                                         scoring='roc_auc')
                # print("ROC AUC: %0.3f (+/- %0.3f) [%s]"% (scores.mean(), scores.std(), label))

                auc_val.append([FS8, label, np.mean(scores), np.std(scores)])
            result = sorted(auc_val, key=lambda s: s[2], reverse=True)
            AB_result2 = pd.DataFrame(
                result, columns=["FS", "Classifier", "AUC", "STD"]).round(3)
            result1 = sorted(auc_val, key=lambda s: s[2], reverse=False)
            # print(result)
            # for FS, label, auc_mean, auc_std in result:
            #     print("%s, AUC = %0.3f +/- %0.3f" % (label, auc_mean, auc_std))
            # for FS, label, auc_mean, auc_std in result1:
            # print("%s, AUC = %0.3f +/- %0.3f" %(label, auc_mean, auc_std))
            # b = plt.barh(label, auc_mean, height=0.5)
            # plt.bar_label(b,fmt='%.3f',label_type='edge',fontsize=12)
            # plt.xlabel('AUC')

            X_AB1_T = pd.DataFrame(np.transpose(X_AB1.columns))
            X_AB1_T.columns = ["Name of Feature"]
            X_AB1_T.index = [
                "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                "13", "14", "15", "16", "17", "18", "19", "20"
            ]
            # X_AB1_T.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            AB_result2 = AB_result2
            result_arr.append(AB_result2)
            colors_dict['AdaBoost Classifier'] = '#A896ED'

    Barh_auc1 = pd.concat(result_arr).sort_values(by=["AUC", "STD"],
                                                  ascending=[False, True])
    Barh_auc1 = Barh_auc1.reset_index(drop=True)
    Barh_auc2 = pd.concat(result_arr).sort_values(by=["AUC", "STD"],
                                                  ascending=[True, False])
    Barh_auc2 = Barh_auc2.reset_index(drop=True)
    if len(Barh_auc2) > 10:
        Barh_auc2 = Barh_auc2.tail(10)

    colors = [colors_dict[i] for i in Barh_auc2.FS.values]
    legend_elements = [
        Patch(facecolor=color, label=name, alpha=0.8)
        for name, color in colors_dict.items()
    ]
    Barh_auc3 = Barh_auc2.tail(1)
    featurearr.clear()
    if Barh_auc3.FS.values == 'Linear Regression':
        for i in range(0, 20):
            featurearr.append(X_LiR1.columns[i])

    elif Barh_auc3.FS.values == 'Logistic Regression':
        for i in range(0, 20):
            featurearr.append(X_LoR1.columns[i])
    elif Barh_auc3.FS.values == 'SVC-Linear':
        for i in range(0, 20):
            featurearr.append(X_SVC1.columns[i])
    elif Barh_auc3.FS.values == 'Decision Tree Classifier':
        for i in range(0, 20):
            featurearr.append(X_DT1.columns[i])
    elif Barh_auc3.FS.values == 'Random Forest Classifier':
        for i in range(0, 20):
            featurearr.append(X_RF1.columns[i])
    elif Barh_auc3.FS.values == 'XGBClassifier':
        for i in range(0, 20):
            featurearr.append(X_XGBoost1.columns[i])
    elif Barh_auc3.FS.values == 'Gradient Boosting Classifier':
        for i in range(0, 20):
            featurearr.append(X_GB1.columns[i])
    elif Barh_auc3.FS.values == 'AdaBoost Classifier':
        for i in range(0, 20):
            featurearr.append(X_AB1.columns[i])

    Barh_auc4 = Barh_auc1
    fsarr.clear()
    modelarr.clear()
    aucarr.clear()
    stdarr.clear()
    for i in range(len(Barh_auc4)):
        fsarr.append(Barh_auc4['FS'][i])
        modelarr.append(Barh_auc4['Classifier'][i])
        aucarr.append(Barh_auc4['AUC'][i])
        stdarr.append(Barh_auc4['STD'][i])

    fig_w = 5
    fig_h = 5
    plt.figure(figsize=(75, len(Barh_auc2) * 7))

    b = plt.barh("FS: " + Barh_auc2["FS"] + "\n CLF: " +
                 Barh_auc2["Classifier"],
                 Barh_auc2["AUC"],
                 height=0.5,
                 color=colors)
    plt.bar_label(b, fmt='%.3f', label_type='edge')
    plt.title('AUC BarChart\n', fontsize=110)
    plt.xlabel('\nAUC', fontsize=100)
    plt.yticks(fontsize=100)
    plt.xticks(fontsize=100)
    plt.xlim(0.75, 1.05)
    plt.legend(handles=legend_elements,
               bbox_to_anchor=(1.04, 0.0),
               loc="lower left",
               fontsize=100)
    bwith = 4
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)
    for i, val in enumerate(Barh_auc2.AUC.values):
        plt.text(val, i, val, fontsize=80)

    # plt.show()
    # fig = plt.gcf()
    # plt.show()
    plt.savefig("./static/images/Plot.png", bbox_inches='tight')
    # plt.clf()

    #####################################################################
    #####################################################################
    df_len = Barh_auc1["AUC"].size
    # print(df_len)
    # print(all_fs)
    # print(fs_names)
    if df_len < 3:
        # print("印全部！")小於3條全畫
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot([0, 1], [0, 1],
                linestyle="--",
                lw=2,
                color="blue",
                label="Chance",
                alpha=0.8)
        for fs, fs_name in zip(all_fs, fs_names):
            for clf, label in zip(all_clf, clf_labels):
                aucs = []
                tprs = []
                for train_index, test_index in cv.split(fs, y):
                    X_train, X_test = fs.iloc[train_index], fs.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    clf.fit(X_train, y_train)
                    # y_pred = pd.DataFrame(clf.predict(X_test))

                    # a = confusion_matrix(y_test, y_pred)
                    # conf_mat10 = pd.DataFrame(confusion_matrix(y_test, y_pred))
                    # conf_mat10.columns = ['P', 'N']
                    # conf_mat10.index = ['P', 'N']
                    # print(conf_mat10)
                    # print('\n')

                    y_score = clf.predict_proba(X_test)[:, 1]
                    fpr, tpr, thresholds = roc_curve(y_test, y_score)

                    inter_tpr = 0.0
                    mean_fpr = np.linspace(0, 1, 100)
                    inter_tpr = interp(mean_fpr, fpr, tpr)
                    # inter_tpr[0] = 0
                    tprs.append(inter_tpr)

                    roc_auc = auc(mean_fpr, inter_tpr)
                    aucs.append(roc_auc)

                mean_tpr = np.mean(tprs, axis=0)
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs)
                std_tpr = np.std(tprs, axis=0)
                mean_tpr[0] = 0

                plt.plot(
                    mean_fpr,
                    mean_tpr,
                    label=r"FS:%s / CLF:%s (AUC = %0.3f $\pm$ %0.3f)" %
                    (fs_name, label, mean_auc, std_auc),
                    lw=2,
                    alpha=0.8,
                )
                plt.xlabel('False Positive Rate', fontsize=30)
                plt.ylabel('True Positive Rate', fontsize=30)
                T = r"ROC curve"
                plt.title(T, fontsize=30)
                plt.legend(loc="lower right", fontsize=10)
                plt.savefig("./static/images/roc.png", bbox_inches='tight')
                # plt.show()
    else:
        # 最高的那條
        auc_fig_row1 = Barh_auc1.head(1)
        auc_fig_row1_fs = auc_fig_row1["FS"]
        auc_fig_row1_clf = auc_fig_row1["Classifier"]
        # print(Barh_auc1)
        # print("11111111111111")
        # print("auc_fig_row1")
        # print(auc_fig_row1)
        # print("auc_fig_row1_fs")
        # print(auc_fig_row1_fs)
        # print("auc_fig_row1_clf")
        # print(auc_fig_row1_clf)
        if auc_fig_row1_fs.item() == "Linear Regression":
            X1 = X_LiR1
        elif auc_fig_row1_fs.item() == "Logistic Regression":
            X1 = X_LoR1
        elif auc_fig_row1_fs.item() == "SVC-Linear":
            X1 = X_SVC1
        elif auc_fig_row1_fs.item() == "Decision Tree Classifier":
            X1 = X_DT1
        elif auc_fig_row1_fs.item() == "Random Forest Classifier":
            X1 = X_RF1
        elif auc_fig_row1_fs.item() == "XGBClassifier":
            X1 = X_XGBoost1
        elif auc_fig_row1_fs.item() == "Gradient Boosting Classifier":
            X1 = X_GB1
        elif auc_fig_row1_fs.item() == "AdaBoost Classifier":
            X1 = X_AB1

        if auc_fig_row1_clf.item() == "Logistic Regression":
            clf1 = LogisticRegression(random_state=seed, C=lr1, solver=lr2)
        elif auc_fig_row1_clf.item() == "SVC-Linear":
            clf1 = SVC(kernel='linear',
                       random_state=seed,
                       C=svclinear1,
                       probability=True)
        elif auc_fig_row1_clf.item() == "SVC-Polynomial":
            clf1 = SVC(kernel='poly',
                       random_state=seed,
                       C=svcpoly1,
                       degree=svcpoly2,
                       probability=True)
        elif auc_fig_row1_clf.item() == "SVC-RBF":
            clf1 = SVC(kernel='rbf',
                       random_state=seed,
                       C=svcrbf1,
                       probability=True)
        elif auc_fig_row1_clf.item() == "Decision Tree Classifier":
            clf1 = DecisionTreeClassifier(random_state=seed, max_depth=dt1)
        elif auc_fig_row1_clf.item() == "Random Forest Classifier":
            clf1 = RandomForestClassifier(random_state=seed,
                                          n_estimators=rf1,
                                          max_depth=rf2)
        elif auc_fig_row1_clf.item() == "XGBClassifier":
            clf1 = XGBClassifier(random_state=seed,
                                 n_estimators=xgb1,
                                 max_depth=xgb2)
        elif auc_fig_row1_clf.item() == "Gradient Boosting Classifier":
            clf1 = GradientBoostingClassifier(random_state=seed,
                                              n_estimators=gb1,
                                              learning_rate=gb2)
        elif auc_fig_row1_clf.item() == "AdaBoost Classifier":
            clf1 = AdaBoostClassifier(random_state=seed,
                                      n_estimators=ada1,
                                      learning_rate=ada2)
        elif auc_fig_row1_clf.item() == "Multi-Layer Perceptron":
            if mlp2 == '1':
                clf1 = MLPClassifier(random_state=seed,
                                     solver=mlp1,
                                     hidden_layer_sizes=(mlp2num1))
            else:
                clf1 = MLPClassifier(random_state=seed,
                                     solver=mlp1,
                                     hidden_layer_sizes=(mlp2num1, mlp2num2))

        # 中間那條
        if df_len % 2 == 0:
            Barh_auc3 = Barh_auc1.drop([df_len - 1])
            med = Barh_auc3["AUC"].median(axis=0)
        else:
            Barh_auc3 = Barh_auc1
            med = Barh_auc1["AUC"].median(axis=0)
        mask = Barh_auc3["AUC"] == med
        auc_fig_row2 = Barh_auc3[mask]
        auc_fig_row2 = auc_fig_row2.head(1)
        auc_fig_row2_fs = auc_fig_row2["FS"]
        auc_fig_row2_clf = auc_fig_row2["Classifier"]
        # print("22222222222222222")
        # print("auc_fig_row2")
        # print(auc_fig_row2)
        # print("auc_fig_row2_fs")
        # print(auc_fig_row2_fs)
        # print("auc_fig_row2_clf")
        # print(auc_fig_row2_clf)

        if auc_fig_row2_fs.item() == "Linear Regression":
            X2 = X_LiR1
        elif auc_fig_row2_fs.item() == "Logistic Regression":
            X2 = X_LoR1
        elif auc_fig_row2_fs.item() == "SVC-Linear":
            X2 = X_SVC1
        elif auc_fig_row2_fs.item() == "Decision Tree Classifier":
            X2 = X_DT1
        elif auc_fig_row2_fs.item() == "Random Forest Classifier":
            X2 = X_RF1
        elif auc_fig_row2_fs.item() == "XGBClassifier":
            X2 = X_XGBoost1
        elif auc_fig_row2_fs.item() == "Gradient Boosting Classifier":
            X2 = X_GB1
        elif auc_fig_row2_fs.item() == "AdaBoost Classifier":
            X2 = X_AB1

        if auc_fig_row2_clf.item() == "Logistic Regression":
            clf2 = LogisticRegression(random_state=seed, C=lr1, solver=lr2)
        elif auc_fig_row2_clf.item() == "SVC-Linear":
            clf2 = SVC(kernel='linear',
                       random_state=seed,
                       C=svclinear1,
                       probability=True)
        elif auc_fig_row2_clf.item() == "SVC-Polynomial":
            clf2 = SVC(kernel='poly',
                       random_state=seed,
                       C=svcpoly1,
                       degree=svcpoly2,
                       probability=True)
        elif auc_fig_row2_clf.item() == "SVC-RBF":
            clf2 = SVC(kernel='rbf',
                       random_state=seed,
                       C=svcrbf1,
                       probability=True)
        elif auc_fig_row2_clf.item() == "Decision Tree Classifier":
            clf2 = DecisionTreeClassifier(random_state=seed, max_depth=dt1)
        elif auc_fig_row2_clf.item() == "Random Forest Classifier":
            clf2 = RandomForestClassifier(random_state=seed,
                                          n_estimators=rf1,
                                          max_depth=rf2)
        elif auc_fig_row2_clf.item() == "XGBClassifier":
            clf2 = XGBClassifier(random_state=seed,
                                 n_estimators=xgb1,
                                 max_depth=xgb2)
        elif auc_fig_row2_clf.item() == "Gradient Boosting Classifier":
            clf2 = GradientBoostingClassifier(random_state=seed,
                                              n_estimators=gb1,
                                              learning_rate=gb2)
        elif auc_fig_row2_clf.item() == "AdaBoost Classifier":
            clf2 = AdaBoostClassifier(random_state=seed,
                                      n_estimators=ada1,
                                      learning_rate=ada2)
        elif auc_fig_row2_clf.item() == "Multi-Layer Perceptron":
            if mlp2 == '1':
                clf2 = MLPClassifier(random_state=seed,
                                     solver=mlp1,
                                     hidden_layer_sizes=(mlp2num1))
            else:
                clf2 = MLPClassifier(random_state=seed,
                                     solver=mlp1,
                                     hidden_layer_sizes=(mlp2num1, mlp2num2))

        # 最低那條
        auc_fig_row3 = Barh_auc1.tail(1)
        auc_fig_row3_fs = auc_fig_row3["FS"]
        auc_fig_row3_clf = auc_fig_row3["Classifier"]
        # print("333333333333333")
        # print("auc_fig_row3")
        # print(auc_fig_row3)
        # print("auc_fig_row3_fs")
        # print(auc_fig_row3_fs)
        # print("auc_fig_row3_clf")
        # print(auc_fig_row3_clf)

        if auc_fig_row3_fs.item() == "Linear Regression":
            X3 = X_LiR1
        elif auc_fig_row3_fs.item() == "Logistic Regression":
            X3 = X_LoR1
        elif auc_fig_row3_fs.item() == "SVC-Linear":
            X3 = X_SVC1
        elif auc_fig_row3_fs.item() == "Decision Tree Classifier":
            X3 = X_DT1
        elif auc_fig_row3_fs.item() == "Random Forest Classifier":
            X3 = X_RF1
        elif auc_fig_row3_fs.item() == "XGBClassifier":
            X3 = X_XGBoost1
        elif auc_fig_row3_fs.item() == "Gradient Boosting Classifier":
            X3 = X_GB1
        elif auc_fig_row3_fs.item() == "AdaBoost Classifier":
            X3 = X_AB1

        if auc_fig_row3_clf.item() == "Logistic Regression":
            clf3 = LogisticRegression(random_state=seed, C=lr1, solver=lr2)
        elif auc_fig_row3_clf.item() == "SVC-Linear":
            clf3 = SVC(kernel='linear',
                       random_state=seed,
                       C=svclinear1,
                       probability=True)
        elif auc_fig_row3_clf.item() == "SVC-Polynomial":
            clf3 = SVC(kernel='poly',
                       random_state=seed,
                       C=svcpoly1,
                       degree=svcpoly2,
                       probability=True)
        elif auc_fig_row3_clf.item() == "SVC-RBF":
            clf3 = SVC(kernel='rbf',
                       random_state=seed,
                       C=svcrbf1,
                       probability=True)
        elif auc_fig_row3_clf.item() == "Decision Tree Classifier":
            clf3 = DecisionTreeClassifier(random_state=seed, max_depth=dt1)
        elif auc_fig_row3_clf.item() == "Random Forest Classifier":
            clf3 = RandomForestClassifier(random_state=seed,
                                          n_estimators=rf1,
                                          max_depth=rf2)
        elif auc_fig_row3_clf.item() == "XGBClassifier":
            clf3 = XGBClassifier(random_state=seed,
                                 n_estimators=xgb1,
                                 max_depth=xgb2)
        elif auc_fig_row3_clf.item() == "Gradient Boosting Classifier":
            clf3 = GradientBoostingClassifier(random_state=seed,
                                              n_estimators=gb1,
                                              learning_rate=gb2)
        elif auc_fig_row3_clf.item() == "AdaBoost Classifier":
            clf3 = AdaBoostClassifier(random_state=seed,
                                      n_estimators=ada1,
                                      learning_rate=ada2)
        elif auc_fig_row3_clf.item() == "Multi-Layer Perceptron":
            if mlp2 == '1':
                clf3 = MLPClassifier(random_state=seed,
                                     solver=mlp1,
                                     hidden_layer_sizes=(mlp2num1))
            else:
                clf3 = MLPClassifier(random_state=seed,
                                     solver=mlp1,
                                     hidden_layer_sizes=(mlp2num1, mlp2num2))

        # 畫圖！！
        aucs = []
        tprs = []
        ax1 = plt.figure(figsize=(10, 10))
        ax1 = plt.plot([0, 1], [0, 1],
                       linestyle="--",
                       lw=2,
                       color="darkgray",
                       label="Chance",
                       alpha=0.8)
        i = 0
        for train_index, test_index in cv.split(X1, y):
            i += 1
            X_train, X_test = X1.iloc[train_index], X1.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf1.fit(X_train, y_train)
            # y_pred = pd.DataFrame(clf1.predict(X_test))

            # a = confusion_matrix(y_test, y_pred)
            # conf_mat10 = pd.DataFrame(confusion_matrix(y_test, y_pred))
            # conf_mat10.columns = ['P', 'N']
            # conf_mat10.index = ['P', 'N']
            # print(conf_mat10)
            # print('\n')

            y_score = clf1.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_score)

            inter_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 200)
            inter_tpr = interp(mean_fpr, fpr, tpr)
            # inter_tpr[0] = 0
            tprs.append(inter_tpr)

            roc_auc = auc(mean_fpr, inter_tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        mean_tpr[0] = 0

        # axes = plt.axes()
        # axes.set_xlim([-0.05, 1.05])
        # axes.set_ylim([-0.05, 1.05])
        plt.plot(mean_fpr,
                 mean_tpr,
                 label=r"FS: %s / CLF: %s (AUC = %0.3f $\pm$ %0.3f)" %
                 (auc_fig_row1_fs.item(), auc_fig_row1_clf.item(), mean_auc,
                  std_auc),
                 lw=2,
                 alpha=0.8,
                 color="crimson")
        # print('------------------------------------------')
        # print('mean_fpr')
        # print(mean_fpr)
        # print('mean_tpr')
        # print(mean_tpr)
        # print('------------------------------------------')
        # print('auc_fig_row1_fs.item()')
        # print(auc_fig_row1_fs.item())
        # print('auc_fig_row1_clf.item()')
        # print(auc_fig_row1_clf.item())

        aucs = []
        tprs = []
        i = 0
        for train_index, test_index in cv.split(X2, y):
            i += 1
            X_train, X_test = X2.iloc[train_index], X2.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf2.fit(X_train, y_train)
            # y_pred = pd.DataFrame(clf2.predict(X_test))

            # a = confusion_matrix(y_test, y_pred)
            # conf_mat10 = pd.DataFrame(confusion_matrix(y_test, y_pred))
            # conf_mat10.columns = ['P', 'N']
            # conf_mat10.index = ['P', 'N']
            # print(conf_mat10)
            # print('\n')

            y_score = clf2.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_score)

            inter_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 200)
            inter_tpr = interp(mean_fpr, fpr, tpr)
            # inter_tpr[0] = 0
            tprs.append(inter_tpr)

            roc_auc = auc(mean_fpr, inter_tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        mean_tpr[0] = 0

        # axes = plt.axes()
        # axes.set_xlim([-0.05, 1.05])
        # axes.set_ylim([-0.05, 1.05])
        ax1 = plt.plot(mean_fpr,
                       mean_tpr,
                       label=r"FS: %s / CLF: %s (AUC = %0.3f $\pm$ %0.3f)" %
                       (auc_fig_row2_fs.item(), auc_fig_row2_clf.item(),
                        mean_auc, std_auc),
                       lw=2,
                       alpha=0.8,
                       color="royalblue")

        # print('------------------------------------------')
        # print('auc_fig_row2_fs')
        # print(auc_fig_row2_fs)
        # print('auc_fig_row2_clf')
        # print(auc_fig_row2_clf)
        aucs = []
        tprs = []
        i = 0
        for train_index, test_index in cv.split(X3, y):
            i += 1
            X_train, X_test = X3.iloc[train_index], X3.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf3.fit(X_train, y_train)
            # y_pred = pd.DataFrame(clf3.predict(X_test))

            # a = confusion_matrix(y_test, y_pred)
            # conf_mat10 = pd.DataFrame(confusion_matrix(y_test, y_pred))
            # conf_mat10.columns = ['P', 'N']
            # conf_mat10.index = ['P', 'N']
            # print(conf_mat10)
            # print('\n')

            y_score = clf3.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_score)

            inter_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 200)
            inter_tpr = interp(mean_fpr, fpr, tpr)
            # inter_tpr[0] = 0
            tprs.append(inter_tpr)

            roc_auc = auc(mean_fpr, inter_tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        mean_tpr[0] = 0

        # axes = plt.axes()
        # axes.set_xlim([-0.05, 1.05])
        # axes.set_ylim([-0.05, 1.05])
        ax1 = plt.plot(mean_fpr,
                       mean_tpr,
                       label=r"FS: %s / CLF: %s (AUC = %0.3f $\pm$ %0.3f)" %
                       (auc_fig_row3_fs.item(), auc_fig_row3_clf.item(),
                        mean_auc, std_auc),
                       lw=2,
                       alpha=0.8,
                       color="mediumseagreen")

        # print('------------------------------------------')
        # print('mean_fpr')
        # print(mean_fpr)
        # print('mean_tpr')
        # print(mean_tpr)
        # print('------------------------------------------')
        # print('auc_fig_row3_fs.item()')
        # print(auc_fig_row3_fs.item())
        # print('auc_fig_row3_clf.item()')
        # print(auc_fig_row3_clf.item())
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        T = r"ROC curve"
        plt.title(T, fontsize=20)
        plt.legend(bbox_to_anchor=(1.005, 0.0), loc="lower right", fontsize=10)
        plt.savefig("./static/images/Roc.png", bbox_inches='tight')
        # plt.show()

    print("finalcheck ok")

    return render(request, 'home/final.html', locals())  # JsonResponse({})


@csrf_exempt
def finaljump(request):
    print("finaljump ok")

    return render(request, 'home/final.html', locals())


@csrf_exempt
def finalsend(request):
    print("finalsend ok")

    return JsonResponse({
        'modelarr': modelarr,
        'fsarr': fsarr,
        'aucarr': aucarr,
        'featurearr': featurearr,
        'stdarr': stdarr
    })


@csrf_exempt
def download_plot(request):
    print("download_plot ok")
    the_file_name = 'Plot.png'  # 显示在弹出对话框中的默认的下载文件名
    filename = 'D:/CSMU_AI/csmuproject/picture/Plot.png'  # 要下载的文件路径
    response = StreamingHttpResponse(readFile(filename))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format(
        the_file_name)

    return response


@csrf_exempt
def download_roc(request):
    print("download_roc ok")
    the_file_name = 'roc.png'  # 显示在弹出对话框中的默认的下载文件名
    filename = 'D:/CSMU_AI/csmuproject/picture/roc.png'  # 要下载的文件路径
    response = StreamingHttpResponse(readFile(filename))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format(
        the_file_name)

    return response


@csrf_exempt
def readFile(filename, chunk_size=512):
    with open(filename, 'rb') as f:
        while True:
            c = f.read(chunk_size)
            if c:
                yield c
            else:
                break


@csrf_exempt
def download_feature(request):
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename="feature_rank.xls"'
    wb = xlwt.Workbook(encoding='utf-8')

    # feature_rank
    ws = wb.add_sheet(
        'feature_rank')  # this will make a sheet named Users Data

    # Sheet header, first row
    row_num = 0

    font_style = xlwt.XFStyle()
    font_style.font.bold = True

    columns = ['Rank', 'Feature Name']

    for col_num in range(len(columns)):
        ws.write(row_num, col_num, columns[col_num],
                 font_style)  # at 0 row 0 column

    # Sheet body, remaining rows
    font_style = xlwt.XFStyle()

    rows = featurearr
    for col_num in range(len(rows)):
        row_num += 1
        ws.write(row_num, 0, row_num, font_style)
        ws.write(row_num, 1, rows[col_num], font_style)

    # auc
    ws = wb.add_sheet('auc')  # this will make a sheet named Users Data

    # Sheet header, first row
    row_num = 0

    font_style = xlwt.XFStyle()
    font_style.font.bold = True

    columns = ['Rank', 'Feature Selection', 'Classifier', 'AUC']

    for col_num in range(len(columns)):
        ws.write(row_num, col_num, columns[col_num],
                 font_style)  # at 0 row 0 column

    # Sheet body, remaining rows
    font_style = xlwt.XFStyle()

    fsrows = fsarr
    modelrows = modelarr
    aucrows = aucarr
    stdrows = stdarr
    for col_num in range(len(fsrows)):
        row_num += 1
        auc = format(float(aucrows[col_num]), '.3f') + " ± " + format(
            float(stdrows[col_num]), '.3f')
        ws.write(row_num, 0, row_num, font_style)
        ws.write(row_num, 1, fsrows[col_num], font_style)
        ws.write(row_num, 2, modelrows[col_num], font_style)
        ws.write(row_num, 3, auc, font_style)

    # Plot.png
    # ws = wb.add_sheet('BarChart')  # this will make a sheet named Users Data
    # # ws.insert_image('B10', r'C:/Users/User/PycharmProjects/csmuProject/static/images/roc.png')
    # file_in = r'C:/Users/User/PycharmProjects/csmuProject/static/images/Plot.png'
    # img = Image.open(file_in)
    # file_out = 'test1.bmp'
    # print(len(img.split()))  # test
    # if len(img.split()) == 4:
    #     # prevent IOError: cannot write mode RGBA as BMP
    #     r, g, b, a = img.split()
    #     img = Image.merge("RGB", (r, g, b))
    #     img.save(file_out)
    # else:
    #     img.save(file_out)
    # ws.insert_bitmap(file_out, 0, 0)
    #
    # # Plot.png
    # ws = wb.add_sheet('ROCplot')  # this will make a sheet named Users Data
    # # ws.insert_image('B10', r'C:/Users/User/PycharmProjects/csmuProject/static/images/roc.png')
    # file_in = r'C:/Users/User/PycharmProjects/csmuProject/static/images/roc.png'
    # img = Image.open(file_in)
    # file_out = 'test2.bmp'
    # print(len(img.split()))  # test
    # if len(img.split()) == 4:
    #     # prevent IOError: cannot write mode RGBA as BMP
    #     r, g, b, a = img.split()
    #     img = Image.merge("RGB", (r, g, b))
    #     img.save(file_out)
    # else:
    #     img.save(file_out)
    # ws.insert_bitmap(file_out, 0, 0)

    wb.save(response)

    return response


@csrf_exempt
def eacel_img(request):
    wb = xlsxwriter.Workbook('feature_rank.xlsx')
    ws = wb.add_worksheet('BarChart')
    ws.insert_image('B2', r'D:/CSMU_AI/csmuproject/picture/Plot.png')
    wb.close()
    workbook = xlsxwriter.Workbook(
        'D:/CSMU_AI/csmuproject/picture/Downloadshello.xlsx')
    worksheet = workbook.add_worksheet()

    filename = r'D:/CSMU_AI/csmuproject/static/images/Plot.png'

    file = open(filename, 'rb')
    data = BytesIO(file.read())
    file.close()

    worksheet.insert_image('C5', filename, {'image_data': data})

    workbook.close()
    filename = 'django_simple.xlsx'
    response = HttpResponse(
        content_type=
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=%s' % filename

    return response


def download_feature(request):
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename="feature_rank.xls"'
    wb = xlsxwriter.Workbook()

    # feature_rank
    ws = wb.add_worksheet(
        'feature_rank')  # this will make a sheet named Users Data

    # Sheet header, first row

    font_style = xlwt.XFStyle()
    font_style.font.bold = True

    columns = ['Rank', 'Feature Name']
    ws.write('A0', columns[0], font_style)
    ws.write('B0', columns[1], font_style)

    # Sheet body, remaining rows
    font_style = xlwt.XFStyle()

    row_num = 0
    rows = featurearr
    for col_num in range(len(rows)):
        row_num += 1
        position1 = 'A' + str(row_num)
        position2 = 'B' + str(row_num)
        ws.write(position1, row_num, font_style)
        ws.write(position2, rows[col_num], font_style)

    # auc
    ws = wb.add_worksheet('auc')  # this will make a sheet named Users Data

    # Sheet header, first row

    font_style = xlwt.XFStyle()
    font_style.font.bold = True

    columns = ['Rank', 'Feature Selection', 'Classifier', 'AUC']
    ws.write('A0', columns[0], font_style)
    ws.write('A1', columns[1], font_style)
    ws.write('A2', columns[2], font_style)
    ws.write('A3', columns[3], font_style)

    # Sheet body, remaining rows
    font_style = xlwt.XFStyle()
    row_num = 0

    fsrows = fsarr
    modelrows = modelarr
    aucrows = aucarr
    stdrows = stdarr
    for col_num in range(len(fsrows)):
        row_num += 1
        position1 = 'A' + str(row_num)
        position2 = 'B' + str(row_num)
        position3 = 'C' + str(row_num)
        position4 = 'D' + str(row_num)
        auc = format(float(aucrows[col_num]), '.3f') + " ± " + format(
            float(stdrows[col_num]), '.3f')
        ws.write(position1, row_num, font_style)
        ws.write(position2, fsrows[col_num], font_style)
        ws.write(position3, modelrows[col_num], font_style)
        ws.write(position4, auc, font_style)

    # Plot.png
    ws = wb.add_worksheet('Plot')  # this will make a sheet named Users Data
    ws.insert_image('B2', r'D:/CSMU_AI/csmuproject/picture/Plot.png')
    ws.insert_image('B10', r'D:/CSMU_AI/csmuproject/picture/roc.png')

    wb.close()
    # wb.save(response)

    return response
