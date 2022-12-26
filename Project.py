import tkinter as tk
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import pandas as pd
from pandas import DataFrame
import numpy as np
import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

#####################################  Path of Training and Testing Files   ##########################################
rf = r"train.json"
rf_test = r"test.json"


#######################################   Reading Data Files  ###########################################################
train_data = pd.read_json(rf)      
test_data = pd.read_json(rf_test)       
print("File Reading Completed.\n")

#######################################   Saving Testing ID  ###########################################################
test_id_list = test_data["id"].tolist()

def string_com(x):
    ans = ' '.join(x)
    ans = porter.stem(ans)
    return ans

######################################   Ingredients combine and saving as sentence  ####################################
train_data["ingredients"] = train_data["ingredients"].apply(string_com)
test_data["ingredients"] = test_data["ingredients"].apply(string_com)

######################################   Tf_Idf Vectorizer   ########################################################### 
transformer = TfidfVectorizer()
transformer.fit(train_data["ingredients"])
train_data_vect = transformer.transform(train_data["ingredients"])
test_data_vect = transformer.transform(test_data["ingredients"])

X_train,X_test,y_train,y_test = train_test_split(train_data_vect,train_data["cuisine"],random_state=0,train_size=0.9)
# print("Splitting completed")

def file_write(path, dict1):
    f = open(path,"w")
    f.write(str(dict1))
    f.close()

def dict_of_models(list_cus):
    dict_cus = {}
    i = 0
    for idc in test_id_list:
        dict_cus[idc] = list_cus[i]
        i += 1
    return dict_cus

def number_predictions(pre_list):
    results = pd.DataFrame()
    results["test"] = y_test
    results["predict"] = pre_list

    results_incorrect = results[results["test"]!=results["predict"]]
    results_correct = results[results["test"]==results["predict"]]

    len_correct = len(results_correct)
    len_incorrect = len(results_incorrect)
    return len_correct, len_incorrect 
    # print ("Number of cuisines predicted correctly :",len(results_correct))
    # print ("Number of cuisines not predicted correctly :",len(results_incorrect))

window = tk.Tk()
window.title(" Whats Cooking? Predicting Cuisine. ")
display = tk.Canvas(window, width=800, height=500)      #### Creating GUI window #####
display.pack()

image = Image.open(r"try4.jpeg")
image = image.resize((800,800), Image.ANTIALIAS)

img = ImageTk.PhotoImage(image)      ## Image loading ##
imglabel = tk.Label(window, image=img)
display.create_window(400, 200, window=imglabel)

label1 = tk.Label(window, text='Whats Cooking? Predicting Cuisine.')
label1.config(font=('Arial', 16))
display.create_window(400, 40, window=label1)

##########################################             KNN Model                 ####################################
#################################    Saving K values    ################################

from sklearn import metrics                                     
from sklearn.metrics import classification_report

knn = KNeighborsClassifier(n_neighbors=5)
# knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
pred_test = knn.predict(X_test)
# print(pred_test)
final_predict = knn.predict(test_data_vect)          ###############    Final KNN   #####################
# print("\n Actual Predict\n",final_predict)


acc_knn = (metrics.accuracy_score(y_test, pred_test))*100
# print("Accuracy KNN Model:",acc_knn)
prec_rec_knn = classification_report(y_test, pred_test)
# print("Precision and Recall:\n", classification_report(y_test, pred_test))

#################################   Saving Output in File    #######################################
dict_id_cuisine = dict_of_models(final_predict)
path = 'output_knn.txt'
file_write(path, dict_id_cuisine)
print("Result written in Output File(KNN)")
print("\n********************************************************************************\n")

def knn_btn():
    label_knn = tk.Label(window, text=acc_knn) 
    display.create_window(70, 250, window=label_knn)

knn_pred_numb_crt, knn_pred_numb_incrt = number_predictions(pred_test)


#####################################            Logistic Regression             ###################################
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=3)                              ###########   Increasing Accuracy by decreasing C value  ##########
logreg.fit(X_train,y_train)
logistic_pred=logreg.predict(X_test)                          #############    Train Test Own chech  ##################
log_predict = logreg.predict(test_data_vect)                  #############  Final List  Logistic  ###################
# print("\n Logictic regression Predict\n",log_predict)
acc_log = (metrics.accuracy_score(y_test, logistic_pred))*100
# print("\nAccuracy Logistic Model:", acc_log)
prec_rec_log = classification_report(y_test, logistic_pred)
# print("Precision and Recall:\n", classification_report(y_test, logistic_pred))

#################################   Saving Output in File    #######################################
dict_id_cuisine_log_regr = dict_of_models(log_predict)
path_log = 'output_log_reg.txt'
file_write(path_log, dict_id_cuisine_log_regr)
print("Result written in Output File(log_reg)")
print("\n********************************************************************************\n")

log_pred_numb_crt, log_pred_numb_incrt = number_predictions(logistic_pred)

def log_btn():
    label_log = tk.Label(window, text=acc_log) 
    display.create_window(270, 250, window=label_log)

##############################                    NAIVE BAYES(MULTINOMIAL)               ###################################

# from sklearn.cross_validation import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(train_data_vect,train_data["cuisine"],random_state=109,train_size=0.9)
# from sklearn.naive_bayes import GaussianNB   #############   accuracy 24% approx  ##########
from sklearn.naive_bayes import MultinomialNB  #############   accuracy 68% approx  ##########

NB = MultinomialNB(alpha=0.5)
NB.fit(X_train.toarray(), y_train)
naive_bayes = NB.predict(X_test.toarray())
naive_bayes_predict = NB.predict(test_data_vect.toarray())                ############  Final Naive Bayes  #########################
acc_naive = (metrics.accuracy_score(y_test, naive_bayes))*100
# print("Accuracy of Naive Bayes:",acc_naive)
prec_rec_nb = classification_report(y_test, naive_bayes)
# print("Precision and Recall:\n", classification_report(y_test, naive_bayes))

#################################   Saving Output in File    #######################################
dict_id_cuisine_naive = dict_of_models(naive_bayes_predict)
path_naive = 'output_naive.txt'
file_write(path_naive, dict_id_cuisine_naive)
print("Result written in Output File(naive)")
print("\n********************************************************************************\n")

def nb_btn():
    label_nb = tk.Label(window, text=acc_naive)
    display.create_window(470, 250, window=label_nb)

nb_pred_numb_crt, nb_pred_numb_incrt = number_predictions(naive_bayes)

####################################                SGD CLASSIFIER                ##########################################
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier

sgd_clf = linear_model.SGDClassifier()
sgd_clf.fit(X_train, y_train)
SGD_train = sgd_clf.predict(X_test)
SGD_final_predict = sgd_clf.predict(test_data_vect)                       ############  Final SGD  #########################
acc_sgd = (metrics.accuracy_score(y_test, SGD_train))*100
# print("Accuracy of SGD classifier:",acc_sgd)
prec_rec_sgd = classification_report(y_test, SGD_train)
# print("Precision and Recall:\n", classification_report(y_test, SGD_train))

#################################   Saving Output in File    #######################################
dict_id_cuisine_SGD = dict_of_models(SGD_final_predict)
path_SGD = 'output_SGD.txt'
file_write(path_SGD, dict_id_cuisine_SGD)
print("Result written in Output File(SGD)")
print("\n********************************************************************************\n")

def sgd_btn():
    label_sgd = tk.Label(window, text=acc_sgd)
    display.create_window(670, 250, window=label_sgd)

sgd_pred_numb_crt, sgd_pred_numb_incrt = number_predictions(SGD_train)

######################################            RANDOM_FOREST CLASSIFIER            ###########################################
from sklearn.ensemble import RandomForestClassifier

rand_for_fit = RandomForestClassifier(max_depth=60, n_estimators=20).fit(X_train,y_train)
rand_forest_train = rand_for_fit.predict(X_test)
rand_forest_final_predict = rand_for_fit.predict(test_data_vect)                 ############  Final Random Forest  #############
acc_rand = (metrics.accuracy_score(y_test, rand_forest_train))*100
# print("Accuracy of Random_Forest classifier:",acc_rand)
prec_rec_rf = classification_report(y_test, rand_forest_train)
# print("Precision and Recall:\n", classification_report(y_test, rand_forest_train))

#################################   Saving Output in File    #######################################
dict_id_cuisine_rand_forest = dict_of_models(rand_forest_final_predict)
path_rand = 'output_rand_forest.txt'
file_write(path_rand, dict_id_cuisine_rand_forest)
print("Result written in Output File(rand_forest)")
print("\n********************************************************************************\n")

def rf_btn():
    label_rf = tk.Label(window, text=acc_rand)
    display.create_window(250, 370, window=label_rf)

rf_pred_numb_crt, rf_pred_numb_incrt = number_predictions(rand_forest_train)

def graph():
    x = ["KNN", "Logistic Reg", "Naive Bayes", "SGD", "Random Forest"]
    y = [acc_knn, acc_log, acc_naive, acc_sgd, acc_rand]
    plt.bar(x,y)
    plt.xlabel(' Models For Predicting Cuisines')
    plt.ylabel('Accuracy')
    plt.title(' Whats Cooking? Predicting Cuising') 
    plt.show()

def graph_pred_correct():
    x = ["KNN", "Logistic Reg", "Naive Bayes", "SGD", "Random Forest"]
    y = [knn_pred_numb_crt, log_pred_numb_crt, nb_pred_numb_crt, sgd_pred_numb_crt, rf_pred_numb_crt]
    y1 = [knn_pred_numb_incrt, log_pred_numb_incrt, nb_pred_numb_incrt, sgd_pred_numb_incrt, rf_pred_numb_incrt]
    plt.bar(x,y)
    plt.bar(x,y1)
    plt.ylim(0,3500)

    plt.xlabel(' Models For Predicting Cuisines')
    plt.ylabel('Prediction')
    plt.title(' Whats Cooking? Predicting Cuising') 
    plt.legend(["Correct", "Incorrect"], loc='upper right', bbox_to_anchor=(1, 1))
    plt.show()

############################################################################################################################################
label_head_knn = tk.Label(window, text=' KNN Model ', font=('Arial', 13))   
display.create_window(70, 180, window=label_head_knn)

button_knn = tk.Button(text='Show', width=8, command=knn_btn)
display.create_window(70, 220, window=button_knn)
############################################################################################################################################
label_head_log = tk.Label(window, text=' Logistic Regression ', font=('Arial', 13))
display.create_window(270, 180, window=label_head_log)

button_log = tk.Button(text='Show', width=8, command=log_btn)
display.create_window(270, 220, window=button_log)
############################################################################################################################################
label_head_nb = tk.Label(window, text=' Naive Bayes(Multinomial) ', font=('Arial', 13)) 
display.create_window(470, 180, window=label_head_nb)

button_nb = tk.Button(text='Show', width=8, command=nb_btn)
display.create_window(470, 220, window=button_nb)
############################################################################################################################################
label_head_sgd = tk.Label(window, text=' SGD Classifier ', font=('Arial', 13))  
display.create_window(670, 180, window=label_head_sgd)

button_sgd = tk.Button(text='Show', width=8, command=sgd_btn)
display.create_window(670, 220, window=button_sgd)
############################################################################################################################################
label_head_rf = tk.Label(window, text=' Random Forest ', font=('Arial', 13))    
display.create_window(250, 310, window=label_head_rf)

button_rf = tk.Button(text='Show', width=8, command=rf_btn)
display.create_window(250, 340, window=button_rf)
############################################################################################################################################
label_head_graph = tk.Label(window, text=' Accuracy Graph of models ', font=('Arial', 13))     
display.create_window(450, 310, window=label_head_graph)

button_graph = tk.Button(text='Show', width=8, command=graph)
display.create_window(450, 340, window=button_graph)
############################################################################################################################################
label_head_graph_pred_correct = tk.Label(window, text=' Correct Prediction Graph of models ', font=('Arial', 13))     
display.create_window(350, 400, window=label_head_graph_pred_correct)

button_graph_pred_correct = tk.Button(text='Show', width=8, command=graph_pred_correct)
display.create_window(350, 430, window=button_graph_pred_correct)


def createNewWindow():
    # global img
    newWindow = tk.Toplevel(window)

    display = tk.Canvas(newWindow, width=800, height=500)
    display.pack()
    
    label_knn_new = tk.Label(newWindow, text=' KNN Model ', font=('Arial', 13))
    label_knn_new.place(relx=0.2,rely=0)

    label_knn = tk.Label(newWindow, text=prec_rec_knn)                                       
    display.create_window(70, 250, window=label_knn)
###############################################################################################################################################
    label_head_log = tk.Label(newWindow, text=' Logistic Regression ', font=('Arial', 13))   
    label_head_log.place(relx=0.36,rely=0)

    label_log = tk.Label(newWindow, text=prec_rec_log)                                       
    display.create_window(270, 250, window=label_log)
###############################################################################################################################################
    label_head_nb = tk.Label(newWindow, text=' Naive Bayes ', font=('Arial', 13))     
    label_head_nb.place(relx=0.5,rely=0)

    label_nb = tk.Label(newWindow, text=prec_rec_nb)                                  
    display.create_window(470, 250, window=label_nb)
###############################################################################################################################################
    label_head_sgd = tk.Label(newWindow, text=' SGD Classifier ', font=('Arial', 13)) 
    label_head_sgd.place(relx=0.65,rely=0)

    label_sgd = tk.Label(newWindow, text=prec_rec_sgd)                                
    display.create_window(670, 250, window=label_sgd)
###############################################################################################################################################
    label_head_rf = tk.Label(newWindow, text=' Random Forest ', font=('Arial', 13))   
    label_head_rf.place(relx=0.8,rely=0)

    label_rf = tk.Label(newWindow, text=prec_rec_rf)                                  
    display.create_window(870, 250, window=label_rf)
###############################################################################################################################################

buttonExample = tk.Button(window, text="Show Classification Report", command=createNewWindow)
buttonExample.pack()

window.mainloop()