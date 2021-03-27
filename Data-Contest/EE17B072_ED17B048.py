#!/usr/bin/env python
# coding: utf-8

# In[1]:


##########        Data contest       ##########
#                                             #
###### EE17B072 - Team name - Brainstorm ######

import numpy as np
import os
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import random
import lightgbm as lgb 
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="http")
import statistics 
from statistics import mode 
from sklearn.cluster import KMeans
import csv
import xgboost as xgb


# In[56]:


# reading the csv files

with open('bikers.csv','r') as f:
    bikers = pd.read_csv(
        f, dtype={"biker_id": str, "language_id": str, "location_id": str, "bornIn": str, "gender": str, "member_since": str, "area": str, "time_zone": float})
with open('bikers_network.csv','r') as f:
    bikers_network = pd.read_csv(
        f, dtype={"biker_id": str, "friends": str})
with open('tours.csv','r') as f:
    tours = pd.read_csv(
        f, dtype={"tour_id": str, "biker_id": str, "tour_date": str, "city": str, "state": str, "pincode": str, "country": str, "latitude": str, "longitude": str, "w1": int, "w2": int, "w3": int, "w4": int, "w5": int, "w6": int, "w7": int, "w8": int, "w9": int, "w10": int, "w11": int, "w12": int, "w13": int, "w14": int, "w15": int, "w16": int, "w17": int, "w18": int, "w19": int, "w20": int})
with open('tour_convoy.csv','r') as f:
    tour_convoy = pd.read_csv(
        f, dtype={"tour_id": str, "going": str, "maybe": str, "invited": str, "not_going": str})
with open('train.csv','r') as f:
    train = pd.read_csv(
        f, dtype={"biker_id": str, "tour_id": str, "invited": str, "timestamp": str ,"like": str, "dislike": str})
with open('test.csv','r') as f:
    test = pd.read_csv(
      f, dtype={"biker_id": str, "tour_id": str, "invited": str, "timestamp": str})


# In[57]:


# cleaning and organizing bikers data

bikers["bornIn"].replace(['None'], np.nan, inplace=True)
bikers["bornIn"].replace(['23-May'], np.nan, inplace=True)
bikers["bornIn"].replace(['16-Mar'], np.nan, inplace=True)
bikers.replace(['None'],np.nan,inplace=True)
bikers.member_since.replace(['--None'],['0'],inplace=True)
bikers.member_since.replace([np.nan],['0'],inplace=True)
bikers_network.replace(['None'],np.nan,inplace=True)
bornIn_int = bikers["bornIn"].values.astype(str).astype(float)
time_zone_int = bikers["time_zone"].values
biker_id_labels, biker_id_levels = pd.factorize(bikers.biker_id)
language_labels, levels = pd.factorize(bikers.language_id)
location_labels, levels = pd.factorize(bikers.location_id)
gender_labels, levels = pd.factorize(bikers.gender)
area_labels, levels = pd.factorize(bikers.area)
area_array = bikers.area.values
   
biker_information = np.vstack([time_zone_int,bornIn_int,np.asanyarray(language_labels),np.asanyarray(area_labels),np.asanyarray(location_labels),np.asanyarray(gender_labels)])


# In[58]:


# cleaning and organizing train and test data

tours_in_train_labels, tours_in_train_levels = pd.factorize(train.tour_id)
tours_in_test_labels, tours_in_test_levels = pd.factorize(test.tour_id)
bikers_in_train_labels, bikers_in_train_levels = pd.factorize(train.biker_id)
bikers_in_test_labels, bikers_in_test_levels = pd.factorize(test.biker_id)
train.replace(['None'],np.nan,inplace=True)
test.replace(['None'],np.nan,inplace=True)

tours.tour_date.replace(['--None'],['0'],inplace=True)
tours.tour_date.replace([np.nan],['0'],inplace=True)
tour_convoy.replace([np.nan],0,inplace=True)


# In[5]:


# fixing missing data for bikers and creating latitude and longitude lists for bikers in train and test files

latitude_train_list = []
longitude_train_list = []

latitude_test_list = []
longitude_test_list = []

for i in tqdm(range(len(bikers_in_train_levels))):
    bikerID = bikers_in_train_levels[i]
    k = bikers.biker_id==bikerID
    if area_array[k][0] is np.nan:
        loc = bikers.location_id.values[k][0]
    else:
        loc = area_array[k][0]
    location = geolocator.geocode(loc,timeout=500)
    if location is None:
        latitude = np.nan
        longitude = np.nan
    else:
        latitude = location.latitude
        longitude = location.longitude 
        
    latitude_train_list.append(latitude)
    longitude_train_list.append(longitude)
    
for i in tqdm(range(len(bikers_in_test_levels))):
    bikerID = bikers_in_test_levels[i]
    k = bikers.biker_id==bikerID
    if area_array[k][0] is np.nan:
        loc = bikers.location_id.values[k][0]
    else:
        loc = area_array[k][0]
    location = geolocator.geocode(loc,timeout=500)
    if location is None:
        latitude = np.nan
        longitude = np.nan
    else:
        latitude = location.latitude
        longitude = location.longitude 
        
    latitude_test_list.append(latitude)
    longitude_test_list.append(longitude)


# In[6]:


longitude_test_list = np.asanyarray(longitude_test_list)
latitude_test_list = np.asanyarray(latitude_test_list)
longitude_train_list = np.asanyarray(longitude_train_list)
latitude_train_list = np.asanyarray(latitude_train_list)


# In[7]:


# extracting latitudes and longitudes for tours in train and test files

lat_array = tours.latitude.values.astype(str).astype(float)
long_array = tours.longitude.values.astype(str).astype(float)
latitude_train_tours =[]
longitude_train_tours =[]

for i in tqdm(range(len(tours_in_train_levels))):
    tourID = tours_in_train_levels[i]
    k = tours.tour_id==tourID
    lat = lat_array[k]
    long = long_array[k]
    if np.isnan(lat[0]) and np.isnan(long[0]):
        if tours.city.values[k][0] is not np.nan:
            lat = geolocator.geocode(tours.city.values[k][0],timeout=500).latitude
            long = geolocator.geocode(tours.city.values[k][0],timeout=500).longitude
        else:
            if tours.state.values[k][0] is not np.nan:
                lat = geolocator.geocode(tours.state.values[k][0],timeout=500).latitude
                long = geolocator.geocode(tours.state.values[k][0],timeout=500).longitude
            else:
                if tours.country.values[k][0] is not np.nan:
                    lat = geolocator.geocode(tours.country.values[k][0],timeout=500).latitude
                    long = geolocator.geocode(tours.country.values[k][0],timeout=500).longitude
                else:
                    lat = lat[0]
                    long = long[0]
    else:
        lat = lat[0]
        long = long[0]
        
    latitude_train_tours.append(lat)
    longitude_train_tours.append(long)
    
latitude_test_tours =[]
longitude_test_tours =[]

for i in tqdm(range(len(tours_in_test_levels))):
    tourID = tours_in_test_levels[i]
    k = tours.tour_id==tourID
    lat = lat_array[k]
    long = long_array[k]
    if np.isnan(lat[0]) and np.isnan(long[0]):
        if tours.city.values[k][0] is not np.nan:
            lat = geolocator.geocode(tours.city.values[k][0],timeout=500).latitude
            long = geolocator.geocode(tours.city.values[k][0],timeout=500).longitude
        else:
            if tours.state.values[k][0] is not np.nan:
                lat = geolocator.geocode(tours.state.values[k][0],timeout=500).latitude
                long = geolocator.geocode(tours.state.values[k][0],timeout=500).longitude
            else:
                if tours.country.values[k][0] is not np.nan:
                    lat = geolocator.geocode(tours.country.values[k][0],timeout=500).latitude
                    long = geolocator.geocode(tours.country.values[k][0],timeout=500).longitude
                else:
                    lat = lat[0]
                    long = long[0]
    else:
        lat = lat[0]
        long = long[0]
        
    latitude_test_tours.append(lat)
    longitude_test_tours.append(long)


# In[59]:


# cleaning and organizing tours data

tours.replace(['None'],np.nan,inplace=True)
tour_convoy.replace(['None'],np.nan,inplace=True)
biker_id_labels, levels = pd.factorize(tours.biker_id)
tour_id_labels, levels = pd.factorize(tours.tour_id)
city_labels, levels = pd.factorize(tours.city)
state_labels, levels = pd.factorize(tours.state)
pincode_labels, levels = pd.factorize(tours.pincode)
country_labels, levels = pd.factorize(tours.country)

tour_information = np.vstack([np.asanyarray(biker_id_labels),np.asanyarray(city_labels),np.asanyarray(state_labels),np.asanyarray(pincode_labels),np.asanyarray(country_labels),np.asanyarray(tours.latitude.values.astype(str).astype(float)),np.asanyarray(tours.longitude.values.astype(str).astype(float))])


# In[9]:


def get_int_from_datetime_object(date):
    return 10000*(date.year)+100*(date.month)+date.day

def get_int_from_datetime_object_full(date):
    return 15768000*(date.year)+1296000*(date.month)+43200*(date.day)+3600*(date.hour)+60*(date.minute)+date.second

train_joiningdate = []
train_timestamp = []
r = train.timestamp.values
for i in range(len(bikers_in_train_labels)):
    bikerID = bikers_in_train_levels[bikers_in_train_labels[i]]
    p = bikers.member_since.values[bikers.biker_id==bikerID]
    if p[0]=='0' or p[0]==np.nan:
        train_joiningdate.append(np.nan)
        continue
    date = datetime.strptime(p[0], '%d-%m-%Y')
    train_joiningdate.append(get_int_from_datetime_object(date))
    timestamp = datetime.strptime(r[i], '%d-%m-%Y %H:%M:%S')
    train_timestamp.append(get_int_from_datetime_object_full(timestamp))

test_joiningdate = []
test_timestamp = []
r = test.timestamp.values
for i in range(len(bikers_in_test_labels)):
    bikerID = bikers_in_test_levels[bikers_in_test_labels[i]]
    p = bikers.member_since.values[bikers.biker_id==bikerID]
    
    if p[0]=='0' or p[0]==np.nan:
        test_joiningdate.append(np.nan)
        continue

    date = datetime.strptime(p[0], '%d-%m-%Y')
    test_joiningdate.append(get_int_from_datetime_object(date))
    timestamp = datetime.strptime(r[i], '%d-%m-%Y %H:%M:%S')
    test_timestamp.append(get_int_from_datetime_object_full(timestamp))


# In[61]:


def generate_features():
    
    print("Generating features for the training data and test data.")
        
    X_train = np.zeros([len(tours_in_train_labels),135])
    X_test = np.zeros([len(tours_in_test_labels),135])
    
    Y_train_like = np.asanyarray(train["like"].astype(str).astype(int)).reshape(-1,1)
    Y_train_dislike = np.asanyarray(train["dislike"].astype(str).astype(int)).reshape(-1,1)
    
    for i in tqdm(range(len(tours_in_train_labels))):
        b = bikers_in_train_labels[i]
        tourID = tours_in_train_levels[tours_in_train_labels[i]]
        bikerID = bikers_in_train_levels[b]
        biker_count = np.sum(bikers_in_train_labels==b)
        friend_list = bikers_network.friends[bikers_network.biker_id==bikerID].values[0].strip(" ").split()
        t1 = tour_convoy.tour_id==tourID
        t2 = tours.tour_id==tourID
        w_array = np.asanyarray(tours[t2].values[0][9:110]).reshape(1,101)
        sum_w = np.sum(w_array[0,0:100])
        sum_all_w = np.sum(w_array[0,0:101])
        
        tempvar = tour_convoy.going[t1].values[0]
        if type(tempvar)==int:
            len1 = 0
        else:
            temp_list_going = tempvar.strip(" ").split()
            len1 = len(temp_list_going)
        
        tempvar = tour_convoy.not_going[t1].values[0]
        if type(tempvar)==int:
            len2 = 0
        else:
            temp_list_notgoing = tempvar.strip(" ").split()
            len2 = len(temp_list_notgoing)
    
        tempvar = tour_convoy.maybe[t1].values[0]
        if type(tempvar)==int:
            len3 = 0
        else:
            temp_list_maybe = tempvar.strip(" ").split()
            len3 = len(temp_list_maybe)
        
        tempvar = tour_convoy.invited[t1].values[0]
        if type(tempvar)==int:
            len4 = 1
        else:
            temp_list_invited = tempvar.strip(" ").split()
            len4 = len(temp_list_invited)
            
        ratio1 = len1/len4
        ratio2 = len2/len4
        ratio3 = len3/len4
        
        
        
        no_friends = len(friend_list)
        no_going = len(list(set(friend_list) & set(temp_list_going)))
        no_notgoing = len(list(set(friend_list) & set(temp_list_notgoing)))
        no_maybe = len(list(set(friend_list) & set(temp_list_maybe)))
        no_invited = len(list(set(friend_list) & set(temp_list_invited)))
        
        p = tours.tour_date.values[t2]
        if p[0]=='0' or p[0] is np.nan:
            train_tourdate_temp = np.nan
        else:
            date = datetime.strptime(p[0], '%d-%m-%Y')
            train_tourdate_temp = get_int_from_datetime_object(date)
            
        weekday = date.weekday()
        inv = train.invited.values[i]
        
        biker_info_temp = biker_information[:,bikers.biker_id==bikerID].T
        tour_info_temp = tour_information[:,t2].T
        lat_tour = tour_info_temp[0,5]
        long_tour = tour_info_temp[0,6]
        
        b1 = bikers_in_train_levels==bikerID
        latitude = latitude_train_list[b1]
        longitude = longitude_train_list[b1]

        distance = (latitude-lat_tour)**2+(longitude-long_tour)**2
        time_diff1 = train_timestamp[i] - train_joiningdate[i]
        time_diff2 = train_timestamp[i] - train_tourdate_temp
        time_diff3 = train_joiningdate[i] - train_tourdate_temp
        X_train[i,:] = np.hstack([len1, ratio1, ratio2, ratio3, biker_count,weekday,no_friends,time_diff1,time_diff2,time_diff3,distance,10*inv, biker_info_temp[0],tour_info_temp[0],train_joiningdate[i],train_timestamp[i],train_tourdate_temp,no_going,no_notgoing,no_maybe,no_invited,w_array[0],sum_w,sum_all_w])
        
    print("Training data generated.")
    
    for i in tqdm(range(len(tours_in_test_labels))):        
        
        b = bikers_in_test_labels[i]
        tourID = tours_in_test_levels[tours_in_test_labels[i]]
        bikerID = bikers_in_test_levels[b]
        biker_count = np.sum(bikers_in_test_labels==b)
        
        friend_list = bikers_network.friends[bikers_network.biker_id==bikerID].values[0].strip(" ").split()
        
        t1 = tour_convoy.tour_id==tourID
        t2 = tours.tour_id==tourID
        w_array = np.asanyarray(tours[t2].values[0][9:110]).reshape(1,101)
        sum_w = np.sum(w_array[0,0:100])
        sum_all_w = np.sum(w_array[0,0:101])
        
        tempvar = tour_convoy.going[t1].values[0]
        if type(tempvar)==int:
            len1 = 0
        else:
            temp_list_going = tempvar.strip(" ").split()
            len1 = len(temp_list_going)
        
        tempvar = tour_convoy.not_going[t1].values[0]
        if type(tempvar)==int:
            len2 = 0
        else:
            temp_list_notgoing = tempvar.strip(" ").split()
            len2 = len(temp_list_notgoing)
    
        tempvar = tour_convoy.maybe[t1].values[0]
        if type(tempvar)==int:
            len3 = 0
        else:
            temp_list_maybe = tempvar.strip(" ").split()
            len3 = len(temp_list_maybe)
        
        tempvar = tour_convoy.invited[t1].values[0]
        if type(tempvar)==int:
            len4 = 1
        else:
            temp_list_invited = tempvar.strip(" ").split()
            len4 = len(temp_list_invited)
            
        ratio1 = len1/len4
        ratio2 = len2/len4
        ratio3 = len3/len4
        
        no_friends = len(friend_list)
        no_going = len(list(set(friend_list) & set(temp_list_going)))
        no_notgoing = len(list(set(friend_list) & set(temp_list_notgoing)))
        no_maybe = len(list(set(friend_list) & set(temp_list_maybe)))
        no_invited = len(list(set(friend_list) & set(temp_list_invited)))
        
        p = tours.tour_date.values[t2]
        if p[0]=='0' or p[0] is np.nan:
            test_tourdate_temp = np.nan
        else:
            date = datetime.strptime(p[0], '%d-%m-%Y')
            test_tourdate_temp = get_int_from_datetime_object(date)
        inv = test.invited.values[i]
        weekday = date.weekday()

        
        biker_info_temp = biker_information[:,bikers.biker_id==bikerID].T
        tour_info_temp = tour_information[:,t2].T
        lat_tour = tour_info_temp[0,5]
        long_tour = tour_info_temp[0,6]
        b1 = bikers_in_test_levels==bikerID
        latitude = latitude_test_list[b1]
        longitude = longitude_test_list[b1]
        distance = (latitude-lat_tour)**2+(longitude-long_tour)**2
        
        time_diff1 = test_timestamp[i] - test_joiningdate[i]
        time_diff2 = test_timestamp[i] - test_tourdate_temp
        time_diff3 = test_joiningdate[i] - test_tourdate_temp
        X_test[i,:] = np.hstack([len1, ratio1, ratio2, ratio3, biker_count, weekday, no_friends,time_diff1,time_diff2,time_diff3,distance,10*inv, biker_info_temp[0],tour_info_temp[0],test_joiningdate[i],test_timestamp[i],test_tourdate_temp,no_going,no_notgoing,no_maybe,no_invited,w_array[0],sum_w,sum_all_w])
        
    print("Test data generated.")
    
    
    return X_train, X_test, Y_train_like, Y_train_dislike

            
X_train, X_test, Y_train_like, Y_train_dislike = generate_features()

np.save('X_train_new1.npy', X_train)
np.save('X_test_new1.npy', X_test)
np.save('Y_train1_new1.npy', Y_train_like)
np.save('Y_train2_new1.npy', Y_train_dislike)


# In[63]:


X_train = np.load('X_train_new1.npy')
Y_train_like = np.load('Y_train1_new1.npy')
Y_train_dislike = np.load('Y_train2_new1.npy')
X_test = np.load('X_test_new1.npy')


# In[71]:


def classifier1(X_train, X_test, Y_train_like):
    
    max_leaves = [27]
    learning_rates = [0.056]
    
    num_leaves = random.choice(max_leaves)
    
    learning_rate = random.choice(learning_rates)
    
    
    index = np.random.choice(X_train.shape[0], int(0.8*X_train.shape[0]), replace=False)  
    X_train_new = X_train[index,:]
    Y_train_new = Y_train_like[index]
    
    
    lgb_train = lgb.Dataset(X_train_new, Y_train_new.reshape(len(index),),categorical_feature = [5,14,15,16,17,18,19,20,22])
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'seed':0
    }
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                categorical_feature = [5,14,15,16,17,18,19,20,22])

    pred_tst = gbm.predict(X_test)
        
    return pred_tst
    


# In[72]:


def classifier2(X_train, X_test, Y_train_like):
    
    max_leaves = [27]
    learning_rates = [0.056]
    
    num_leaves = random.choice(max_leaves)
    
    learning_rate = random.choice(learning_rates)
    
    
    index = np.random.choice(X_train.shape[0], int(0.8*X_train.shape[0]), replace=False)  
    X_train_new = X_train[index,:]
    Y_train_new = Y_train_like[index]
    
    
    lgb_train = lgb.Dataset(X_train_new, Y_train_new.reshape(len(index),),categorical_feature = [5,14,15,16,17,18,19,20,22])
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'seed':1
    }
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=150,
                categorical_feature = [5,14,15,16,17,18,19,20,22])

    pred_tst = gbm.predict(X_test)
        
    return pred_tst
    


# In[73]:


def classifier3(X_train, X_test, Y_train_like):
    
    max_leaves = [27]
    learning_rates = [0.056]
    
    num_leaves = random.choice(max_leaves)
    
    learning_rate = random.choice(learning_rates)
    
    
    index = np.random.choice(X_train.shape[0], int(0.8*X_train.shape[0]), replace=False)  
    X_train_new = X_train[index,:]
    Y_train_new = Y_train_like[index]
    
    
    lgb_train = lgb.Dataset(X_train_new, Y_train_new.reshape(len(index),),categorical_feature = [5,14,15,16,17,18,19,20,22])
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'seed':1
    }
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=250,categorical_feature = [5,14,15,16,17,18,19,20,22])

    pred_tst = gbm.predict(X_test)
        
    return pred_tst
    


# In[74]:


def classifier4(X_train, X_test, Y_train_like):
    
    max_leaves = [27]
    learning_rates = [0.056]
    
    num_leaves = random.choice(max_leaves)
    
    learning_rate = random.choice(learning_rates)
    
    
    index = np.random.choice(X_train.shape[0], int(0.8*X_train.shape[0]), replace=False)  
    X_train_new = X_train[index,:]
    Y_train_new = Y_train_like[index]
    
    
    lgb_train = lgb.Dataset(X_train_new, Y_train_new.reshape(len(index),),categorical_feature = [5,14,15,16,17,18,19,20,22])
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'seed':0
    }
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=300,categorical_feature = [5,14,15,16,17,18,19,20,22])

    pred_tst = gbm.predict(X_test)
        
    return pred_tst
    


# In[85]:


iters = 50
y_pred_list1 = np.zeros((X_test.shape[0], iters))
y_pred_list2 = np.zeros((X_test.shape[0], iters))
y_pred_list3 = np.zeros((X_test.shape[0], iters))
y_pred_list4 = np.zeros((X_test.shape[0], iters))
y_pred_list = np.zeros((X_test.shape[0], iters))

np.random.seed(42)
pred_test = np.zeros((X_test.shape[0], 1))

for i in tqdm(range(iters)):
    
    y_pred_list1[:,i] = classifier1(X_train, X_test, Y_train_like)
    y_pred_list2[:,i] = classifier2(X_train, X_test, Y_train_like)
    y_pred_list3[:,i] = classifier3(X_train, X_test, Y_train_like)
    y_pred_list4[:,i] = classifier4(X_train, X_test, Y_train_like)
    
y_pred_list = (y_pred_list1+y_pred_list2+y_pred_list3+y_pred_list4)/4
#print(y_pred_list)
pred_test1_temp = np.asarray(np.sum(y_pred_list, axis=1)/iters)


# In[87]:


filename_to_save = "EE17B072_ED17B048_1.csv"

with open(filename_to_save, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['biker_id', 'tour_id'])
    for bikers in range(297):
    
        bikerID = bikers_in_test_levels[bikers]
    
        a = tours_in_test_levels[tours_in_test_labels[(bikers_in_test_levels[bikers_in_test_labels]==bikerID).reshape(2690,)]]
        
        score_temp = pred_test1_temp[(bikers_in_test_levels[bikers_in_test_labels]==bikerID).reshape(2690,)]
        preference_list = [x for _,x in sorted(zip(score_temp,a))]
        preference_list.reverse()

    
        id1 = bikers_in_test_levels[bikers]
        writer.writerow([id1, " ".join(preference_list)])


# In[77]:


def classifier5(X_train, X_test, Y_train_like):
    
    max_leaves = [27]
    learning_rates = [0.056]
    
    num_leaves = random.choice(max_leaves)
    
    learning_rate = random.choice(learning_rates)
    
    
    index = np.random.choice(X_train.shape[0], int(0.9*X_train.shape[0]), replace=False)  
    X_train_new = X_train[index,:]
    Y_train_new = Y_train_like[index]
    
    
    lgb_train = lgb.Dataset(X_train_new, Y_train_new.reshape(len(index),),categorical_feature = [5,14,15,16,17,18,19,20,22])
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'seed':0
    }
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,categorical_feature = [5,14,15,16,17,18,19,20,22])

    pred_tst = gbm.predict(X_test)
        
    return pred_tst
    


# In[78]:


def classifier6(X_train, X_test, Y_train_like):
    
    max_leaves = [27]
    learning_rates = [0.056]
    
    num_leaves = random.choice(max_leaves)
    
    learning_rate = random.choice(learning_rates)
    
    
    index = np.random.choice(X_train.shape[0], int(0.9*X_train.shape[0]), replace=False)  
    X_train_new = X_train[index,:]
    Y_train_new = Y_train_like[index]
    
    
    lgb_train = lgb.Dataset(X_train_new, Y_train_new.reshape(len(index),),categorical_feature = [5,14,15,16,17,18,19,20,22])
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'seed':1
    }
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=150,categorical_feature = [5,14,15,16,17,18,19,20,22])

    pred_tst = gbm.predict(X_test)
        
    return pred_tst
    


# In[79]:


def classifier7(X_train, X_test, Y_train_like):
    
    max_leaves = [27]
    learning_rates = [0.056]
    
    num_leaves = random.choice(max_leaves)
    
    learning_rate = random.choice(learning_rates)
    
    
    index = np.random.choice(X_train.shape[0], int(0.8*X_train.shape[0]), replace=False)  
    X_train_new = X_train[index,:]
    Y_train_new = Y_train_like[index]
    
    
    lgb_train = lgb.Dataset(X_train_new, Y_train_new.reshape(len(index),),categorical_feature = [5,14,15,16,17,18,19,20,22])
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'seed':1
    }
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=250,categorical_feature = [5,14,15,16,17,18,19,20,22])

    pred_tst = gbm.predict(X_test)
        
    return pred_tst
    


# In[80]:


def classifier8(X_train, X_test, Y_train_like):
    
    max_leaves = [27]
    learning_rates = [0.056]
    
    num_leaves = random.choice(max_leaves)
    
    learning_rate = random.choice(learning_rates)
    
    
    index = np.random.choice(X_train.shape[0], int(0.8*X_train.shape[0]), replace=False)  
    X_train_new = X_train[index,:]
    Y_train_new = Y_train_like[index]
    
    
    lgb_train = lgb.Dataset(X_train_new, Y_train_new.reshape(len(index),),categorical_feature = [5,14,15,16,17,18,19,20,22])
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'seed':0
    }
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=300,categorical_feature = [5,14,15,16,17,18,19,20,22])

    pred_tst = gbm.predict(X_test)
        
    return pred_tst
    


# In[83]:


iters = 50
y_pred_list1 = np.zeros((X_test.shape[0], iters))
y_pred_list2 = np.zeros((X_test.shape[0], iters))
y_pred_list3 = np.zeros((X_test.shape[0], iters))
y_pred_list4 = np.zeros((X_test.shape[0], iters))
y_pred_list = np.zeros((X_test.shape[0], iters))

np.random.seed(42)
pred_test = np.zeros((X_test.shape[0], 1))

for i in tqdm(range(iters)):
    
    y_pred_list1[:,i] = classifier5(X_train, X_test, Y_train_like)
    y_pred_list2[:,i] = classifier6(X_train, X_test, Y_train_like)
    y_pred_list3[:,i] = classifier7(X_train, X_test, Y_train_like)
    y_pred_list4[:,i] = classifier8(X_train, X_test, Y_train_like)
    
y_pred_list = (y_pred_list1+y_pred_list2+y_pred_list3+y_pred_list4)/4

pred_test2_temp = np.asarray(np.sum(y_pred_list, axis=1)/iters)


# In[84]:


filename_to_save = "EE17B072_ED17B048_2.csv"

with open(filename_to_save, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['biker_id', 'tour_id'])
    for bikers in range(297):
    
        bikerID = bikers_in_test_levels[bikers]
    
        a = tours_in_test_levels[tours_in_test_labels[(bikers_in_test_levels[bikers_in_test_labels]==bikerID).reshape(2690,)]]
        
        score_temp = pred_test2_temp[(bikers_in_test_levels[bikers_in_test_labels]==bikerID).reshape(2690,)]
        preference_list = [x for _,x in sorted(zip(score_temp,a))]
        preference_list.reverse()

    
        id1 = bikers_in_test_levels[bikers]
        writer.writerow([id1, " ".join(preference_list)])


# In[42]:


os.remove('X_train_new1.npy')
os.remove('Y_train1_new1.npy')
os.remove('Y_train2_new1.npy')
os.remove('X_test_new1.npy')

