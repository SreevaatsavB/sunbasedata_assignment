import streamlit as st
import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
from pickle import load

#####################################################################################################

# TO RUN :- 
# streamlit run Home.py

#####################################################################################################


class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(8, 64)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
    
class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(6, 64)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

st.header("Customer Churn prediction")

st.write("### Choose the options and fill in the information to check if the customer would quit the service he/she is interseted in.")


gend_type = st.selectbox("Gender: ",
                     ['Male', 'Female'])

gend_dict = {'Female': 1, 'Male': 2}
gend_type = gend_dict[gend_type]


loc = st.selectbox("Choose a location of the customer: ",
                     ['Miami', 'New York', 'Los Angeles', 'Chicago','Houston'])
loc_dict = {'Miami': 1, 'New York': 2, 'Los Angeles': 3, 'Chicago': 4, 'Houston': 5}
loc = loc_dict[loc]


age = st.number_input('Enter the age of the customer', min_value = 0, max_value = 100)

monthly_bill = st.number_input('Enter the monthly bill of the customer', min_value = 0)

total_usage_gb = st.number_input('Enter the total usage (in GB) of the customer', min_value = 0)

subscription_length_months = st.number_input('Enter no.of subscription months of the customer', min_value = 0)



if (st.button('Get an estimate!')):
  user_inp = [age, gend_type, loc, subscription_length_months, monthly_bill,  total_usage_gb]
  inp_df = pd.DataFrame(user_inp).T

  print(inp_df)

  inp_df.columns = ['Age', 'Gender', 'Location',
       'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
  
  inp_df['Age_group'] = None
  inp_df.loc[inp_df['Age']<24,'Age_group']='<24'
  inp_df.loc[(inp_df['Age']>=24) & (inp_df['Age']<36),'Age_group']='24-36'
  inp_df.loc[(inp_df['Age']>=36) & (inp_df['Age']<48),'Age_group']='36-48'
  inp_df.loc[(inp_df['Age']>=48) & (inp_df['Age']<60),'Age_group']='48-60'
  inp_df.loc[inp_df['Age']>=60,'Age_group']='60+'

  inp_df['Total_Usage_GB_cat'] = None
  inp_df.loc[inp_df['Total_Usage_GB']<= 100,'Total_Usage_GB_cat']='0-100'
  inp_df.loc[(inp_df['Total_Usage_GB']> 100) & (inp_df['Total_Usage_GB']<=200),'Total_Usage_GB_cat']='100-200'
  inp_df.loc[(inp_df['Total_Usage_GB']> 200) & (inp_df['Total_Usage_GB']<=300),'Total_Usage_GB_cat']='200-300'
  inp_df.loc[(inp_df['Total_Usage_GB']> 300) & (inp_df['Total_Usage_GB']<=400),'Total_Usage_GB_cat']='300-400'
  inp_df.loc[(inp_df['Total_Usage_GB']> 400) & (inp_df['Total_Usage_GB']<=500),'Total_Usage_GB_cat']='400-500'
  inp_df.loc[(inp_df['Total_Usage_GB']> 500),'Total_Usage_GB_cat']='500+'

  inp_df['Monthly_Bill_cat'] = None
  inp_df.loc[(inp_df['Monthly_Bill']>= 0) & (inp_df['Monthly_Bill']<40),'Monthly_Bill_cat']='<40'
  inp_df.loc[(inp_df['Monthly_Bill']>= 40) & (inp_df['Monthly_Bill']< 60),'Monthly_Bill_cat']='40-60'
  inp_df.loc[(inp_df['Monthly_Bill']>= 60) & (inp_df['Monthly_Bill']< 80),'Monthly_Bill_cat']='60-80'
  inp_df.loc[(inp_df['Monthly_Bill']>= 80),'Monthly_Bill_cat']='80+'

  inp_df['SLM_cat'] = None
  inp_df.loc[(inp_df['Subscription_Length_Months']>= 0) & (inp_df['Subscription_Length_Months']<5),'SLM_cat']='<5'
  inp_df.loc[(inp_df['Subscription_Length_Months']>= 5) & (inp_df['Subscription_Length_Months']< 10),'SLM_cat']='5-10'
  inp_df.loc[(inp_df['Subscription_Length_Months']>= 10) & (inp_df['Subscription_Length_Months']< 15),'SLM_cat']='10-15'
  inp_df.loc[(inp_df['Subscription_Length_Months']>= 15) & (inp_df['Subscription_Length_Months']< 20),'SLM_cat']='15-20'
  inp_df.loc[(inp_df['Subscription_Length_Months']>= 20),'SLM_cat']='20+'

  age_group_dict = {'<24': 1, '36-48': 2, '48-60': 3, '24-36': 4, '60+': 5}
  tot_usg_dict = {'200-300': 1, '100-200': 2, '300-400': 3, '400-500': 4, '0-100': 5}
  monthbill_dict = {'80+': 1, '<40': 2, '60-80': 3, '40-60': 4}
  slm_dict = {'5-10': 1, '20+': 2, '10-15': 3, '15-20': 4, '<5': 5}

  def age_group(s):
    return age_group_dict[s]
  
  def tot_usg(s):
    return tot_usg_dict[s]
  
  def monthbill(s):
    return monthbill_dict[s]
  
  def slm(s):
    return slm_dict[s]
  
  inp_df["Age_group"] = inp_df["Age_group"].apply(age_group)
  inp_df["Monthly_Bill_cat"] = inp_df["Monthly_Bill_cat"].apply(monthbill)
  inp_df["Total_Usage_GB_cat"] = inp_df["Total_Usage_GB_cat"].apply(tot_usg)
  inp_df["SLM_cat"] = inp_df["SLM_cat"].apply(slm)

  df_continous = inp_df[["Subscription_Length_Months", "Monthly_Bill", "Total_Usage_GB", "Age"]]

  df_catergorical = inp_df[['Age_group',
    'Monthly_Bill_cat',
    'Gender',
    'Total_Usage_GB_cat',
    'Location',
    'SLM_cat']]
  

  df_continous["Age**2"] = df_continous["Age"]**2
  df_continous["SLM**2"] = df_continous["Subscription_Length_Months"]**2
  df_continous["use_by_slm"] = df_continous["Total_Usage_GB"]/df_continous["Subscription_Length_Months"]
  df_continous["bill_by_slm"] = df_continous["Monthly_Bill"]/df_continous["Subscription_Length_Months"]



  scaler_num  = load(open('scaler_numeric.pkl', 'rb'))

  x_test_num = np.array(df_continous)
  x_test_num = scaler_num.transform(x_test_num)


  model_state_dict = torch.load('customer_churn_model.pth')
  model_temp = Model1()  
  model_temp.load_state_dict(model_state_dict)
  model_temp.eval()  
  x_test_num_sc_tensor = torch.tensor(x_test_num, dtype=torch.float32)
  y_pred_m1 = model_temp(x_test_num_sc_tensor)
  y_pred_numpy = (y_pred_m1 > 0.5).squeeze().cpu().detach().numpy()

#####################################################################################################

  rc_cat = pickle.load(open('rc_cat.pkl', 'rb'))
  y_pred_rccat = rc_cat.predict(df_catergorical)


#####################################################################################################

  x_test_cat1 = df_catergorical[["Gender", "Location"]]
  x_test_continous1 = df_continous[["Subscription_Length_Months", "Monthly_Bill", "Age", "Total_Usage_GB"]]
  df_test_new = pd.concat([x_test_continous1, x_test_cat1],axis =  1)   

  scaler_all  = load(open('scaler_all.pkl', 'rb'))

  x_test_all_sc = scaler_all.transform(df_test_new)


  model_state_dict2 = torch.load('customer_churn_model2.pth')
  model_2 = Model3()  
  model_2.load_state_dict(model_state_dict2)
  model_2.eval()  

  x_test_all_sc_tensor = torch.tensor(x_test_all_sc, dtype=torch.float32)

  y_pred_m2 = model_2(x_test_all_sc_tensor)
  y_pred_numpy2 = (y_pred_m2 > 0.5).squeeze().cpu().detach().numpy()
  

  ops = [int(y_pred_numpy), int(y_pred_numpy2), int(y_pred_rccat)]
  

  if (sum(ops)) > 1:
     op = "Quit"   

  else:
     op = "Continue"

      
  st.write("## The customer is likely to {}.".format(op))

