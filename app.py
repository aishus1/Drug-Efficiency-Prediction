import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
import pickle
import requests
import json
from sklearn.preprocessing import StandardScaler


# constants
IMG_ADDRESS = "https://drzinph.com/wp-content/uploads/2020/05/image-6.png"
MOLECULAR_DESCRIPTORS = "Molecular descriptors, also known as chemical descriptors or molecular representations, are numerical or symbolic representations of chemical compounds. These descriptors provide a way to quantitatively describe the structural, physical, and chemical properties of molecules. They play a crucial role in various areas of chemistry, including drug discovery, computational chemistry, and chemoinformatics."
COLUMNS = ['MaxAbsEStateIndex', 'MaxEStateIndex', 'MinEStateIndex', 'MolWt',
       'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
       'BCUT2D_MWHI', 'BCUT2D_MRHI', 'BalabanJ', 'BertzCT', 'Chi0',
       'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v',
       'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc',
       'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1',
       'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
       'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
       'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1',
       'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
       'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
       'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3',
       'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
       'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10',
       'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4',
       'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8',
       'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',
       'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6',
       'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'HeavyAtomCount',
       'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
       'NumAliphaticHeterocycles', 'NumAliphaticRings',
       'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
       'NumAromaticRings', 'NumHAcceptors', 'NumHDonors',
       'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
       'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount',
       'MolLogP', 'MolMR', 'fr_Al_OH', 'fr_Ar_N', 'fr_Ar_OH', 'fr_C_O',
       'fr_C_O_noCOO', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_Ndealkylation2',
       'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_aniline',
       'fr_aryl_methyl', 'fr_benzene', 'fr_bicyclic', 'fr_ether',
       'fr_halogen', 'fr_methoxy', 'fr_para_hydroxylation', 'fr_phenol',
       'fr_phenol_noOrthoHbond', 'fr_unbrch_alkane']

# functions
def extract_descriptors(smile: str) -> dict:
    descriptors = {}
    processed_smile = Chem.MolFromSmiles(smile)

    for descriptor_name, descriptor_function in Descriptors._descList:
        try:
            get_descriptor = descriptor_function(processed_smile)
            descriptors[descriptor_name] = get_descriptor
        except Exception as error:
            print(str(error))
            descriptors[descriptor_name] = np.nan

    return descriptors


def get_prediction(data):
  url = 'https://askai.aiclub.world/a2f0a1b6-ee59-422e-bd32-7baf363496a4'
  r = requests.post(url, data=json.dumps(data))
  response = getattr(r,'_content').decode("utf-8")
  response = json.loads(response)
  response = json.loads(response["body"])
  return response["predicted_label"]


@st.cache_resource
def load_model(path):
    with open(path, "rb") as pickle_model:
        return pickle.load(pickle_model)
    
# load the model
normalization_model = load_model("normalize_new_model")

# web app

# title
st.title("Predictions through Chem Descriptor")
# set an image
st.image(IMG_ADDRESS, caption = "Molecular Descriptors")

# tabs
tab1, tab2 = st.tabs(["Predictions 🖥️", "Explanations 📈"])

with tab1:
    st.header("Extract Descriptors")
    # text input 
    smile = st.text_input("Enter the Canonical Smile Notation")

    if smile:
        
        descriptor_dict = extract_descriptors(smile)
        #st.dataframe(pd.DataFrame(descriptor_dict, index=[0]))
        with st.sidebar:
            # set header
            st.header("Extracted Descriptors")
            # set descriptors
            st.json(descriptor_dict)
        
        make_predictions = st.button("Calculate PIC50", type='primary')

        if make_predictions:
            # create a dataframe using user data
            user_data = pd.DataFrame(descriptor_dict, index = [0])

            # select only required columns
            final_data = user_data[COLUMNS]
            get_columns = list(final_data.columns)

            # egt numpy array
            value_array = final_data.iloc[:,:].values

            # make predictions
            get_normalize_values = normalization_model.transform(value_array)

            # new dataframe
            df_new = pd.DataFrame(get_normalize_values, columns=get_columns)
            records = df_new.to_dict('records')
            with st.spinner("Getting Predictions..."):
                response = get_prediction(records[0])
                if response:
                    st.toast("Prediction Completed 🙂")
                    st.subheader("pIC50 of the Canonical Smile is: {}".format(response))

            

with tab2:
    st.header("Explanation on Descriptors")

    st.subheader("What are Molecular Descriptors")
    st.write(MOLECULAR_DESCRIPTORS)







    

