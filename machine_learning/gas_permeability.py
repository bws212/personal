import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Draw
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR


def clean_df(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['SMILES'])
    return df


def calculate_fingerprint(mol):
    fps = []  # Initialize empty array to store fingerprint
    arr = np.zeros((1,))  # Zero vector to create fingerprint
    # Defaults are minPath=1,maxPath=7,fpSize=2048,bitsPerHash=2,useHs=True,tgtDensity=0.0 defaults
    fp_temp = Chem.RDKFingerprint(mol, minSize=128)  # Calculate fingerprint
    DataStructs.ConvertToNumpyArray(fp_temp, arr)  # Swap fingerprint values into zero array
    fps.append(arr)  # Store fingerprint into array
    return fps


def get_MACCS(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)  # Generate MACCS key vector
    fp = fp.ToBitString()
    fp = np.fromiter(fp, float)  # This line here is just to save a headache or two
    return fp


def lin_reg(data):
    X_train, X_test, y_train, y_test = train_test_split(fp, np.log10(data), train_size=0.8, random_state=0)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    print('Training score: ' + str(linreg.score(X_train, y_train)))
    print('Testing score: ' + str(linreg.score(X_test, y_test)))
    y_pred = linreg.predict(X_test)
    print(len(y_pred))
    print(len(y_test))
    return y_pred, y_test


def svm_reg(data):
    X_train, X_test, y_train, y_test = train_test_split(fp, np.log10(data), train_size=0.8, random_state=0)
    svm = SVR(kernel='rbf')
    svm.fit(X_train, y_train)
    print('Training score (SVM): ' + str(svm.score(X_train, y_train)))
    print('Testing score (SVM): ' + str(svm.score(X_test, y_test)))
    y_pred = svm.predict(X_test)
    return y_pred, y_test


def rf_reg(data):
    X_train, X_test, y_train, y_test = train_test_split(fp, np.log10(data), train_size=0.8, random_state=0)
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    print('Training score (Random Forest): ' + str(rf.score(X_train, y_train)))
    print('Testing score (Random Forest): ' + str(rf.score(X_test, y_test)))
    y_pred = rf.predict(X_test)
    return y_pred, y_test


def gb_reg(data):
    X_train, X_test, y_train, y_test = train_test_split(fp, np.log10(data), train_size=0.8, random_state=0)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=0)
    gb.fit(X_train, y_train)
    print('Training score (Gradient Boosting): ' + str(gb.score(X_train, y_train)))
    print('Testing score (Gradient Boosting): ' + str(gb.score(X_test, y_test)))
    y_pred = gb.predict(X_test)
    return y_pred, y_test


def plotting(x, y, model):
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color='blue')
    plt.xlabel('$y_{test}$')
    plt.ylabel('$y_{pred}$')
    plt.title(f"Model is {model}")
    plt.grid(True)
    errors = x - y
    plt.figure(figsize=(8, 8))
    plt.hist(errors, bins=20, color='blue', density=True)
    plt.xlabel('Error [$log_{10}$ space]')
    plt.ylabel('Frequency')
    plt.title(f'Linear Model Errors for {model}')
    plt.show()
    return


if __name__ == "__main__":
    data = pd.read_excel('Perm_Data.xlsx')
    print(data.head(5))
    He_data = data.loc[:,["Polymer", "SMILES", "He"]]
    H2_data = data.loc[:,["Polymer", "SMILES", "H2"]]
    CO2_data = data.loc[:,["Polymer", "SMILES", "CO2"]]
    O2_data = data.loc[:,["Polymer", "SMILES", "O2"]]
    N2_data = data.loc[:,["Polymer", "SMILES", "N2"]]
    CH4_data = data.loc[:,["Polymer", "SMILES", "CH4"]]
    CH4_data.head(5)
    He_data = clean_df(He_data)
    H2_data = clean_df(H2_data)
    CO2_data = clean_df(CO2_data)
    O2_data = clean_df(O2_data)
    N2_data = clean_df(N2_data)
    CH4_data = clean_df(CH4_data)
    print(O2_data.head(10))
    mol = Chem.MolFromSmiles(O2_data.iloc[0, 1])
    img = Draw.MolToImage(mol)
    plt.imshow(img)
    plt.axis('off')  # Turn off axis
    plt.show()
    for atom in mol.GetAtoms():
        print(atom.GetAtomicNum())

    test_finger = calculate_fingerprint(mol)
    print(f" the test fingerpring is {test_finger}, and the length is {len(test_finger[0])}")
    test_MACCS = get_MACCS(mol)
    print(f" the test MACCS is {test_MACCS}, and the length is {len(test_MACCS)}")

    ML_data = O2_data.copy(deep=True)
    mol_init = Chem.MolFromSmiles(ML_data.iloc[0, 1])
    fp = calculate_fingerprint(mol_init)
    for i in range(1, len(ML_data)):
        try:
            mol_temp = Chem.MolFromSmiles(ML_data.iloc[i, 1])
            fp_temp = calculate_fingerprint(mol_temp)
        except:
            print("Error getting fingerprint")
        fp = np.vstack([fp, fp_temp])
        if i < 10 or i >= len(ML_data) - 10:
            print(i)
        elif i == 10:
            print("...")
        elif i == len(ML_data) - 11:
            print("...")

    print(len(fp))
    print('--------------------------------------------------')
    print(fp)
    perms = ML_data['O2'].values
    print(len(perms))
    print('--------------------------------------------------')
    print(perms)
    y_pred, y_test = lin_reg(perms)
    plotting(y_test, y_pred, "Linear Regression")

    svm_y_pred, svm_y_test = svm_reg(perms)
    plotting(svm_y_test, svm_y_pred, "SVM")

    rf_y_pred, rf_y_test = rf_reg(perms)
    plotting(rf_y_test, rf_y_pred, "Random Forest")

    gb_y_pred, gb_y_test = gb_reg(perms)
    plotting(gb_y_test, gb_y_pred, "Gradient Boosting")
    #Random Forest performed the best

