# --------------------------------------------------------------------------------------
# Full pipeline for applying Leading Eigenvector Dynamics Analysis (LEiDA) to AD data
# using pyLEiDA: https://github.com/PSYMARKER/leida-python
#
# By Gustavo Patow
#
# note: start by configuring this!!!
# --------------------------------------------------------------------------------------
import os
import csv
import random
import numpy as np
import h5py

# --------------------------------------------------------------------------
# functions to select which subjects to process
# --------------------------------------------------------------------------
# ---------------- load a previously saved list
def loadSubjectList(path):
    subjects = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            subjects.append(int(row[0]))
    return subjects


# ---------------- save a freshly created list
def saveSelectedSubjects(path, subj):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for s in subj:
            writer.writerow([s])


# ---------------- fix subset of subjects to sample
def selectSubjects(selectedSubjectsF, maxSubj, numSampleSubj, excluded, forceRecompute=False):
    if not os.path.isfile(selectedSubjectsF) or forceRecompute:  # if we did not already select a list...
        listIDs = random.sample(range(0, maxSubj), numSampleSubj)
        while excluded & set(listIDs):
            listIDs = random.sample(range(0, maxSubj), numSampleSubj)
        saveSelectedSubjects(selectedSubjectsF, listIDs)
    else:  # if we did, load it!
        listIDs = loadSubjectList(selectedSubjectsF)
    # ---------------- OK, let's proceed
    return listIDs


def handle_nan_values(data):
    nan_indices = np.isnan(data)
    non_nan_indices = ~nan_indices
    mean_non_nan = np.mean(data[non_nan_indices])
    data[nan_indices] = mean_non_nan
    print(f'Nan {nan_indices} substituted')
    return data

# --------------------------------------------------------------------------
# functions to load fMRI data for certain subjects
# --------------------------------------------------------------------------
def read_matlab_h5py(filename, excluded):
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        # print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        # a_group_key = list(f.keys())[0]
        # get the object type for a_group_key: usually group or dataset
        # print(type(f['subjects_idxs']))
        # If a_group_key is a dataset name,
        # this gets the dataset values and returns as a list
        # data = list(f[a_group_key])
        # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]  # returns as a h5py dataset object
        # ds_arr = f[a_group_key][()]  # returns as a numpy array

        all_fMRI = {}
        subjects = list(f['subject'])
        for pos, subj in enumerate(subjects):
            print(f'reading subject {pos}')
            group = f[subj[0]]
            try:
                dbs80ts = np.array(group['dbs80ts'])
                #dbs80ts = handle_nan_values(dbs80ts)  # Handle NaN values
                all_fMRI[pos] = dbs80ts.T
            except:
                print(f'ignoring register {subj} at {pos}')
                excluded.add(pos)

    return all_fMRI, excluded


def testSubjectData(fMRI_path, excluded):
    print(f'testing {fMRI_path}')
    fMRIs, excluded = read_matlab_h5py(fMRI_path, excluded)  # now, we re only interested in the excluded list
    return excluded


def loadSubjectsData(fMRI_path, listIDs):
    print(f'Loading {fMRI_path}')
    fMRIs, excluded = read_matlab_h5py(fMRI_path, set())   # ignore the excluded list
    res = {}  # np.zeros((numSampleSubj, nNodes, Tmax))
    for pos, s in enumerate(listIDs):
        res[pos] = fMRIs[s]
    return res


# --------------------------------------------------------------------------
# Paths and subject selection
# --------------------------------------------------------------------------
maxSubjects = 1003
numSampleSubjects = 20  # 20 for exploring the data
tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'REST1', 'SOCIAL', 'WM']

base_path = 'C:/Users/adria/Documents/HCP-DataBase/DataHCP80/'
fMRI_path = base_path + 'hcp1003_{}_LR_dbs80.mat'
SC_path = base_path + 'SC_dbs80HARDIFULL.mat'

save_path = 'C:/Users/adria/Documents/HCP-DataBase/DataHCP80/Data_produced_Whole_Brain/'
selectedSubjectsFile = save_path + f'selected_{numSampleSubjects}.txt'

timeseries = {}
excluded = set()
if not os.path.isfile(selectedSubjectsFile):
    for task in tasks:
        print(f'----------- Checking: {task} --------------')
        fMRI_task_path = fMRI_path.format(task)
        excluded = testSubjectData(fMRI_task_path, excluded)

listSelectedIDs = selectSubjects(selectedSubjectsFile, maxSubjects, numSampleSubjects, excluded)

for task in tasks:
    print(f'----------- Processing: {task} --------------')
    fMRI_task_path = fMRI_path.format(task)
    timeseries[task] = loadSubjectsData(fMRI_task_path, listSelectedIDs)
    primer_subjecte = list(timeseries[task].values())[0]  # Obtenim les dades fMRI del primer subjecte
    mida_subjecte = primer_subjecte.shape  # Mida del subjecte (dimensió de les dades fMRI)
    print("La mida dels subjectes per a la tasca", task, "és:", mida_subjecte)

# all_fMRI = {s: d for s,d in enumerate(timeseries)}
# numSubj, nNodes, Tmax = timeseries.shape  # actually, 80, 1200
N = 80  # we are using the dbs80 format!

print('Done!!!')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
